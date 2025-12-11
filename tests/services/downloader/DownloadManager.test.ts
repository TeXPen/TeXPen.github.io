import { describe, it, expect, vi, beforeEach, afterEach, beforeAll, afterAll } from 'vitest';
import { DownloadManager } from '../../../services/downloader/DownloadManager';
import { getPartialDownload } from '../../../services/downloader/db';

// Mock DB
vi.mock('../../../services/downloader/db', () => ({
  getDB: vi.fn(),
  saveChunk: vi.fn(),
  getPartialDownload: vi.fn(),
  clearPartialDownload: vi.fn(),
}));

// Access mocked functions
import { saveChunk, getPartialDownload as mockGetPartial, clearPartialDownload as mockClearPartial } from '../../../services/downloader/db';

describe('DownloadManager', () => {
  let downloadManager: DownloadManager;
  let mockCachePut: any;
  let mockCacheMatch: any;
  let mockCacheDelete: any;
  let OriginalResponse: any;

  beforeAll(() => {
    OriginalResponse = global.Response;
    // Patch Response to handle Blob bodies correctly in jsdom/node environment
    global.Response = class MockResponse extends OriginalResponse {
      private _blobBody: Blob | null = null;
      constructor(body: BodyInit | null, init?: ResponseInit) {
        // Pass null to super if it's a blob to prevent stringification "feature" in jsdom
        super(body instanceof Blob ? null : body, init);
        if (body instanceof Blob) {
          this._blobBody = body;
        }
      }
      async blob() {
        if (this._blobBody) return this._blobBody;
        return super.blob();
      }
      clone() {
        // Create a new instance essentially
        const cloned = new (global.Response as any)(this._blobBody || null, {
          status: this.status,
          statusText: this.statusText,
          headers: this.headers
        });
        return cloned;
      }
    } as any;
  });

  afterAll(() => {
    global.Response = OriginalResponse;
  });

  beforeEach(() => {
    downloadManager = DownloadManager.getInstance();
    vi.clearAllMocks();

    // Mock global fetch
    global.fetch = vi.fn();

    // Mock caches
    mockCachePut = vi.fn().mockResolvedValue(undefined);
    mockCacheMatch = vi.fn().mockResolvedValue(null);
    mockCacheDelete = vi.fn().mockResolvedValue(true);

    (global as any).caches = {
      open: vi.fn().mockResolvedValue({
        match: mockCacheMatch,
        put: mockCachePut,
        delete: mockCacheDelete,
      }),
    };

    // Enable corruption check for tests
    (downloadManager as any).ENABLE_CORRUPTION_CHECK = true;
    (downloadManager as any).isIDBDisabled = false;
  });

  afterEach(() => {
    vi.resetAllMocks();
  });

  it('should download a file, buffer small chunks, and save complete file to cache', async () => {
    const mockUrl = 'https://example.com/model.onnx';
    const mockContent = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const mockStream = new ReadableStream({
      start(controller) {
        controller.enqueue(mockContent.slice(0, 5));
        controller.enqueue(mockContent.slice(5, 10));
        controller.close();
      }
    });

    (global.fetch as any).mockResolvedValue({
      ok: true,
      headers: new Headers({ 'Content-Length': '10', 'Etag': 'test-etag' }),
      body: mockStream,
    });

    await downloadManager.downloadFile(mockUrl);

    // Verify buffering: 10 bytes < 50MB threshold, so saveChunk should only be called ONCE at the end (flush)
    // or if the implementation saves on done.
    expect(saveChunk).toHaveBeenCalledTimes(1);

    // Check call arguments for index 0 (merged)
    expect(saveChunk).toHaveBeenCalledWith(mockUrl, expect.any(Blob), 10, 0, 'test-etag');

    // CRITICAL: Verify cache.put received the full blob
    expect(mockCachePut).toHaveBeenCalledTimes(1);
    const [putUrl, putResponse] = mockCachePut.mock.calls[0];
    expect(putUrl).toBe(mockUrl);

    // Verify response blob size
    const blob = await putResponse.blob();
    expect(blob.size).toBe(10);
    // This assertion failed previously (was 0) which caused the bug.
  });

  it('should resume from partial state and append new chunks', async () => {
    const mockUrl = 'https://example.com/large.bin';
    const existingData = new Uint8Array(50).fill(1);

    // Mock existing partial state
    (mockGetPartial as any).mockResolvedValue({
      url: mockUrl,
      downloadedBytes: 50,
      totalBytes: 100,
      chunks: [new Blob([existingData])], // 50 bytes existing
      chunkCount: 1
    });

    // Mock fetch to return remaining 50 bytes
    const mockStream = new ReadableStream({
      start(controller) {
        controller.enqueue(new Uint8Array(50).fill(2));
        controller.close();
      }
    });

    (global.fetch as any).mockResolvedValue({
      ok: true,
      headers: new Headers({ 'Content-Length': '50', 'Etag': 'test-etag' }),
      status: 206, // Partial Content
      body: mockStream,
    });

    await downloadManager.downloadFile(mockUrl);

    // Should call fetch with Range header
    expect(global.fetch).toHaveBeenCalledWith(mockUrl, expect.objectContaining({
      headers: { 'Range': 'bytes=50-' }
    }));

    // Should save new chunk at index 1 (since 0 existed)
    // It is < 50MB so it will be flushed at the end
    expect(saveChunk).toHaveBeenCalledWith(mockUrl, expect.any(Blob), 100, 1, 'test-etag');

    // Verify final cache put has 100 bytes (50 existing + 50 new)
    expect(mockCachePut).toHaveBeenCalledTimes(1);
    const response = mockCachePut.mock.calls[0][1];
    const blob = await response.blob();
    expect(blob.size).toBe(100);
  });

  it('should detect and heal corrupted (size mismatch) cache', async () => {
    const mockUrl = 'https://example.com/corrupt.file';
    const mockSize = 100;

    // Mock cache returning a BAD response
    // Content-Length says 100, but Blob size is 50 (truncated)
    const badBlob = new Blob([new Uint8Array(50)]);
    const badResponse = new Response(badBlob, {
      headers: { 'Content-Length': '100' }
    });

    mockCacheMatch.mockResolvedValue(badResponse);

    // Prepare fresh download mock
    const mockStream = new ReadableStream({
      start(controller) {
        controller.enqueue(new Uint8Array(100)); // Full correct file
        controller.close();
      }
    });
    (global.fetch as any).mockResolvedValue({
      ok: true,
      headers: new Headers({ 'Content-Length': '100' }),
      body: mockStream,
    });

    await downloadManager.downloadFile(mockUrl);

    // Expect delete to be called for the bad cache
    expect(mockCacheDelete).toHaveBeenCalledWith(mockUrl);

    // Expect re-download (fetch called)
    expect(global.fetch).toHaveBeenCalledTimes(1);
    expect(mockCachePut).toHaveBeenCalledTimes(1);
  });

  it('should detect and heal empty (0-byte) cache', async () => {
    const mockUrl = 'https://example.com/empty.file';

    // Mock cache returning an EMPTY response (the bug case)
    const emptyBlob = new Blob([]);
    const emptyResponse = new Response(emptyBlob, {
      headers: { 'Content-Length': '100' }
    });

    mockCacheMatch.mockResolvedValue(emptyResponse);

    // Mock successful re-download
    const mockStream = new ReadableStream({
      start(controller) {
        controller.enqueue(new Uint8Array(100));
        controller.close();
      }
    });
    (global.fetch as any).mockResolvedValue({
      ok: true,
      headers: new Headers({ 'Content-Length': '100' }),
      body: mockStream,
    });

    await downloadManager.downloadFile(mockUrl);

    // Expect delete
    expect(mockCacheDelete).toHaveBeenCalledWith(mockUrl);
    // Expect re-download
    expect(global.fetch).toHaveBeenCalledTimes(1);
  });
  it('should deduplicate concurrent requests for the same URL', async () => {
    const mockUrl = 'https://example.com/duplicate.file';
    let fetchCallCount = 0;

    // Simulate a slow fetch
    (global.fetch as any).mockImplementation(async () => {
      fetchCallCount++;
      await new Promise(resolve => setTimeout(resolve, 100)); // Delay
      return {
        ok: true,
        headers: new Headers({ 'Content-Length': '10' }),
        body: new ReadableStream({
          start(controller) {
            controller.enqueue(new Uint8Array(10));
            controller.close();
          }
        })
      };
    });

    const p1 = downloadManager.downloadFile(mockUrl);
    const p2 = downloadManager.downloadFile(mockUrl);

    expect(p1).toBe(p2); // Same promise instance

    await Promise.all([p1, p2]);

    expect(fetchCallCount).toBe(1); // Only one network request
  });

  it('should limit concurrent downloads to MAX_CONCURRENT (3)', async () => {
    const urls = ['http://1', 'http://2', 'http://3', 'http://4'];
    let runningCount = 0;
    let maxRunning = 0;

    // Mock fetch to track concurrency
    (global.fetch as any).mockImplementation(async (url: string) => {
      runningCount++;
      maxRunning = Math.max(maxRunning, runningCount);
      await new Promise(resolve => setTimeout(resolve, 50)); // Hold connection
      runningCount--;
      return {
        ok: true,
        headers: new Headers({ 'Content-Length': '10' }),
        body: new ReadableStream({
          start(controller) {
            controller.enqueue(new Uint8Array(10));
            controller.close();
          }
        })
      };
    });

    // Fire 4 requests
    await Promise.all(urls.map(url => downloadManager.downloadFile(url)));

    expect(maxRunning).toBe(3); // Should strictly adhere to limit
  });

  it('should flush buffer periodically when chunks exceed buffer threshold (mobile behavior)', async () => {
    // 1. Setup: Patch BUFFER_THRESHOLD to a small value (50 bytes)
    // We cast to any because it's a private property
    (downloadManager as any).BUFFER_THRESHOLD = 50;

    const mockUrl = 'https://example.com/mobile-large.bin';

    // 2. Simulate stream larger than threshold (e.g. 150 bytes)
    // We expect 3 flushes (50, 100, 150)
    const chunkSize = 10;
    const totalSize = 150;
    const mockContent = new Uint8Array(totalSize).fill(1); // 150 bytes of 1s

    const mockStream = new ReadableStream({
      start(controller) {
        for (let i = 0; i < totalSize; i += chunkSize) {
          controller.enqueue(mockContent.slice(i, i + chunkSize));
        }
        controller.close();
      }
    });

    (global.fetch as any).mockResolvedValue({
      ok: true,
      headers: new Headers({ 'Content-Length': totalSize.toString(), 'Etag': 'mobile-test' }),
      body: mockStream,
    });

    // 3. Run download
    await downloadManager.downloadFile(mockUrl);

    // 4. Verification:
    // With threshold 50 and total 150 is exactly 3 flushes?
    // Logic: pendingSize >= BUFFER_THRESHOLD.
    // Chunk flow: 
    // ... adds up to 50 -> flush -> saveChunk(index 0, size 50)
    // ... adds up to 50 -> flush -> saveChunk(index 1, size 50)
    // ... adds up to 50 -> flush -> saveChunk(index 2, size 50)

    // Note: depending on loop timing (async), it might flush at end too if last chunk fits exactly.
    // If pending is 0 at end, flushBuffer returns early.
    // So we expect roughly 3 calls to saveChunk.

    expect(saveChunk).toHaveBeenCalledTimes(3);

    // Verify arguments of calls
    expect(saveChunk).toHaveBeenCalledWith(mockUrl, expect.any(Blob), totalSize, 0, 'mobile-test'); // index 0
    expect(saveChunk).toHaveBeenCalledWith(mockUrl, expect.any(Blob), totalSize, 1, 'mobile-test'); // index 1
    expect(saveChunk).toHaveBeenCalledWith(mockUrl, expect.any(Blob), totalSize, 2, 'mobile-test'); // index 2

    // Verify final cache put is full size
    expect(mockCachePut).toHaveBeenCalledTimes(1);
    const response = mockCachePut.mock.calls[0][1];
    const blob = await response.blob();
    expect(blob.size).toBe(totalSize);
  });

  it('should throw error if download stream ends prematurely (size mismatch)', async () => {
    const mockUrl = 'https://example.com/incomplete.file';
    const totalSize = 100;

    // Simulate stream that ends after 50 bytes
    const mockContent = new Uint8Array(50).fill(1);
    const mockStream = new ReadableStream({
      start(controller) {
        controller.enqueue(mockContent);
        controller.close(); // End early
      }
    });

    (global.fetch as any).mockResolvedValue({
      ok: true,
      headers: new Headers({ 'Content-Length': '100', 'Etag': 'incomplete-tag' }), // Header says 100
      body: mockStream,
    });

    // Expect error
    await expect(downloadManager.downloadFile(mockUrl)).rejects.toThrow(/Download incomplete/);

    // Verify it didn't cache the bad file
    expect(mockCachePut).not.toHaveBeenCalled();


    // It might have saved partial chunks to IDB, but the main cache should be clean.
    // Ideally we would want it to clean up IDB too, but throwing prevents usage.
  });

  it('should continue download in memory if IDB write fails (e.g. Incognito/Quota)', async () => {
    const mockUrl = 'https://example.com/incognito.file';
    const totalSize = 20;

    // Mock saveChunk to throw error
    (saveChunk as any).mockRejectedValue(new Error('QuotaExceededError'));

    const mockContent = new Uint8Array(20).fill(1);
    const mockStream = new ReadableStream({
      start(controller) {
        controller.enqueue(mockContent);
        controller.close();
      }
    });

    (global.fetch as any).mockResolvedValue({
      ok: true,
      headers: new Headers({ 'Content-Length': '20' }),
      body: mockStream,
    });

    // Should NOT throw
    await downloadManager.downloadFile(mockUrl);

    // Verify warnings were logged (optional, harder to test without spying on console)

    // Verify it TRIED to save to IDB (but failed)
    expect(saveChunk).toHaveBeenCalled();

    // CRITICAL: Verify file was still cached correctly despite IDB failure
    expect(mockCachePut).toHaveBeenCalledTimes(1);
    const response = mockCachePut.mock.calls[0][1];
    const blob = await response.blob();
    expect(blob.size).toBe(20);
  });

  it('should abort download if quota handler returns false (User clicks Cancel)', async () => {
    const mockUrl = 'https://example.com/abort.file';

    // Mock handler returns false (Cancel)
    const mockHandler = vi.fn().mockResolvedValue(false);
    downloadManager.setQuotaErrorHandler(mockHandler);

    // Mock saveChunk to throw
    (saveChunk as any).mockRejectedValue(new Error('QuotaExceeded'));

    const mockContent = new Uint8Array(20).fill(1);
    const mockStream = new ReadableStream({
      start(controller) {
        controller.enqueue(mockContent);
        controller.close();
      }
    });

    (global.fetch as any).mockResolvedValue({
      ok: true,
      headers: new Headers({ 'Content-Length': '20' }),
      body: mockStream,
    });

    // Expect abort error
    await expect(downloadManager.downloadFile(mockUrl)).rejects.toThrow(/aborted by user/);

    expect(mockHandler).toHaveBeenCalled();
    // Should NOT have cached anything
    expect(mockCachePut).not.toHaveBeenCalled();
  });

  it('should continue download if quota handler returns true (User clicks OK)', async () => {
    const mockUrl = 'https://example.com/continue.file';

    // Mock handler returns true (OK)
    const mockHandler = vi.fn().mockResolvedValue(true);
    downloadManager.setQuotaErrorHandler(mockHandler);

    // Mock saveChunk to throw
    (saveChunk as any).mockRejectedValue(new Error('QuotaExceeded'));

    const mockContent = new Uint8Array(20).fill(1);
    const mockStream = new ReadableStream({
      start(controller) {
        controller.enqueue(mockContent);
        controller.close();
      }
    });

    (global.fetch as any).mockResolvedValue({
      ok: true,
      headers: new Headers({ 'Content-Length': '20' }),
      body: mockStream,
    });

    // Should complete
    await downloadManager.downloadFile(mockUrl);

    expect(mockHandler).toHaveBeenCalled();
    // Should have cached file
    expect(mockCachePut).toHaveBeenCalledTimes(1);
  });

  describe('checkCacheIntegrity', () => {
    it('should return ok for valid file', async () => {
      const mockUrl = 'https://example.com/valid.file';
      const mockBlob = { size: 100 };
      const mockResponse = {
        headers: { get: vi.fn().mockReturnValue('100') },
        clone: vi.fn().mockReturnThis(),
        blob: vi.fn().mockResolvedValue(mockBlob)
      };
      mockCacheMatch.mockResolvedValue(mockResponse);

      const result = await downloadManager.checkCacheIntegrity(mockUrl);
      expect(result).toEqual({ ok: true });
    });

    it('should return corrupt for size mismatch', async () => {
      const mockUrl = 'https://example.com/corrupt.file';
      const mockBlob = { size: 50 }; // Actual 50
      const mockResponse = {
        headers: { get: vi.fn().mockReturnValue('100') },
        clone: vi.fn().mockReturnThis(),
        blob: vi.fn().mockResolvedValue(mockBlob)
      };
      mockCacheMatch.mockResolvedValue(mockResponse);

      const result = await downloadManager.checkCacheIntegrity(mockUrl);
      expect(result.ok).toBe(false);
      expect(result.reason).toContain('Size mismatch');
    });

    it('should return missing for file not in cache', async () => {
      const mockUrl = 'https://example.com/missing.file';
      mockCacheMatch.mockResolvedValue(undefined);

      const result = await downloadManager.checkCacheIntegrity(mockUrl);
      expect(result.ok).toBe(false);
      expect(result.missing).toBe(true);
    });

    it('should return corrupt for empty file (0 bytes) even if Content-Length matches or is missing', async () => {
      // Case 1: CL says 100, size is 0
      const mockUrl = 'https://example.com/empty.file';
      const mockBlob = { size: 0 };
      const mockResponse = {
        headers: { get: vi.fn().mockReturnValue('100') },
        clone: vi.fn().mockReturnThis(),
        blob: vi.fn().mockResolvedValue(mockBlob)
      };
      mockCacheMatch.mockResolvedValue(mockResponse);

      const result = await downloadManager.checkCacheIntegrity(mockUrl);
      expect(result.ok).toBe(false);
      expect(result.reason).toContain('empty');
    });
  });
});

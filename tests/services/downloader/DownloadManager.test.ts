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
});

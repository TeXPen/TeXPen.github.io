/* eslint-disable @typescript-eslint/no-explicit-any */
import { describe, it, expect, vi, beforeEach, afterEach, beforeAll, afterAll } from 'vitest';
import { DownloadManager } from '../../../services/downloader/DownloadManager';
import { downloadScheduler } from '../../../services/downloader/v2/DownloadScheduler';

// Mocks
const mockStore = {
  getMetadata: vi.fn(),
  appendChunk: vi.fn(),
  clear: vi.fn(),
  getStream: vi.fn(),
};

// Mock DownloadScheduler
vi.mock('../../../services/downloader/v2/DownloadScheduler', () => ({
  downloadScheduler: {
    download: vi.fn(),
    getStore: vi.fn(() => mockStore),
  }
}));

describe('DownloadManager', () => {
  let downloadManager: DownloadManager;
  let mockCachePut: any;
  let mockCacheMatch: any;
  let OriginalResponse: any;

  beforeAll(() => {
    OriginalResponse = global.Response;
    // Basic Response mock for arrayBuffer/blob
    global.Response = class MockResponse extends OriginalResponse {
      constructor(body: any, init?: any) {
        super(null, init);
        (this as any)._body = body;
      }
      async blob() { return (this as any)._body; }
      async arrayBuffer() { return (this as any)._body.arrayBuffer(); }
    } as any;
  });

  afterAll(() => {
    global.Response = OriginalResponse;
  });

  beforeEach(() => {
    downloadManager = DownloadManager.getInstance();
    vi.clearAllMocks();

    mockCachePut = vi.fn().mockResolvedValue(undefined);
    mockCacheMatch = vi.fn().mockResolvedValue(null);

    (global as any).caches = {
      open: vi.fn().mockResolvedValue({
        match: mockCacheMatch,
        put: mockCachePut,
        delete: vi.fn(),
      })
    };
  });

  it('should delegate download to scheduler and finalize cache', async () => {
    const url = 'http://test.com/file';

    // Mock Scheduler behavior
    (downloadScheduler.download as any).mockImplementation(async (u: string, cb: any) => {
      // Simulate progress
      cb({ loaded: 100, total: 100 });
    });

    // Mock Store behavior for finalize
    mockStore.getMetadata.mockResolvedValue({
      url,
      totalBytes: 100,
      downloadedBytes: 100,
      chunkCount: 1
    });

    const mockStream = new ReadableStream({
      start(controller) {
        controller.enqueue(new Uint8Array(100));
        controller.close();
      }
    });
    mockStore.getStream.mockResolvedValue(mockStream);

    await downloadManager.downloadFile(url);

    expect(downloadScheduler.download).toHaveBeenCalledWith(url, expect.any(Function));
    expect(mockStore.getMetadata).toHaveBeenCalledWith(url);
    expect(mockStore.getStream).toHaveBeenCalledWith(url, 1);
    expect(mockCachePut).toHaveBeenCalled();
    expect(mockStore.clear).toHaveBeenCalledWith(url);
  });

  it('should throw integrity error if downloaded bytes mismatched', async () => {
    const url = 'http://test.com/corrupt';

    (downloadScheduler.download as any).mockResolvedValue(undefined);

    mockStore.getMetadata.mockResolvedValue({
      url,
      totalBytes: 100,
      downloadedBytes: 50, // Mismatch
      chunkCount: 1
    });

    await expect(downloadManager.downloadFile(url)).rejects.toThrow(/Integrity check failed/);
    expect(mockCachePut).not.toHaveBeenCalled();
  });

  it('should emit progress events', async () => {
    const url = 'http://test.com/progress';
    const onProgress = vi.fn();

    (downloadScheduler.download as any).mockImplementation(async (u: string, cb: any) => {
      cb({ loaded: 50, total: 100 });
    });

    mockStore.getMetadata.mockResolvedValue({
      url,
      totalBytes: 100,
      downloadedBytes: 100,
      chunkCount: 1
    });
    mockStore.getStream.mockResolvedValue(new ReadableStream({
      start(c) { c.close(); }
    }));

    await downloadManager.downloadFile(url, onProgress);

    expect(onProgress).toHaveBeenCalledWith({
      loaded: 50,
      total: 100,
      file: 'progress'
    });
  });
});

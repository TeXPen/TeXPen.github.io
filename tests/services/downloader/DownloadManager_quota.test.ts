import { describe, it, expect, vi, beforeEach, afterEach, Mock } from 'vitest';
import { DownloadManager } from '../../../services/downloader/DownloadManager';
import * as db from '../../../services/downloader/db';

// Mock dependencies
vi.mock('../../../services/downloader/db', () => ({
  getDB: vi.fn(),
  getPartialDownload: vi.fn().mockResolvedValue(null),
  saveChunk: vi.fn(),
  clearPartialDownload: vi.fn(),
}));

// Mock globals
global.fetch = vi.fn();
global.caches = {
  open: vi.fn().mockResolvedValue({
    match: vi.fn().mockResolvedValue(null),
    put: vi.fn(),
    delete: vi.fn(),
  }),
} as any;

global.Response = class {
  constructor(body: any, init?: any) {
    (this as any).headers = new Map(Object.entries(init?.headers || {}));
  }
} as any;

describe('DownloadManager Quota Handling', () => {
  let downloadManager: DownloadManager;

  beforeEach(() => {
    // Reset singleton instance by clearing require cache or using a fresh instance if possible.
    // Since DownloadManager is a singleton, we might need a way to reset it or just cast it.
    // For testing purposes, we can try to re-instantiate if the constructor wasn't private or via 'any'.
    (DownloadManager as any).instance = new (DownloadManager as any)();
    downloadManager = DownloadManager.getInstance();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it('should trigger quota error handler only once for concurrent downloads', async () => {
    const quotaHandler = vi.fn().mockResolvedValue(true); // User clicks "OK" to continue in memory
    downloadManager.setQuotaErrorHandler(quotaHandler);

    // Mock saveChunk to fail with QuotaExceededError
    (db.saveChunk as Mock).mockRejectedValue(new Error('QuotaExceededError'));

    // Mock fetch to return a stream
    (global.fetch as Mock).mockResolvedValue({
      ok: true,
      headers: { get: () => '10' }, // Match body size
      body: {
        getReader: () => {
          let readCount = 0;
          return {
            read: async () => {
              if (readCount++ === 0) {
                return { done: false, value: new Uint8Array(10) };
              }
              return { done: true, value: undefined };
            }
          };
        }
      }
    } as any);

    // Start two concurrent downloads
    const p1 = downloadManager.downloadFile('http://example.com/model1.onnx');
    const p2 = downloadManager.downloadFile('http://example.com/model2.onnx');

    await Promise.all([p1, p2]);

    // Verify quota handler was called exactly once
    expect(quotaHandler).toHaveBeenCalledTimes(1);

    // Verify both downloads completed (implied by Promise.all resolving without error)
  });

  it('should disable IDB for subsequent chunks after quota error', async () => {
    const quotaHandler = vi.fn().mockResolvedValue(true);
    downloadManager.setQuotaErrorHandler(quotaHandler);

    // First call fails, subsequent calls should not happen due to disable flag
    (db.saveChunk as Mock).mockRejectedValueOnce(new Error('QuotaExceededError'));

    (global.fetch as Mock).mockResolvedValue({
      ok: true,
      headers: { get: () => '30' },
      body: {
        getReader: () => {
          let readCount = 0;
          return {
            read: async () => {
              // Return 3 chunks
              if (readCount++ < 3) {
                return { done: false, value: new Uint8Array(10) };
              }
              return { done: true, value: undefined };
            }
          };
        }
      }
    } as any);

    await downloadManager.downloadFile('http://example.com/test.onnx');

    expect(quotaHandler).toHaveBeenCalledTimes(1);
    // ideally saveChunk is called once (fails), then never again for this file
    // OR called multiple times but checking the flag locally?
    // In our impl: "if (!this.isIDBDisabled) ... saveChunk"
    // So subsequent chunks should NOT call saveChunk.
    expect(db.saveChunk).toHaveBeenCalledTimes(1);
  });
});

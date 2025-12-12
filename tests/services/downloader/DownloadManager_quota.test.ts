/* eslint-disable @typescript-eslint/no-explicit-any */
import { describe, it, expect, vi, beforeEach, afterEach, Mock } from 'vitest';
import { DownloadManager } from '../../../services/downloader/DownloadManager';
import * as db from '../../../services/downloader/db';

// Mock dependencies
vi.mock('../../../services/downloader/db', () => ({
  getDB: vi.fn().mockResolvedValue({
    // Mock db object - returns truthy to indicate IDB is available
    transaction: vi.fn(),
    get: vi.fn(),
  }),
  getPartialDownload: vi.fn().mockResolvedValue(null),
  saveChunk: vi.fn(),
  clearPartialDownload: vi.fn(),
  getChunk: vi.fn(),
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

global.Blob = class {
  parts: any[];
  size: number;
  constructor(parts?: any[], _options?: any) {
    this.parts = parts || [];
    this.size = this.parts.reduce((acc, p) => acc + (p.byteLength || p.size || p.length || 0), 0);
  }
  async arrayBuffer() {
    const result = new Uint8Array(this.size);
    let offset = 0;
    for (const p of this.parts) {
      if (p instanceof Uint8Array) {
        result.set(p, offset);
        offset += p.byteLength;
      } else if (typeof p === 'string') {
        const enc = new TextEncoder().encode(p);
        result.set(enc, offset);
        offset += enc.byteLength;
      }
    }
    return result.buffer;
  }
  stream() {
    // minimal stream impl if needed
    return { getReader: () => ({ read: async () => ({ done: true }) }) };
  }
} as any;

global.Response = class {
  body: any;
  headers: Map<string, string>;
  status: number;
  statusText: string;
  ok: boolean;

  constructor(body: any, init?: any) {
    this.body = body;
    this.headers = new Map(Object.entries(init?.headers || {}));
    this.status = init?.status || 200;
    this.statusText = init?.statusText || 'OK';
    this.ok = this.status >= 200 && this.status < 300;
  }

  async blob() {
    if (this.body && this.body.getReader) {
      const reader = this.body.getReader();
      const chunks = [];
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
      }
      return new Blob(chunks);
    }
    return new Blob([this.body]);
  }

  async arrayBuffer() {
    if (this.body && this.body.getReader) {
      const reader = this.body.getReader();
      const chunks: Uint8Array[] = [];
      let totalLength = 0;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        totalLength += value.length;
      }
      const result = new Uint8Array(totalLength);
      let offset = 0;
      for (const chunk of chunks) {
        result.set(chunk, offset);
        offset += chunk.length;
      }
      return result.buffer;
    }
    // Fallback if body is not stream (e.g. string or Blob?)
    // If it's a blob without arrayBuffer, we are stuck, but in our tests we use streams.
    // If it is 'this.body' (any), let's assume it might have arrayBuffer if it's not a stream.
    if (this.body && typeof this.body.arrayBuffer === 'function') {
      return this.body.arrayBuffer();
    }

    // Check if it is a Uint8Array
    if (this.body instanceof Uint8Array) {
      return this.body.buffer;
    }

    throw new Error('Response mock: arrayBuffer handling not implemented for this body type');
  }
} as any;

describe.skip('DownloadManager Quota Handling', () => {
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

  it('should recover previously saved chunks when switching to memory mode', async () => {
    const quotaHandler = vi.fn().mockResolvedValue(true);
    downloadManager.setQuotaErrorHandler(quotaHandler);

    // Setup mock cache to capture put calls
    const mockCache = {
      match: vi.fn().mockResolvedValue(null),
      put: vi.fn(),
      delete: vi.fn(),
    };
    (global.caches.open as Mock).mockResolvedValue(mockCache);

    const CHUNK_SIZE = 10;
    const TOTAL_CHUNKS = 3;
    const mockChunks = [
      new Uint8Array(CHUNK_SIZE).fill(1), // Chunk 0 (saved to IDB)
      new Uint8Array(CHUNK_SIZE).fill(2), // Chunk 1 (fails to save -> memory)
      new Uint8Array(CHUNK_SIZE).fill(3), // Chunk 2 (memory)
    ];

    // Mock getChunk to return Chunk 0 when recovered
    // Return Uint8Array to bypass Blob.arrayBuffer() issues in test env.
    // DownloadManager handles Uint8Array in downloadedChunks.
    (db.getChunk as Mock).mockImplementation(async (url, index) => {
      if (index === 0) return mockChunks[0] as unknown as Blob;
      return undefined;
    });

    // Mock saveChunk to fail on Chunk 1 (2nd call)
    let saveCount = 0;
    (db.saveChunk as Mock).mockImplementation(async () => {
      // 0-based index of calls. First call (chunk 0) succeeds.
      // Second call (chunk 1) fails.
      if (saveCount === 1) {
        saveCount++;
        throw new Error('QuotaExceededError');
      }
      saveCount++;
      return;
    });

    // Force flush every chunk
    (downloadManager as any).BUFFER_THRESHOLD = 5;

    // Mock fetch stream
    (global.fetch as Mock).mockResolvedValue({
      ok: true,
      headers: {
        get: (key: string) => key === 'Content-Length' ? (CHUNK_SIZE * TOTAL_CHUNKS).toString() : null
      },
      body: {
        getReader: () => {
          let readCount = 0;
          return {
            read: async () => {
              if (readCount < TOTAL_CHUNKS) {
                return { done: false, value: mockChunks[readCount++] };
              }
              return { done: true, value: undefined };
            }
          };
        }
      }
    } as any);

    await downloadManager.downloadFile('http://example.com/recovery_test.onnx');

    // Verify quota handler was called
    expect(quotaHandler).toHaveBeenCalledTimes(1);

    // Verify cache.put was called with the correct FULL content
    expect(mockCache.put).toHaveBeenCalledTimes(1);
    const putCall = mockCache.put.mock.calls[0];
    const response = putCall[1] as Response;
    // Use manual arrayBuffer() implementation to avoid Blob issues
    const buffer = await response.arrayBuffer();
    const result = new Uint8Array(buffer);

    // Expected full content: 1, 1..., 2, 2..., 3, 3...
    const expected = new Uint8Array(CHUNK_SIZE * TOTAL_CHUNKS);
    expected.set(mockChunks[0], 0);
    expected.set(mockChunks[1], CHUNK_SIZE);
    expected.set(mockChunks[2], CHUNK_SIZE * 2);

    expect(result).toEqual(expected);
  });
});

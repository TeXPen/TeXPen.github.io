/* eslint-disable @typescript-eslint/no-explicit-any */
import { describe, it, expect, vi, beforeEach, afterEach, Mock } from 'vitest';
import { DownloadManager } from '../../../services/downloader/DownloadManager';
import { downloadScheduler } from '../../../services/downloader/v2/DownloadScheduler';

// Mock ChunkStore
vi.mock('../../../services/downloader/v2/ChunkStore', () => {
  const ChunkStore = vi.fn();
  ChunkStore.prototype.getMetadata = vi.fn();
  ChunkStore.prototype.appendChunk = vi.fn();
  ChunkStore.prototype.clear = vi.fn();
  ChunkStore.prototype.getStream = vi.fn().mockResolvedValue(new ReadableStream({
    start(controller) { controller.close(); }
  }));
  return { ChunkStore };
});

// Mock globals
global.fetch = vi.fn();
global.caches = {
  open: vi.fn().mockResolvedValue({
    match: vi.fn().mockResolvedValue(null),
    put: vi.fn(),
    delete: vi.fn(),
  }),
} as any;

describe('DownloadManager Resume (V2)', () => {
  let downloadManager: DownloadManager;
  let mockStore: any;

  beforeEach(async () => {
    vi.resetModules();
    vi.clearAllMocks();

    // Reset Scheduler instance hack
    (downloadScheduler as any).jobs = new Map();
    (downloadScheduler as any).queue = [];
    (downloadScheduler as any).activeCount = 0;

    // Get the mocked store instance from the scheduler
    // Since Scheduler creates new ChunkStore(), and we mocked the class, we need to grab the instance.
    // Ideally we can just inject it? No, it's private.
    // But since we mocked the module, the instance inside scheduler IS our mock.
    mockStore = (downloadScheduler as any).store;

    // Re-instantiate Manager
    (DownloadManager as any).instance = new (DownloadManager as any)();
    downloadManager = DownloadManager.getInstance();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it('should resume download using downloadedBytes from metadata', async () => {
    // Mock initial state for DownloadJob check
    mockStore.getMetadata
      .mockResolvedValueOnce({
        url: 'http://example.com/resume.onnx',
        chunkCount: 2,
        downloadedBytes: 200,
        totalBytes: 500,
        etag: 'test-etag'
      })
      // Mock final state for finalizeCache check
      .mockResolvedValueOnce({
        url: 'http://example.com/resume.onnx',
        chunkCount: 3, // +1 chunk
        downloadedBytes: 500, // Completed
        totalBytes: 500,
        etag: 'test-etag'
      });

    // Mock fetch to return remaining bytes (300 bytes)
    const remainingSize = 300;
    (global.fetch as Mock).mockResolvedValue({
      ok: true,
      headers: {
        get: (key: string) => {
          if (key === 'Content-Length') return remainingSize.toString();
          if (key === 'Etag') return 'test-etag';
          return null;
        }
      },
      status: 206,
      body: {
        getReader: () => {
          let served = false;
          return {
            read: async () => {
              if (!served) {
                served = true;
                return { done: false, value: new Uint8Array(remainingSize) }; // Serve 300 bytes
              }
              return { done: true, value: undefined };
            }
          };
        }
      }
    } as any);

    await downloadManager.downloadFile('http://example.com/resume.onnx');

    // Verify Range header
    const fetchCall = (global.fetch as Mock).mock.calls[0];
    const headers = fetchCall[1].headers;
    expect(headers['Range']).toBe('bytes=200-');
  });

  it('should restart download if downloadedBytes is missing or 0', async () => {
    // Legacy metadata or empty start
    mockStore.getMetadata
      .mockResolvedValueOnce({
        url: 'http://example.com/restart.onnx',
        chunkCount: 2,
        downloadedBytes: 0,
        totalBytes: 500,
      })
      .mockResolvedValueOnce({
        url: 'http://example.com/restart.onnx',
        chunkCount: 1,
        downloadedBytes: 500,
        totalBytes: 500,
      });

    // Full download (500 bytes)
    (global.fetch as Mock).mockResolvedValue({
      ok: true,
      headers: {
        get: (key: string) => key === 'Content-Length' ? '500' : null
      },
      status: 200, // Full download
      body: {
        getReader: () => {
          let served = false;
          return {
            read: async () => {
              if (!served) {
                served = true;
                return { done: false, value: new Uint8Array(500) }; // Serve 500 bytes
              }
              return { done: true, value: undefined };
            }
          };
        }
      }
    } as any);

    await downloadManager.downloadFile('http://example.com/restart.onnx');

    // Verify fetched from 0 (Range header depends on startByte > 0)
    const fetchCall = (global.fetch as Mock).mock.calls[0];
    const headers = fetchCall[1].headers;
    expect(headers['Range']).toBeUndefined();

    // Verify cleanup was called
    expect(mockStore.clear).toHaveBeenCalledWith('http://example.com/restart.onnx');
  });
});

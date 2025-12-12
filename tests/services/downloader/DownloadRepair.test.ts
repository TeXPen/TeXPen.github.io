/* eslint-disable @typescript-eslint/no-explicit-any */
import { describe, it, expect, vi, beforeEach, beforeAll, afterAll } from 'vitest';
import { DownloadManager } from '../../../services/downloader/DownloadManager';
import { ModelLoader } from '../../../services/inference/ModelLoader';

// Mock IDBKeyRange
global.IDBKeyRange = {
  bound: vi.fn(),
  lowerBound: vi.fn(),
  upperBound: vi.fn(),
  only: vi.fn(),
} as any;

// Mock V2 dependencies
vi.mock('../../../services/downloader/v2/DownloadScheduler', () => ({
  downloadScheduler: {
    download: vi.fn().mockImplementation(async (url, onProgress) => {
      // Immediate completion mock
      if (onProgress) onProgress({ loaded: 100, total: 100, file: 'test' });
    }),
    getStore: vi.fn().mockReturnValue({
      getMetadata: vi.fn().mockResolvedValue({
        downloadedBytes: 100,
        totalBytes: 100,
        chunkCount: 1
      }),
      getStream: vi.fn().mockResolvedValue(new ReadableStream({
        start(controller) { controller.close(); }
      })),
      clear: vi.fn(),
    })
  }
}));

// Legacy DB mock (still needed if referenced by other imports, but verified unused by DM)
vi.mock('../../../services/downloader/db', () => ({
  getDB: vi.fn(),
  saveChunk: vi.fn(),
  getPartialDownload: vi.fn(),
  clearPartialDownload: vi.fn(),
}));

describe('Repair Flow Integration', () => {
  let downloadManager: DownloadManager;
  let modelLoader: ModelLoader;
  let mockCachePut: any;
  let mockCacheMatch: any;
  let mockCacheDelete: any;
  let OriginalResponse: any;

  const MODEL_ID = 'test-model-id';
  const TEST_FILE = 'onnx/encoder_model.onnx';
  const FULL_URL = `https://huggingface.co/${MODEL_ID}/resolve/main/${TEST_FILE}`;

  beforeAll(() => {
    OriginalResponse = global.Response;
    global.Response = class MockResponse extends OriginalResponse {
      private _blobBody: Blob | null = null;
      constructor(body: BodyInit | null, init?: ResponseInit) {
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
        return new (global.Response as any)(this._blobBody || null, {
          status: this.status,
          statusText: this.statusText,
          headers: this.headers
        });
      }
    } as any;
  });

  afterAll(() => {
    global.Response = OriginalResponse;
  });

  beforeEach(() => {
    vi.resetModules();
    downloadManager = DownloadManager.getInstance();
    modelLoader = ModelLoader.getInstance();

    // Clear singleton instance logic if possible, or just reset state
    // Since we can't easily reset private singleton instance, we rely on mocks being fresh

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

    (downloadManager as any).activeDownloads = new Map();
  });

  it('should detect corruption, delete file, and re-download during repair flow', async () => {
    // 1. Simulate Corrupted State
    // Cache has a file, but it's corrupted (size mismatch)
    const expectedSize = 100;
    const actualSize = 50;
    const corruptedBlob = new Blob([new Uint8Array(actualSize)]);
    const corruptedResponse = new Response(corruptedBlob, {
      headers: { 'Content-Length': expectedSize.toString() }
    });

    // Mock cache match to return corrupted file first
    mockCacheMatch.mockImplementation(async (url: string) => {
      if (url === FULL_URL) {
        return corruptedResponse;
      }
      return null;
    });

    // 2. Validate (simulate useVerifyDownloads logic)
    // We need to mock sessionOptions to point to our test file
    const sessionOptions = {
      encoder_model_file_name: 'encoder_model.onnx',
      decoder_model_file_name: 'decoder_model.onnx',
      device: 'wasm',
      dtype: 'fp32'
    };

    // Mock validateModelFiles calling checkCacheIntegrity
    // We can actually use real ModelLoader logic since we mocked DownloadManager deps
    const corrupted = await modelLoader.validateModelFiles(MODEL_ID, sessionOptions);

    expect(corrupted).toContain(FULL_URL);

    // 3. Repair (simulate user clicking "Repair")
    // Step A: Delete
    for (const url of corrupted) {
      await downloadManager.deleteFromCache(url);
    }

    expect(mockCacheDelete).toHaveBeenCalledWith(FULL_URL);

    // Update mock: After delete, cache.match returns null
    mockCacheMatch.mockResolvedValue(null);

    // Step B: Re-download (preDownloadModels)
    // Mock fetch for successful download
    (global.fetch as any).mockImplementation(() => {
      const validContent = new Uint8Array(expectedSize).fill(1);
      const validStream = new ReadableStream({
        start(controller) {
          controller.enqueue(validContent);
          controller.close();
        }
      });
      return Promise.resolve({
        ok: true,
        headers: new Headers({ 'Content-Length': expectedSize.toString() }),
        body: validStream,
      });
    });

    await modelLoader.preDownloadModels(MODEL_ID, sessionOptions, () => { });

    // 4. Verify
    // Expect scheduler to be called instead of direct fetch (implementation detail masked by mock)
    const { downloadScheduler } = await import('../../../services/downloader/v2/DownloadScheduler');
    expect(downloadScheduler.download).toHaveBeenCalledWith(FULL_URL, expect.anything());

    // Expect cache.put to be called (download saved)
    expect(mockCachePut).toHaveBeenCalledTimes(2); // Once per file (encoder, decoder)
  });

  it('should clear partial download from DB when deleting from cache', async () => {
    const { downloadScheduler } = await import('../../../services/downloader/v2/DownloadScheduler');
    const store = downloadScheduler.getStore();

    const testUrl = 'https://example.com/test.file';
    await downloadManager.deleteFromCache(testUrl);

    expect(store.clear).toHaveBeenCalledWith(testUrl);
  });
});


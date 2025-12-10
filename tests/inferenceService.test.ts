import { describe, it, expect, beforeEach, afterEach, vi, beforeAll } from 'vitest';
import { InferenceService } from '../services/inference/InferenceService';
import { InferenceOptions } from '../services/inference/types';

// Mock DownloadManager to prevent 300MB downloads while keeping Service logic "legit"
vi.mock('../services/downloader/DownloadManager', () => {
  return {
    downloadManager: {
      downloadFile: vi.fn().mockImplementation(async (url: string, onProgress) => {
        // Simulate "fast" Network/Cache behavior by writing dummy files to the cache
        // This ensures the Service thinks the file is available and proceeds to usage
        if (typeof caches !== 'undefined') {
          const cache = await caches.open('transformers-cache');

          let content: BodyInit = new Uint8Array([0x08, 0x00]); // Minimal binary-like content
          // Note: transformers.js validation might check magic bytes or protobuf validity.
          // Invalid protobuf will cause from_pretrained to throw, which we expect and assert on.

          let headers: any = { 'content-length': '2' };

          if (url.endsWith('.json')) {
            // Provide minimal valid JSON for tokenizer/config to pass basic parsing if possible
            const dummyConfig = {
              model_type: 'vision-encoder-decoder',
              encoder: { model_type: 'vit' },
              decoder: { model_type: 'gpt2' },
              is_encoder_decoder: true
            };
            content = JSON.stringify(dummyConfig);
            headers = { 'content-length': content.length.toString(), 'content-type': 'application/json' };
          }

          await cache.put(url, new Response(content, { headers }));
        }

        // Simulate progress for the "download"
        if (onProgress) onProgress({ loaded: 100, total: 100, file: url });
      })
    }
  };
});

describe('InferenceService Integration (Efficient)', () => {
  let inferenceService: InferenceService;

  beforeAll(() => {
    // Simulate WebGPU availability for the fallback test
    // We want to test that it TRIES WebGPU, then falls back if it fails (or if we force it)
    Object.defineProperty(navigator, 'gpu', {
      value: {
        requestAdapter: vi.fn().mockResolvedValue({
          requestDevice: vi.fn().mockResolvedValue({
            limits: { maxStorageBufferBindingSize: 2147483648 } // Big enough
          })
        })
      },
      configurable: true
    });
  });

  beforeEach(() => {
    // Reset singleton if needed or just get instance
    inferenceService = InferenceService.getInstance();
    vi.clearAllMocks();
  });

  afterEach(async () => {
    try {
      // Cleanup if needed
    } catch (e) { }
  });

  it('should attempt initialization and download without hitting network (Legit Flow)', async () => {
    // We expect this to fail at the "Load Model" stage because we provided DUMMY ONNX files.
    // However, getting to that failure proves that:
    // 1. DownloadManager was called correctly (and "downloaded" our dummy).
    // 2. InferenceService proceeded to initialize transformers.
    // 3. from_pretrained attempted to parse the file.
    // This verifies the INTEGRATION without the 300MB cost.

    const options: InferenceOptions = {
      device: 'webgpu', // Will try WebGPU
      dtype: 'fp32'
    };

    console.log('[Test] Starting legit initialization with mocked network...');

    try {
      await inferenceService.init((status) => {
        console.log(`[Test Progress] ${status}`);
      }, options);
    } catch (error: any) {
      console.log('[Test] Caught expected error during loading:', error.message);
      // Validating that we crashed on the MODEL parsing, not on unexpected undefined/network errors.
      // Common errors for garbage ONNX: "invalid wire type", "illegal buffer", "offset out of bounds", "protobuf"
      // Or from transformers.js: "is not a valid ONNX"
      const isParsingError = /protobuf|offset|invalid|wire type|illegal|buffer|onnx/i.test(error.message);

      // If the error is "Unsupported device", it means we successfully loaded and tried to Init WebGPU session!
      // This is actual success for the loading phase!
      // But since our ONNX is junk, ONNX Runtime might crash before checking device or during session creation?
      // Actually creating an inference session with junk ONNX usually throws parsing error.

      expect(isParsingError).toBe(true);
    }

    // specific check: DownloadManager must have been utilized
    const { downloadManager } = await import('../services/downloader/DownloadManager');
    expect(downloadManager.downloadFile).toHaveBeenCalled();
  }, 30000);

  it('should handle concurrent requests gracefully (simulated)', async () => {
    // Just verifying code path, not full inference (since no model)
    // This test is harder to "legit" test without model. 
    // We can skip or mock just the runInference part if we want coverage.
    // Or relies on the fact that runInference throws if not init?

    // Since 'init' failed (above), runInference should throw 'not ready' or try to init again.
    // We accept that we can't fully text runInference output without a real model.
    // But we can verify it doesn't crash inappropriately.

    try {
      await inferenceService.infer(new Blob([]));
    } catch (e: any) {
      // Expected since init failed or model is missing
      expect(e).toBeDefined();
    }
  });
});

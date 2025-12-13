
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { ModelLoader } from '../../../services/inference/ModelLoader';

// Mock DownloadManager
const mockDownloadFile = vi.fn();
vi.mock('../../../services/downloader/DownloadManager', () => ({
  downloadManager: {
    downloadFile: mockDownloadFile
  }
}));

// Mock config
vi.mock('../../../services/inference/config', () => ({
  getSessionOptions: () => ({
    encoder_model_file_name: 'encoder.onnx',
    decoder_model_file_name: 'decoder.onnx'
  })
}));

describe('ModelLoader Progress Reporting', () => {
  let modelLoader: ModelLoader;

  beforeEach(() => {
    vi.clearAllMocks();
    modelLoader = ModelLoader.getInstance();
  });

  it('should report detailed progress', async () => {
    const onProgress = vi.fn();

    // Setup mock to simulate progress immediately
    mockDownloadFile.mockImplementation(async (url, callback) => {
      // simulate 50% progress for 20MB file
      callback({
        loaded: 10 * 1024 * 1024,
        total: 20 * 1024 * 1024
      });
    });

    await modelLoader.preDownloadModels('test-model', {
      encoder_model_file_name: 'encoder.onnx',
      decoder_model_file_name: 'decoder.onnx',
      device: 'wasm',
    }, onProgress);

    // Initial check
    expect(onProgress).toHaveBeenCalledWith('Loading models...', 0);

    // Check for progress call
    // We have 2 files (encoder, decoder) from the mocked config.
    // Both will report 10/20 MB.
    // Total loaded: 20 MB, Total size: 40 MB.
    // Global pct: 50%.
    // Encoder: 10.00/20.00 MB (50%)
    // Decoder: 10.00/20.00 MB (50%)

    // Note: The order of keys in progressState might vary, but usually insertion order.
    // File names are constructed in preDownloadModels: `onnx/${sessionOptions.encoder_model_file_name}`

    // We expect onProgress to be called with something like:
    // "Enc: 10.00/20.00 MB (50%) | Dec: 10.00/20.00 MB (50%)", 50

    const calls = onProgress.mock.calls;
    // Check the last call to ensure both files reported progress
    const progressCall = calls[calls.length - 1];

    expect(progressCall).toBeDefined();
    const status = progressCall![0];
    const percentage = progressCall![1];

    expect(status).toContain('Enc: 10.00/20.00 MB (50%)');
    expect(status).toContain('Dec: 10.00/20.00 MB (50%)');
    expect(percentage).toBe(50);
  });
});

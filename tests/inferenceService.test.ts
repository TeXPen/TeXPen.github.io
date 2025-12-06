import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { InferenceService } from '../services/inference/InferenceService';
import { INFERENCE_CONFIG } from '../services/inference/config';

// Mocks
const mockGenerate = vi.fn();
const mockDecode = vi.fn();
const mockFromPretrainedModel = vi.fn();
const mockFromPretrainedTokenizer = vi.fn();
const mockPreprocess = vi.fn();
const mockBeamSearch = vi.fn();

vi.mock('@huggingface/transformers', () => ({
  AutoModelForVision2Seq: {
    from_pretrained: (...args: any[]) => mockFromPretrainedModel(...args),
  },
  AutoTokenizer: {
    from_pretrained: (...args: any[]) => mockFromPretrainedTokenizer(...args),
  },
  Tensor: class {
    constructor(public type: string, public data: any, public dims: any[]) { }
    dispose() { }
  }
}));

vi.mock('../services/inference/imagePreprocessing', () => ({
  preprocess: (...args: any[]) => mockPreprocess(...args),
}));

vi.mock('../services/inference/beamSearch', () => ({
  beamSearch: (...args: any[]) => mockBeamSearch(...args),
}));

describe('InferenceService', () => {
  let service: InferenceService;

  beforeEach(() => {
    // Reset singleton instance (this is a bit hacky because it's private, but needed for testing singleton)
    (InferenceService as any).instance = null;
    service = InferenceService.getInstance();

    // Reset mocks
    vi.clearAllMocks();

    // Setup default mock behaviors
    mockGenerate.mockResolvedValue([[1, 2, 3]]);
    mockDecode.mockReturnValue('raw latex');

    mockFromPretrainedModel.mockResolvedValue({
      config: { dtype: 'q8', device: 'wasm' },
      generate: mockGenerate,
      dispose: vi.fn(),
    });

    mockFromPretrainedTokenizer.mockResolvedValue({
      decode: mockDecode,
      eos_token_id: 1,
      bos_token_id: 0,
      pad_token_id: 0,
    });

    mockPreprocess.mockResolvedValue({
      tensor: { dispose: vi.fn() },
      debugImage: 'data:image/png;base64,mock',
    });
  });

  afterEach(async () => {
    await service.dispose();
  });

  it('should initialize model and tokenizer', async () => {
    await service.init();
    expect(mockFromPretrainedModel).toHaveBeenCalledWith(INFERENCE_CONFIG.MODEL_ID, expect.any(Object));
    expect(mockFromPretrainedTokenizer).toHaveBeenCalledWith(INFERENCE_CONFIG.MODEL_ID);
  });

  it('should run single candidate inference using beamSearch', async () => {
    const mockBlob = new Blob([''], { type: 'image/png' });
    mockBeamSearch.mockResolvedValue(['raw latex']);

    const result = await service.infer(mockBlob, 1);

    expect(mockPreprocess).toHaveBeenCalled();
    expect(mockFromPretrainedModel).toHaveBeenCalled();
    // It now uses beamSearch even for 1 candidate
    expect(mockBeamSearch).toHaveBeenCalledWith(
      expect.anything(),
      expect.anything(),
      expect.anything(),
      1, // numBeams
      expect.anything(), // signal
      expect.anything(), // maxTokens
      expect.anything() // repetitionPenalty
    );
    expect(result.latex).toBe('raw latex');
    expect(result.candidates).toHaveLength(1);
  });

  it('should queue concurrent requests', async () => {
    const mockBlob = new Blob([''], { type: 'image/png' });

    // Mock preprocess or beamSearch to take some time to simulate processing
    mockBeamSearch.mockImplementation(async () => {
      await new Promise(r => setTimeout(r, 50));
      return ['latex result'];
    });

    // Call infer twice concurrently
    const p1 = service.infer(mockBlob, 1);
    const p2 = service.infer(mockBlob, 1);

    // Both should resolve successfully
    // Both should resolve successfully
    const [r1, r2] = await Promise.all([p1, p2]);

    expect(r1.latex).toBe('latex result');
    expect(r2.latex).toBe('latex result');
    expect(mockBeamSearch).toHaveBeenCalledTimes(2);
  });

  it('should force dispose and abort current inference', async () => {
    // Start a long running inference
    const mockBlob = new Blob([''], { type: 'image/png' });
    let resolveSearch: (val: any) => void;
    let rejectSearch: (reason: any) => void;

    mockBeamSearch.mockImplementation((...args) => {
      const signal = args.find(arg => arg instanceof AbortSignal);
      return new Promise((res, rej) => {
        resolveSearch = res;
        rejectSearch = rej;

        if (signal) {
          if (signal.aborted) {
            rej(new Error("Aborted"));
            return;
          }
          signal.addEventListener('abort', () => rej(new Error("Aborted")));
        }
      });
    });

    const inferencePromise = service.infer(mockBlob, 1);

    // Give it a moment to start
    await new Promise(r => setTimeout(r, 0));

    // Force dispose
    await service.dispose(true);

    // The inference should reject or settle
    await expect(inferencePromise).rejects.toThrow("Aborted");
  });
});

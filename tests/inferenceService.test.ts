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

  it('should run single candidate inference using model.generate', async () => {
    const mockBlob = new Blob([''], { type: 'image/png' });
    const result = await service.infer(mockBlob, 1);

    expect(mockPreprocess).toHaveBeenCalled();
    expect(mockFromPretrainedModel).toHaveBeenCalled();
    expect(mockGenerate).toHaveBeenCalled();
    expect(result.latex).toBe('raw latex'); // "raw latex" -> removeStyle -> "raw latex" -> addNewlines
    expect(result.candidates).toHaveLength(1);
    expect(mockBeamSearch).not.toHaveBeenCalled();
  });

  it('should run multi-candidate inference using beamSearch', async () => {
    const mockBlob = new Blob([''], { type: 'image/png' });
    mockBeamSearch.mockResolvedValue(['candidate 1', 'candidate 2']);

    const result = await service.infer(mockBlob, 2);

    expect(mockPreprocess).toHaveBeenCalled();
    expect(mockBeamSearch).toHaveBeenCalledWith(expect.anything(), expect.anything(), expect.anything(), 2);
    expect(mockGenerate).not.toHaveBeenCalled();
    expect(result.candidates).toHaveLength(2);
    expect(result.candidates[0]).toBe('candidate 1');
  });

  it('should prevent double initialization', async () => {
    // Call init twice in parallel
    const p1 = service.init();
    const p2 = service.init();

    // We check that the underlying logic was only executed once
    await Promise.all([p1, p2]);
    expect(mockFromPretrainedModel).toHaveBeenCalledTimes(1);
    expect(mockFromPretrainedTokenizer).toHaveBeenCalledTimes(1);
  });

  it('should throw error if already inferring', async () => {
    const mockBlob = new Blob([''], { type: 'image/png' });

    // Simulate long running inference
    mockPreprocess.mockImplementation(async () => {
      await new Promise(r => setTimeout(r, 100));
      return { tensor: { dispose: vi.fn() }, debugImage: '' };
    });

    const p1 = service.infer(mockBlob, 1);
    await expect(service.infer(mockBlob, 1)).rejects.toThrow("Another inference is already in progress");
    await p1;
  });

  it('should force dispose even if inferring', async () => {
    // Simulate inferring
    (service as any).isInferring = true;

    // Should not throw
    await service.dispose(true);

    expect((service as any).isInferring).toBe(false);
  });
});

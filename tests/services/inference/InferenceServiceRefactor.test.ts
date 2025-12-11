
import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { InferenceService } from '../../../services/inference/InferenceService';
import { InferenceOptions, SamplingOptions } from '../../../services/inference/types';

// Mock dependencies
vi.mock('../../../services/inference/imagePreprocessing', () => ({
  preprocess: vi.fn().mockResolvedValue({
    tensor: { dispose: vi.fn() },
    debugImage: 'debug'
  })
}));

vi.mock('../../../services/inference/beamSearch', () => ({
  beamSearch: vi.fn().mockResolvedValue(['beam_result_1', 'beam_result_2'])
}));

vi.mock('../../../utils/latex', () => ({
  removeStyle: (s: string) => s,
  addNewlines: (s: string) => s
}));

// Mock config
vi.mock('../../../services/inference/config', () => ({
  MODEL_CONFIG: {
    DEFAULT_QUANTIZATION: 'q8',
    ID: 'test-model'
  },
  getSessionOptions: vi.fn(),
  getGenerationConfig: vi.fn().mockReturnValue({
    max_new_tokens: 10,
    decoder_start_token_id: 1,
    repetition_penalty: 1.0
  })
}));

describe('InferenceService Refactor Logic', () => {
  let service: InferenceService;
  let mockModel: any;
  let mockTokenizer: any;

  beforeEach(() => {
    // Reset singleton
    (InferenceService as any).instance = null;
    service = InferenceService.getInstance();

    // Mock internal model and tokenizer
    mockModel = {
      generate: vi.fn().mockResolvedValue([[101, 102]]),
      dispose: vi.fn()
    };
    mockTokenizer = {
      batch_decode: vi.fn().mockReturnValue(['decoded_latex'])
    };

    // Inject mocks directly
    (service as any).model = mockModel;
    (service as any).tokenizer = mockTokenizer;
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it('should use greedy decoding when num_candidates=1, even if doSample=true', async () => {
    const blob = new Blob([]);
    const options: SamplingOptions = {
      do_sample: true,
      temperature: 0.7,
      top_k: 50,
      top_p: 0.9,
      num_beams: 1 // Candidates = 1
    };

    const result = await service.infer(blob, options);

    // Verify result
    expect(result.candidates).toEqual(['decoded_latex']);

    // Verify calling convention logic
    // Expect generate NOT to have do_sample: true in options
    expect(mockModel.generate).toHaveBeenCalledTimes(1);
    const callArgs = mockModel.generate.mock.calls[0][0];

    // Crucially, verify do_sample is NOT true (undefined or false)
    expect(callArgs.do_sample).toBeFalsy();
  });

  it('should use sampling logic when num_candidates > 1 and doSample=true', async () => {
    const blob = new Blob([]);
    const options: SamplingOptions = {
      do_sample: true,
      temperature: 0.8,
      top_k: 40,
      top_p: 0.95,
      num_beams: 2 // Candidates = 2
    };

    // Mock generate to return something different for loop calls if possible, 
    // but here we just count calls.
    mockModel.generate.mockResolvedValue([[101]]);

    const result = await service.infer(blob, options);

    // Should call generate 2 times (manual loop)
    expect(mockModel.generate).toHaveBeenCalledTimes(2);

    // Verify options passed
    const callArgs = mockModel.generate.mock.calls[0][0];
    expect(callArgs.do_sample).toBe(true);
    expect(callArgs.temperature).toBe(0.8);
    expect(callArgs.top_k).toBe(40);
    expect(callArgs.top_p).toBe(0.95);
  });

  it('should use custom beamSearch when num_candidates > 1 and doSample=false', async () => {
    const { beamSearch } = await import('../../../services/inference/beamSearch');

    const blob = new Blob([]);
    const options: SamplingOptions = {
      do_sample: false,
      num_beams: 3
    };

    await service.infer(blob, options);

    expect(beamSearch).toHaveBeenCalled();
    expect(mockModel.generate).not.toHaveBeenCalled();
  });
});

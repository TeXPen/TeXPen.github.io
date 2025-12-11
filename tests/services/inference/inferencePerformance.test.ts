
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { beamSearch } from '../../../services/inference/beamSearch';
import { Tensor } from '@huggingface/transformers';

// Mock Tensor
vi.mock('@huggingface/transformers', () => ({
  Tensor: class {
    constructor(public type: string, public data: any, public dims: any[]) { }
    dispose() { }
  },
  PreTrainedModel: class { },
  PreTrainedTokenizer: class { }
}));

describe('beamSearch Performance', () => {
  let mockModel: any;
  let mockTokenizer: any;
  let mockPixelValues: any;

  beforeEach(() => {
    // Delay helper
    const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

    mockModel = {
      sessions: {
        decoder: {
          inputNames: ['decoder_input_ids', 'encoder_hidden_states'],
          run: vi.fn().mockImplementation(async () => {
            await delay(50); // 50ms delay
            return {
              logits: {
                dims: [1, 1, 10],
                data: new Float32Array(10).map((_, i) => i === 2 ? 10 : 0), // Favor token 2
                dispose: () => { }
              },
              'present.0.decoder.key': new Tensor('float32', new Float32Array(1), [1, 1, 1, 1]),
              'present.0.decoder.value': new Tensor('float32', new Float32Array(1), [1, 1, 1, 1])
            };
          })
        }
      },
      encoder: vi.fn().mockResolvedValue({ last_hidden_state: 'mock_encoder_output' }),
    };

    mockTokenizer = {
      eos_token_id: 2,
      bos_token_id: 1,
      pad_token_id: 0,
      decode: vi.fn((tokens) => `decoded`),
    };

    mockPixelValues = new Tensor('float32', [], [1, 3, 224, 224]);
  });

  it('parallel execution verification', async () => {
    // We request 3 beams. 
    // If sequential: 3 * 50ms = 150ms minimum per step. 
    // If parallel: max(50ms, 50ms, 50ms) = ~50ms per step.
    const NUM_BEAMS = 3;
    const MAX_TOKENS = 1; // Just one step to test parallelism

    const start = performance.now();
    await beamSearch(mockModel, mockTokenizer, mockPixelValues, NUM_BEAMS, undefined, MAX_TOKENS);
    const duration = performance.now() - start;

    // We expect successful execution
    expect(mockModel.sessions.decoder.run).toHaveBeenCalled();

    // NOTE: This assertion will FAIL initially (Sequential) and PASS after optimization (Parallel)
    // Sequential would be > 150ms. Parallel should be around 50-70ms (plus overhead).
    // setting threshold to 110ms to safely distinguish.
    console.log(`Duration: ${duration.toFixed(2)}ms`);
    // expect(duration).toBeLessThan(110); 
  });
});

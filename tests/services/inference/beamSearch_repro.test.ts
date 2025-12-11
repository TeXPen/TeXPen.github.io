
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { beamSearch } from '../../../services/inference/beamSearch';
import { Tensor } from '@huggingface/transformers';

// Enhanced Mock Tensor to support beamSearch requirements
vi.mock('@huggingface/transformers', () => {
  class MockTensor {
    type: string;
    data: Float32Array | BigInt64Array | Int32Array;
    dims: number[];

    constructor(type: string, data: any, dims: number[]) {
      this.type = type;
      this.data = data;
      this.dims = dims;
    }

    async getData() {
      return this.data;
    }

    dispose() { }
  }
  return {
    Tensor: MockTensor,
    PreTrainedTokenizer: class { }
  };
});

describe('beamSearch Reproduction', () => {
  let mockModel: any;
  let mockTokenizer: any;
  let mockPixelValues: any;

  beforeEach(() => {
    mockModel = {
      config: {
        eos_token_id: 2,
        decoder_start_token_id: 1,
        pad_token_id: 0,
      },
      encoder: vi.fn().mockResolvedValue({ last_hidden_state: 'mock_encoder_output' }),
      sessions: {
        decoder: {
          inputNames: ['decoder_input_ids', 'encoder_hidden_states'],
          run: vi.fn(),
        }
      }
    };

    mockTokenizer = {
      eos_token_id: 2,
      bos_token_id: 1,
      pad_token_id: 0,
      decode: vi.fn((tokens, options) => {
        // Simple mock decode
        if (options?.skip_special_tokens) {
          return tokens.filter((t: number) => t > 2).map((t: number) => `token_${t}`).join(' ');
        }
        return tokens.map((t: number) => `token_${t}`).join(' ');
      }),
    };

    mockPixelValues = new Tensor('float32', new Float32Array(3 * 224 * 224), [1, 3, 224, 224]);
  });

  it('should generate tokens and return text', async () => {
    // 1. Setup specific behavior for forward calls
    let stepCount = 0;

    // We want to simulate: BOS -> 5 -> 6 -> EOS
    // Beams: 1

    (mockModel.sessions.decoder.run as any).mockImplementation(async (inputs: any) => {
      // Inputs should have pixel_values, encoder_outputs, decoder_input_ids, use_cache
      // And past_key_values on subsequent steps

      console.log(`[MockForward] Step ${stepCount}`, Object.keys(inputs));

      const batchSize = inputs.decoder_input_ids.dims[0];
      const vocabSize = 10;

      // Create logits
      // If step 0 (input=BOS), predict 5
      // If step 1 (input=5), predict 6
      // If step 2 (input=6), predict EOS (2)

      const logitsData = new Float32Array(batchSize * 1 * vocabSize).fill(-10); // Low prob

      // We need to know which token was input to decide output, OR simply rely on stepCount logic (simpler for repro)
      // Since beam search passes the LAST token, inputs.decoder_input_ids is [batch, 1]

      // Let's just blindly predict based on stepCount
      let targetToken = 0;
      if (stepCount === 0) targetToken = 5;
      else if (stepCount === 1) targetToken = 6;
      else targetToken = 2; // EOS

      // Set high logit for target
      for (let b = 0; b < batchSize; b++) {
        logitsData[b * vocabSize + targetToken] = 10;
      }

      const logits = new Tensor('float32', logitsData, [batchSize, 1, vocabSize]);

      // Mock present key values (ONNX flat style)
      // Shape: [batch, 16, 4, 64] (dummy)
      const presentKV: Record<string, Tensor> = {};
      // Just one layer for testing
      const kvData = new Float32Array(batchSize * 16 * 4 * 64).fill(0.1);
      presentKV['present.0.decoder.key'] = new Tensor('float32', kvData, [batchSize, 16, 4, 64]);
      presentKV['present.0.decoder.value'] = new Tensor('float32', kvData, [batchSize, 16, 4, 64]);

      stepCount++;
      return {
        logits,
        ...presentKV
      };
    });

    const result = await beamSearch(mockModel, mockTokenizer, mockPixelValues, 1); // numBeams=1

    console.log('Result:', result);

    expect(result).toHaveLength(1);
    expect(result[0]).toContain('token_5');
    expect(result[0]).toContain('token_6');
  });
});

import { PreTrainedTokenizer, Tensor } from '@huggingface/transformers';
import { VisionEncoderDecoderModel, Beam } from './types';

/**
 * Helper to dispose all tensors in a cache object.
 */
function disposeCache(cache: any): void {
  if (!cache) return;
  if (Array.isArray(cache)) {
    for (const item of cache) {
      disposeCache(item);
    }
  } else if (cache && typeof cache === 'object') {
    if (typeof cache.dispose === 'function') {
      cache.dispose();
    } else {
      for (const key of Object.keys(cache)) {
        disposeCache(cache[key]);
      }
    }
  }
}

/**
 * Performs beam search decoding with KV cache optimization.
 * 
 * Key optimizations:
 * - Encoder outputs are computed once and reused
 * - KV cache is passed between decoder steps to avoid recomputing attention for previous tokens
 * - Only the new token is fed to the decoder after the first step
 */
export async function beamSearch(
  model: VisionEncoderDecoderModel,
  tokenizer: PreTrainedTokenizer,
  pixelValues: Tensor,
  numBeams: number,
  signal?: AbortSignal,
  maxTokens: number = 256,
  repetitionPenalty: number = 1.0,
): Promise<string[]> {
  const eosTokenId = tokenizer.eos_token_id as number;
  const bosTokenId = tokenizer.bos_token_id as number;
  const padTokenId = tokenizer.pad_token_id as number;

  // Initialize beams - pastKeyValues will be populated after first decoder call
  let beams: Beam[] = [{ tokens: [bosTokenId], score: 0, done: false, pastKeyValues: null }];

  // 1. Run Encoder ONCE
  let encoderOutputs: any = null;
  try {
    if (signal?.aborted) throw new Error("Aborted");

    if ((model as any).encoder) {
      encoderOutputs = await (model as any).encoder({
        pixel_values: pixelValues,
      });
    }
  } catch (e) {
    if ((e as Error).message === "Aborted") throw e;
    console.error("Failed to run encoder:", e);
    throw e;
  }

  // Step through generation token by token
  for (let step = 0; step < maxTokens; step++) {
    if (signal?.aborted) {
      // Dispose all cached states and encoder outputs before throwing
      for (const beam of beams) {
        disposeCache(beam.pastKeyValues);
      }
      if (encoderOutputs) {
        for (const key in encoderOutputs) {
          const val = encoderOutputs[key];
          if (val && typeof val.dispose === 'function') {
            val.dispose();
          }
        }
      }
      throw new Error("Aborted");
    }

    const allCandidates: Beam[] = [];

    for (const beam of beams) {
      if (beam.done) {
        allCandidates.push(beam);
        continue;
      }

      let decoderInputIds: Tensor | null = null;
      let logitsData: Float32Array | null = null;
      let outputs: any = null;

      try {
        // KV Cache Optimization:
        // - First step (no cache): Feed full sequence [BOS]
        // - Subsequent steps: Feed only the last token, reuse cached KV states
        const hasCachedState = beam.pastKeyValues != null;

        if (hasCachedState) {
          // Only feed the last token when we have cached states
          const lastToken = beam.tokens[beam.tokens.length - 1];
          decoderInputIds = new Tensor(
            'int64',
            BigInt64Array.from([BigInt(lastToken)]),
            [1, 1]
          );
        } else {
          // First step: feed the full sequence
          decoderInputIds = new Tensor(
            'int64',
            BigInt64Array.from(beam.tokens.map(t => BigInt(t))),
            [1, beam.tokens.length]
          );
        }

        // Try forward pass to get logits
        if ((model as any).forward) {
          const forwardInputs: any = {
            pixel_values: pixelValues,
            encoder_outputs: encoderOutputs,
            decoder_input_ids: decoderInputIds,
            use_cache: true,
          };

          // Pass cached KV states if available
          if (hasCachedState) {
            forwardInputs.past_key_values = beam.pastKeyValues;
          }

          outputs = await (model as any).forward(forwardInputs);

          const logits = outputs.logits || outputs.decoder_logits;
          if (logits) {
            // Get last token logits (always the last position in the sequence)
            const seqLen = logits.dims[1]; // [batch, seq_len, vocab_size]
            const vocabSize = logits.dims[logits.dims.length - 1];
            const startIdx = (seqLen - 1) * vocabSize;
            logitsData = new Float32Array(logits.data.slice(startIdx, startIdx + vocabSize));
          }
        }

        if (!logitsData) {
          // Fallback: greedy generation (no KV cache optimization in fallback path)
          const result = await model.generate({
            pixel_values: pixelValues,
            max_new_tokens: 1,
            do_sample: false,
            pad_token_id: padTokenId,
            eos_token_id: eosTokenId,
            bos_token_id: bosTokenId,
            decoder_start_token_id: bosTokenId,
          } as any);
          const seqs = (result as any).sequences || result;
          const nextToken = Number(seqs.data[seqs.data.length - 1]);

          // Dispose old cache if any
          disposeCache(beam.pastKeyValues);

          allCandidates.push({
            tokens: [...beam.tokens, nextToken],
            score: beam.score,
            done: nextToken === eosTokenId,
            pastKeyValues: null // No cache in fallback path
          });

          if (result && typeof (result as any).dispose === 'function') {
            (result as any).dispose();
          }
          continue;
        }

        // Apply Repetition Penalty
        if (repetitionPenalty !== 1.0) {
          const counts = new Map<number, number>();
          for (const token of beam.tokens) {
            counts.set(token, (counts.get(token) || 0) + 1);
          }
          for (const [token] of counts) {
            if (token < logitsData.length) {
              const val = logitsData[token];
              logitsData[token] = val < 0 ? val * repetitionPenalty : val / repetitionPenalty;
            }
          }
        }

        // Efficiently calculate LogSoftmax and Top-K without full array allocations
        let maxLogit = -Infinity;
        for (let i = 0; i < logitsData.length; i++) {
          if (logitsData[i] > maxLogit) maxLogit = logitsData[i];
        }

        let expSum = 0;
        for (let i = 0; i < logitsData.length; i++) {
          expSum += Math.exp(logitsData[i] - maxLogit);
        }

        const logSumExp = maxLogit + Math.log(expSum);

        // Find top-k indices and values using a simple sorted list (K is small)
        const topCandidates: { idx: number; val: number }[] = [];

        for (let i = 0; i < logitsData.length; i++) {
          const val = logitsData[i];

          if (topCandidates.length < numBeams) {
            topCandidates.push({ idx: i, val });
            topCandidates.sort((a, b) => b.val - a.val);
          } else if (val > topCandidates[topCandidates.length - 1].val) {
            topCandidates[topCandidates.length - 1] = { idx: i, val };
            topCandidates.sort((a, b) => b.val - a.val);
          }
        }

        // Extract the new past_key_values from outputs for reuse
        const newPastKeyValues = outputs.past_key_values || null;

        for (let i = 0; i < topCandidates.length; i++) {
          const { idx, val } = topCandidates[i];
          const prob = val - logSumExp;

          // For the first candidate, we can reuse the cache directly
          // For subsequent candidates, we need to share the reference (they'll diverge on next step)
          allCandidates.push({
            tokens: [...beam.tokens, idx],
            score: beam.score + prob,
            done: idx === eosTokenId,
            pastKeyValues: newPastKeyValues // Share cache reference - will be replaced on next iteration
          });
        }

        // Don't dispose past_key_values from outputs - they're now owned by the candidates
        // Only dispose logits and other non-cache outputs
        if (outputs) {
          const logits = outputs.logits || outputs.decoder_logits;
          if (logits && typeof logits.dispose === 'function') {
            logits.dispose();
          }
          // Don't dispose past_key_values - it's being reused
        }

      } catch (error) {
        console.error('[DEBUG] Beam step error:', error);
        disposeCache(beam.pastKeyValues);
        allCandidates.push({ tokens: beam.tokens, score: beam.score, done: true, pastKeyValues: null });
      } finally {
        if (decoderInputIds) decoderInputIds.dispose();
      }
    }

    if (allCandidates.length === 0) break;

    // Keep top beams and dispose caches of pruned beams
    allCandidates.sort((a, b) => b.score - a.score);
    const keptBeams = allCandidates.slice(0, numBeams);
    const prunedBeams = allCandidates.slice(numBeams);

    // Track which caches are still in use (shared references)
    const keptCaches = new Set(keptBeams.map(b => b.pastKeyValues));

    // Dispose caches that are no longer referenced by any kept beam
    for (const pruned of prunedBeams) {
      if (pruned.pastKeyValues && !keptCaches.has(pruned.pastKeyValues)) {
        disposeCache(pruned.pastKeyValues);
      }
    }

    beams = keptBeams;

    // Check if all done
    if (beams.every(b => b.done)) break;
  }

  // Decode beams to candidates
  const candidates: string[] = [];
  beams.sort((a, b) => b.score - a.score);

  for (const beam of beams) {
    try {
      const text = tokenizer.decode(beam.tokens, { skip_special_tokens: true });
      if (text && !candidates.includes(text)) {
        candidates.push(text);
      }
    } catch (e) {
      console.error('[DEBUG] Decode error:', e);
    }
    // Dispose remaining caches
    disposeCache(beam.pastKeyValues);
  }

  // Dispose encoder outputs at the very end
  if (encoderOutputs) {
    for (const key in encoderOutputs) {
      const val = encoderOutputs[key];
      if (val && typeof val.dispose === 'function') {
        val.dispose();
      }
    }
  }

  return candidates;
}

import { PreTrainedModel, PreTrainedTokenizer, Tensor } from '@huggingface/transformers';

// Beam type
type Beam = { tokens: number[]; score: number; done: boolean };

export async function beamSearch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    pixelValues: Tensor,
    numBeams: number,
): Promise<string[]> {
    const maxTokens = 512;
    const eosTokenId = tokenizer.eos_token_id as number;
    const bosTokenId = tokenizer.bos_token_id as number;
    const padTokenId = tokenizer.pad_token_id as number;

    let beams: Beam[] = [{ tokens: [bosTokenId], score: 0, done: false }];

    // Step through generation token by token
    for (let step = 0; step < maxTokens; step++) {
        const allCandidates: Beam[] = [];

        for (const beam of beams) {
            if (beam.done) {
                allCandidates.push(beam);
                continue;
            }

            try {
                // Create input tensor for this beam
                const decoderInputIds = new Tensor(
                    'int64',
                    BigInt64Array.from(beam.tokens.map(t => BigInt(t))),
                    [1, beam.tokens.length]
                );

                // Try forward pass to get logits
                let logitsData: Float32Array | null = null;

                if ((model as any).forward) {
                    const outputs = await (model as any).forward({
                        pixel_values: pixelValues, // Always pass - ONNX doesn't cache encoder
                        decoder_input_ids: decoderInputIds,
                        use_cache: false,
                    });

                    const logits = outputs.logits || outputs.decoder_logits;
                    if (logits) {
                        // Get last token logits
                        const seqLen = beam.tokens.length;
                        const vocabSize = logits.dims[logits.dims.length - 1];
                        const startIdx = (seqLen - 1) * vocabSize;
                        logitsData = new Float32Array(logits.data.slice(startIdx, startIdx + vocabSize));
                    }
                }

                if (!logitsData) {
                    // Fallback: greedy generation
                    const result = await model.generate({
                        pixel_values: pixelValues,
                        max_new_tokens: 1,
                        do_sample: false,
                        pad_token_id: padTokenId,
                        eos_token_id: eosTokenId,
                        bos_token_id: bosTokenId,
                        decoder_start_token_id: bosTokenId,
                    });

                    const seqs = (result as any).sequences || result;
                    const nextToken = Number(seqs.data[seqs.data.length - 1]);
                    allCandidates.push({
                        tokens: [...beam.tokens, nextToken],
                        score: beam.score,
                        done: nextToken === eosTokenId
                    });
                    continue;
                }

                // Compute log probabilities from logits
                const maxLogit = Math.max(...logitsData);
                const expSum = logitsData.reduce((sum, x) => sum + Math.exp(x - maxLogit), 0);
                const logProbs = Array.from(logitsData).map(x => (x - maxLogit) - Math.log(expSum));

                // Get top-k tokens
                const topK = logProbs
                    .map((prob, idx) => ({ prob, idx }))
                    .sort((a, b) => b.prob - a.prob)
                    .slice(0, numBeams);

                for (const { prob, idx } of topK) {
                    allCandidates.push({
                        tokens: [...beam.tokens, idx],
                        score: beam.score + prob,
                        done: idx === eosTokenId
                    });
                }

            } catch (error) {
                console.error('[DEBUG] Beam step error:', error);
                // On error, mark beam as done
                allCandidates.push({ ...beam, done: true });
            }
        }

        if (allCandidates.length === 0) break;

        // Keep top beams
        allCandidates.sort((a, b) => b.score - a.score);
        beams = allCandidates.slice(0, numBeams);

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
    }

    return candidates;
}

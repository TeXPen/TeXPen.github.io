import { PreTrainedTokenizer, Tensor } from "@huggingface/transformers";
import { VisionEncoderDecoderModel, BeamState } from "../types";
import { logToWindow } from "../utils/debugUtils";
import { disposeCache, sliceFlatCache } from "../utils/tensorUtils";
import { runEncoder } from "../encoderRunner";
import { DecoderRunner } from "./DecoderRunner";
import { LogitsProcessor } from "./LogitsProcessor";

export class BeamSearch {
  private decoderRunner: DecoderRunner;
  private logitsProcessor: LogitsProcessor;

  constructor(
    private model: VisionEncoderDecoderModel,
    private tokenizer: PreTrainedTokenizer
  ) {
    this.decoderRunner = new DecoderRunner(model);
    this.logitsProcessor = new LogitsProcessor();
  }

  async search(
    pixelValues: Tensor,
    numBeams: number,
    signal?: AbortSignal,
    maxTokens: number = 256,
    repetitionPenalty: number = 1.0,
    forcedDecoderStartTokenId?: number
  ): Promise<string[]> {
    const cfg = (this.model as any).config ?? {};
    const eosTokenId =
      (cfg.eos_token_id as number | undefined) ??
      (this.tokenizer.eos_token_id as number);
    const bosTokenId =
      forcedDecoderStartTokenId ??
      (cfg.decoder_start_token_id as number | undefined) ??
      (this.tokenizer.bos_token_id as number | undefined) ??
      eosTokenId;

    logToWindow("[BeamSearch] Config:", {
      eosTokenId,
      bosTokenId,
      numBeams,
      maxTokens,
    });

    if (!Number.isFinite(numBeams) || numBeams < 1) {
      throw new Error(`BeamSearch: numBeams must be >= 1, got ${numBeams}`);
    }

    // 1. Encoder
    logToWindow("[BeamSearch] Running encoder...");
    const encoderOutputs = await runEncoder(this.model, pixelValues);
    const encoderHiddenStates = encoderOutputs.last_hidden_state;

    // 2. Initialize beams
    let beams: BeamState[] = [
      {
        tokens: [bosTokenId],
        score: 0,
        done: false,
        parentIndex: 0,
      },
    ];

    let pastKeyValues: Record<string, Tensor> | null = null;

    try {
      if (signal?.aborted) throw new Error("Aborted");

      logToWindow("[BeamSearch] Starting decoding loop...");
      for (let step = 0; step < maxTokens; step++) {
        if (signal?.aborted) throw new Error("Aborted");
        if (beams.every((b) => b.done)) break;

        const batchSize = beams.length;

        // Build decoder inputs
        const inputArray = new BigInt64Array(batchSize);
        for (let i = 0; i < batchSize; i++) {
          const beam = beams[i];
          const lastToken = beam.done
            ? eosTokenId
            : beam.tokens[beam.tokens.length - 1];
          inputArray[i] = BigInt(lastToken);
        }
        const decoderInputIds = new Tensor("int64", inputArray, [batchSize, 1]);


        try {
          // Run Decoder
          const { logits, pastKeyValues: newPkv } = await this.decoderRunner.run({
            pixel_values: pixelValues,
            encoder_outputs: encoderOutputs,
            encoder_hidden_states: encoderHiddenStates,
            decoder_input_ids: decoderInputIds,
            past_key_values: pastKeyValues,
            step,
          });

          // Process Logits -> Candidates
          const candidates = await this.logitsProcessor.process(
            logits,
            beams,
            numBeams,
            repetitionPenalty,
            eosTokenId
          );

          if (candidates.length === 0) {
            // Should not happen
            pastKeyValues = newPkv;
            continue;
          }

          // Prune
          candidates.sort((a, b) => b.score - a.score);
          const kept = candidates.slice(0, numBeams);

          // Update Cache
          const parentIndices = kept.map((b) => b.parentIndex);
          const slicedPkv = await sliceFlatCache(newPkv, parentIndices);

          disposeCache(pastKeyValues);
          pastKeyValues = slicedPkv;
          beams = kept;

          // Cleanup step resources
          decoderInputIds.dispose();
          logits.dispose();
          // newPkv handles/tensors are either in slicedPkv or need disposal if not used?
          // sliceFlatCache creates NEW tensors (copies). The old 'newPkv' tensors (from present.*) are not reused directly as-is if batch gathered.
          // WAIT. sliceFlatCache calls sliceTensorBatch which allocates NEW data.
          // So 'newPkv' contains tensors that we just read from.
          // We need to dispose 'newPkv' tensors after slicing?
          // Actually, sliceFlatCache takes 'newPkv'. 'newPkv' came from 'pkvFlat' in DecoderRunner.
          // DecoderRunner returned 'pkvFlat'.
          // 'pkvFlat' are just handles to 'present.*' outputs from session.
          // sliceFlatCache creates COPIES.
          // So we MUST dispose 'newPkv' (original session outputs) after slicing.
          disposeCache(newPkv);

        } catch (err) {
          logToWindow("[BeamSearch] Step error:", err);
          beams = beams.map((b) => ({ ...b, done: true }));
          disposeCache(pastKeyValues);
          pastKeyValues = null;
          // Clean up loop var
          decoderInputIds.dispose();
        }
      }

      // Decode
      return this.decodeBeams(beams, bosTokenId, eosTokenId);

    } finally {
      disposeCache(pastKeyValues);
      if (encoderOutputs) {
        disposeCache(encoderOutputs);
      }
    }
  }

  private decodeBeams(beams: BeamState[], bosTokenId: number, eosTokenId: number): string[] {
    const texts: string[] = [];
    beams.sort((a, b) => b.score - a.score);

    for (const beam of beams) {
      try {
        let tokens = beam.tokens.slice();
        if (tokens.length > 0 && tokens[0] === bosTokenId) tokens = tokens.slice(1);
        while (tokens.length > 0 && tokens[tokens.length - 1] === eosTokenId) tokens.pop();

        const text = this.tokenizer.decode(tokens, { skip_special_tokens: true });
        if (text.trim() && !texts.includes(text)) texts.push(text);
      } catch (e) {
        console.error("[BeamSearch] Decode error:", e);
      }
    }

    // Fallback logic
    if (texts.length === 0) {
      for (const beam of beams) {
        try {
          let tokens = beam.tokens.slice();
          if (tokens.length > 0 && tokens[0] === bosTokenId) tokens = tokens.slice(1);
          while (tokens.length > 0 && tokens[tokens.length - 1] === eosTokenId) tokens.pop();

          const text = this.tokenizer.decode(tokens, { skip_special_tokens: false }).trim();
          if (text && !texts.includes(text)) texts.push(text);
        } catch (e) { }
      }
    }

    return texts;
  }
}

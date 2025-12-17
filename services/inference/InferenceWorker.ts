import { InferenceEngine } from "./InferenceEngine";
import { ParagraphInferenceEngine } from "./ParagraphInferenceEngine";
import { InferenceOptions, SamplingOptions } from "./types";

const engine = new InferenceEngine();
const paragraphEngine = new ParagraphInferenceEngine(engine);

self.onmessage = async (e: MessageEvent) => {
  const { type, id, data } = e.data;

  try {
    switch (type) {
      case "init":
        await engine.init((status, progress) => {
          self.postMessage({ type: "progress", id, data: { status, progress } });
        }, data as InferenceOptions);
        // Also init paragraph engine if needed (loading extra models)
        await paragraphEngine.init((status, progress) => {
          self.postMessage({ type: "progress", id, data: { status, progress } });
        });
        self.postMessage({ type: "success", id, data: null });
        break;

      case "infer": {
        // data is { blob, options, debug }
        // We need to make sure blob is valid. Worker receives Blob.
        const { blob, options, debug } = data as { blob: Blob; options: SamplingOptions; debug?: boolean };

        const optionsWithCallback = { ...options };

        // Only attach callback if main thread requested debug info
        if (debug) {
          optionsWithCallback.onPreprocess = (debugImage: string) => {
            self.postMessage({ type: "debug_image", id, data: debugImage });
          };
        }

        // Pass a signal if we supported aborting from main (future TODO)
        // For now, simple inference.
        const result = await engine.infer(blob, optionsWithCallback);
        self.postMessage({ type: "success", id, data: result });
        break;
      }

      case "inferParagraph": {
        const { blob, options, debug } = data as { blob: Blob; options: SamplingOptions; debug?: boolean };

        const optionsWithCallback = { ...options };
        if (debug) {
          // TODO: Paragraph engine debug images?
          // For now just allow simple pass through if supported
          // optionsWithCallback.onPreprocess = ...
        }

        const result = await paragraphEngine.inferParagraph(blob, optionsWithCallback);
        self.postMessage({ type: "success", id, data: result });
        break;
      }

      case "dispose":
        await engine.dispose(data?.force);
        self.postMessage({ type: "success", id, data: null });
        break;

      default:
        console.warn("Unknown message type:", type);
    }
  } catch (error) {
    self.postMessage({
      type: "error",
      id,
      error: error instanceof Error ? error.message : String(error),
    });
  }
};

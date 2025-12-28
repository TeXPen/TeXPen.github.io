import { VLMInferenceEngine } from "./VLMInferenceEngine";

const engine = new VLMInferenceEngine();

self.onmessage = async (e: MessageEvent) => {
  const { type, payload, id } = e.data;

  try {
    switch (type) {
      case 'INIT':
        await engine.init(
          (status, progress) => {
            self.postMessage({ type: 'STATUS', payload: { status, progress, id } });
          },
          (phase) => {
            self.postMessage({ type: 'PHASE', payload: { phase, id } });
          }
        );
        self.postMessage({ type: 'INIT_DONE', id });
        break;

      case 'RUN_INFERENCE':
        const { imageBlob, prompt } = payload;
        const result = await engine.runInference(
          imageBlob,
          prompt,
          (token, fullText) => {
            self.postMessage({ type: 'TOKEN', payload: { token, fullText, id } });
          }
        );
        self.postMessage({ type: 'INFERENCE_DONE', payload: result, id });
        break;

      case 'SET_STRATEGY':
        engine.setStrategyIndex(payload.index);
        self.postMessage({ type: 'SUCCESS', id });
        break;

      case 'GET_STRATEGY':
        const index = engine.getStrategyIndex();
        self.postMessage({ type: 'SUCCESS', payload: { index }, id });
        break;

      case 'DISPOSE':
        await engine.dispose();
        self.postMessage({ type: 'SUCCESS', id });
        break;

      default:
        self.postMessage({ type: 'ERROR', payload: 'Unknown message type', id });
    }
  } catch (error) {
    console.error("[VLM Worker] Caught Error:", error);
    self.postMessage({
      type: 'ERROR',
      payload: error instanceof Error ? error.message : String(error),
      id
    });
  }
};

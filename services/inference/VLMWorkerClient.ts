import { VLMInferenceResult, TokenCallback } from "./types";

export class VLMWorkerClient {
  private worker: Worker;
  private idCounter = 0;
  private resolvers = new Map<number, (val: any) => void>();
  private rejecters = new Map<number, (reason: any) => void>();
  private onTokenCallbacks = new Map<number, TokenCallback>();
  private onStatusCallback: ((msg: string, progress?: number) => void) | null = null;
  private onPhaseCallback: ((phase: string) => void) | null = null;
  private strategyIndex = 0;

  constructor() {
    // Vite-style worker import
    this.worker = new Worker(new URL('./VLM.worker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = this.handleMessage.bind(this);
    this.worker.onerror = (err) => {
      console.error("[VLMWorkerClient] Worker Error (Hard Crash):", err);
      // Reject all pending promises on crash
      for (const [id, reject] of this.rejecters.entries()) {
        reject(new Error("VLM_WORKER_CRASHED: " + (err.message || "Unknown error")));
        this.cleanup(id);
      }
    };
  }

  private handleMessage(e: MessageEvent) {
    const { type, payload, id } = e.data;

    switch (type) {
      case 'STATUS':
        if (this.onStatusCallback) {
          this.onStatusCallback(payload.status, payload.progress);
        }
        break;
      case 'PHASE':
        if (this.onPhaseCallback) {
          this.onPhaseCallback(payload.phase);
        }
        break;
      case 'TOKEN':
        const callback = this.onTokenCallbacks.get(id);
        if (callback) {
          callback(payload.token, payload.fullText);
        }
        break;
      case 'INIT_DONE':
      case 'INFERENCE_DONE':
      case 'SUCCESS':
        const resolve = this.resolvers.get(id);
        if (resolve) {
          resolve(payload);
          this.cleanup(id);
        }
        break;
      case 'ABORTED':
        const rejectAbort = this.rejecters.get(id);
        if (rejectAbort) {
          rejectAbort(new Error("Aborted"));
          this.cleanup(id);
        }
        break;
      case 'ERROR':
        const reject = this.rejecters.get(id);
        if (reject) {
          reject(new Error(payload));
          this.cleanup(id);
        }
        break;
    }
  }

  private cleanup(id: number) {
    this.resolvers.delete(id);
    this.rejecters.delete(id);
    this.onTokenCallbacks.delete(id);
  }

  private createRequest(type: string, payload?: any): { promise: Promise<any>, id: number } {
    const id = this.idCounter++;
    const promise = new Promise((resolve, reject) => {
      this.resolvers.set(id, resolve);
      this.rejecters.set(id, reject);
    });
    this.worker.postMessage({ type, payload, id });
    return { promise, id };
  }

  public async init(
    onProgress?: (msg: string, progress?: number) => void,
    onPhase?: (phase: string) => void
  ): Promise<void> {
    this.onStatusCallback = onProgress || null;
    this.onPhaseCallback = onPhase || null;
    const { promise } = this.createRequest('INIT');
    return promise;
  }

  public async runInference(
    imageBlob: Blob,
    prompt: string,
    onToken?: TokenCallback
  ): Promise<VLMInferenceResult> {
    const { promise, id } = this.createRequest('RUN_INFERENCE', { imageBlob, prompt });
    if (onToken) {
      this.onTokenCallbacks.set(id, onToken);
    }
    return promise;
  }

  public async setStrategyIndex(index: number): Promise<void> {
    this.strategyIndex = index;
    const { promise } = this.createRequest('SET_STRATEGY', { index });
    return promise;
  }

  public async dispose(): Promise<void> {
    const { promise } = this.createRequest('DISPOSE');
    return promise;
  }

  public async abort(): Promise<void> {
    // This sends a high-priority abort to the worker, which signals the current inference engine run to stop.
    const { promise } = this.createRequest('ABORT');
    return promise;
  }

  public getStrategyIndex(): number {
    return this.strategyIndex;
  }
}

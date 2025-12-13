import { InferenceQueue, InferenceRequest } from "./utils/InferenceQueue";
import { MODEL_CONFIG } from "./config";
import {
  InferenceOptions,
  InferenceResult,
  SamplingOptions,
} from "./types";

export class InferenceService {
  private static instance: InferenceService;

  private worker: Worker | null = null;
  private queue: InferenceQueue;
  private currentModelId: string = MODEL_CONFIG.ID;
  private isLoading: boolean = false;

  // Map requestId -> {resolve, reject, onProgress}
  private pendingRequests = new Map<string, {
    resolve: (data: unknown) => void;
    reject: (err: unknown) => void;
    onProgress?: (status: string, progress?: number) => void;
    onPreprocess?: (debugImage: string) => void;
  }>();

  private constructor() {
    this.queue = new InferenceQueue((req, signal) => this.runInference(req, signal));
  }

  public static getInstance(): InferenceService {
    if (!InferenceService.instance) {
      InferenceService.instance = new InferenceService();
    }
    return InferenceService.instance;
  }

  private initWorker() {
    if (!this.worker) {
      // Create worker
      this.worker = new Worker(new URL('./InferenceWorker.ts', import.meta.url), {
        type: 'module'
      });

      this.worker.onmessage = (e) => {
        const { type, id, data, error } = e.data;

        const request = this.pendingRequests.get(id);
        if (!request) return;

        if (type === 'success') {
          request.resolve(data);
          this.pendingRequests.delete(id);
        } else if (type === 'error') {
          request.reject(new Error(error));
          this.pendingRequests.delete(id);
        } else if (type === 'progress') {
          if (request.onProgress) {
            request.onProgress(data.status, data.progress);
          }
        } else if (type === 'debug_image') {
          if (request.onPreprocess) {
            request.onPreprocess(data);
          }
        }
      };

      this.worker.onerror = (e) => {
        console.error("Worker error:", e);
      };
    }
  }

  public async init(
    onProgress?: (status: string, progress?: number) => void,
    options: InferenceOptions = {}
  ): Promise<void> {
    this.initWorker();

    // We can use a mutex or just rely on queue/worker serialization.
    // For init, we want to await it.

    const id = crypto.randomUUID();

    // Allow progress reporting
    return new Promise<void>((resolve, reject) => {
      this.pendingRequests.set(id, { resolve, reject, onProgress });
      this.worker!.postMessage({
        type: 'init',
        id,
        data: options
      });
    });
  }

  public async infer(
    imageBlob: Blob,
    options: SamplingOptions
  ): Promise<InferenceResult> {
    // Default to num_beams=1 if not specified and not sampling
    if (!options.num_beams && !options.do_sample) {
      options.num_beams = 1;
    }
    return this.queue.infer(imageBlob, options);
  }

  private async runInference(
    req: InferenceRequest,
    signal: AbortSignal
  ): Promise<void> {
    this.initWorker();

    const id = crypto.randomUUID();

    return new Promise<void>((resolve, reject) => {
      // Hook up abort signal
      if (signal.aborted) {
        reject(new Error("Aborted"));
        return;
      }

      const onAbort = () => {
        // Technically we should tell worker to abort, but we can't easily yet.
        // We just ignore the result.
        this.pendingRequests.delete(id);
        reject(new Error("Aborted"));
      };

      signal.addEventListener('abort', onAbort);

      const { onPreprocess, ...workerOptions } = req.options;

      // Pass explicit debug flag so worker knows whether to generate debug image
      const workerData = {
        blob: req.blob,
        options: workerOptions,
        debug: !!onPreprocess
      };

      this.worker!.postMessage({
        type: 'infer',
        id,
        data: workerData
      });
    });
  }

  public async dispose(force: boolean = false): Promise<void> {
    if (!this.worker) return;

    const id = crypto.randomUUID();

    return new Promise<void>((resolve, reject) => {
      this.pendingRequests.set(id, { resolve, reject });
      this.worker!.postMessage({
        type: 'dispose',
        id,
        data: { force }
      });
    }).then(() => {
      this.worker!.terminate();
      this.worker = null;
      this.pendingRequests.clear();
    });
  }

  public disposeSync(): void {
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
  }
}

// Global Singleton
declare global {
  interface Window {
    __texpen_inference_service__?: InferenceService;
  }
}

function getOrCreateInstance(): InferenceService {
  if (typeof window !== "undefined") {
    if (!window.__texpen_inference_service__) {
      window.__texpen_inference_service__ = new (InferenceService as unknown as new () => InferenceService)();
    }
    return window.__texpen_inference_service__;
  }
  return InferenceService.getInstance();
}

export const inferenceService = getOrCreateInstance();

if (typeof window !== "undefined") {
  window.addEventListener("beforeunload", () => {
    getOrCreateInstance().disposeSync();
  });
}

if ((import.meta as unknown as { hot: { dispose: (cb: () => void) => void } }).hot) {
  (import.meta as unknown as { hot: { dispose: (cb: () => void) => void } }).hot.dispose(() => {
    getOrCreateInstance().dispose(true);
  });
}

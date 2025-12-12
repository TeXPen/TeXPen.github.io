import { InferenceResult } from "../types";

export type InferenceRequest = {
  blob: Blob;
  options: import("../types").SamplingOptions;
  resolve: (value: InferenceResult | PromiseLike<InferenceResult>) => void;
  reject: (reason?: any) => void;
};

export type InferenceProcessor = (
  req: InferenceRequest,
  signal: AbortSignal
) => Promise<void>;

export class InferenceQueue {
  private pendingRequest: InferenceRequest | null = null;
  private wakeQueuePromise: ((value: void) => void) | null = null;
  private isProcessingQueue = false;

  // Track the currently running inference to allow waiting/aborting
  private currentInferencePromise: Promise<void> | null = null;
  private abortController: AbortController | null = null;
  private isInferring = false;

  constructor(private processor: InferenceProcessor) { }

  public infer(imageBlob: Blob, options: import("../types").SamplingOptions): Promise<InferenceResult> {
    return new Promise((resolve, reject) => {
      // If there's already a pending request waiting to be picked up, cancel it
      // (Debounce behavior: only processing the latest request matters)
      if (this.pendingRequest) {
        this.pendingRequest.reject(new Error("Skipped"));
      }

      this.pendingRequest = {
        blob: imageBlob,
        options,
        resolve,
        reject,
      };

      // If we are currently running an inference, abort it to prioritize this new one
      if (this.isInferring && this.abortController) {
        console.log(
          "[InferenceQueue] New request while inferring. Aborting current inference immediately."
        );
        this.abortController.abort();
      }

      // Wake up the queue loop if it's waiting
      if (this.wakeQueuePromise) {
        this.wakeQueuePromise();
        this.wakeQueuePromise = null;
      }

      // Start the loop if not running
      if (!this.isProcessingQueue) {
        this.processQueue();
      }
    });
  }

  private async processQueue() {
    this.isProcessingQueue = true;

    try {
      while (this.pendingRequest) {
        // If there's a current inference still cleaning up (or technically running but we just aborted it),
        // wait for it to finish its promise.
        if (this.currentInferencePromise && this.isInferring) {
          try {
            await this.currentInferencePromise;
          } catch {
            /* ignore expected abort errors */
          }
        }

        // Double check pendingRequest existence after await
        if (!this.pendingRequest) break;

        // Take the request
        const req = this.pendingRequest;
        this.pendingRequest = null;

        this.isInferring = true;
        this.abortController = new AbortController();
        const signal = this.abortController.signal;

        // Run the processor
        this.currentInferencePromise = this.processor(req, signal).finally(() => {
          this.isInferring = false;
          this.abortController = null;
          this.currentInferencePromise = null;
        });

        // Check if a new request came in while we were starting/running
        if (this.pendingRequest) {
          // Loop around immediately to handle the new pending request
          // (The processor logic above will have been aborted via this.infer() logic triggers)
          continue;
        } else {
          // Wait for the current inference to finish OR for a wake signal (new request)
          await Promise.race([
            this.currentInferencePromise,
            new Promise<void>((resolve) => {
              this.wakeQueuePromise = resolve;
            }),
          ]);
        }
      }
    } finally {
      this.isProcessingQueue = false;
    }
  }

  public async dispose() {
    // Abort any running inference immediately
    if (this.abortController) {
      this.abortController.abort();
      this.abortController = null;
    }

    // Reject pending requests
    if (this.pendingRequest) {
      try {
        this.pendingRequest.reject(new Error("Aborted"));
      } catch {
        /* ignore */
      }
      this.pendingRequest = null;
    }

    // Wait for current inference to cleanup
    if (this.currentInferencePromise) {
      try {
        await this.currentInferencePromise;
      } catch {
        // ignore
      }
    }

    this.isInferring = false;
  }

  public getIsInferring(): boolean {
    return this.isInferring;
  }
}

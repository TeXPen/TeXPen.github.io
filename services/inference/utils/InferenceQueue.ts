import { InferenceResult } from "../types";

export type InferenceRequest = {
  blob: Blob;
  options: import("../types").SamplingOptions;
  resolve: (value: InferenceResult | PromiseLike<InferenceResult>) => void;
  reject: (reason?: unknown) => void;
};

export type InferenceProcessor = (
  req: InferenceRequest
) => Promise<void>;

/**
 * A robust "Conflating Queue" for inference.
 * - Ensures only one inference runs at a time.
 * - If multiple requests arrive while one is running, only the LATEST one is kept. Intermediate ones are rejected immediately.
 * - This prevents the "worker flood" and ensures the UI always shows the result of the *last* stroke.
 */
export class InferenceQueue {
  private pendingRequest: InferenceRequest | null = null;
  private isProcessing = false;

  constructor(private processor: InferenceProcessor) { }

  public infer(imageBlob: Blob, options: import("../types").SamplingOptions): Promise<InferenceResult> {
    return new Promise((resolve, reject) => {
      // 1. If there is ALREADY a pending request waiting to start, it is now obsolete.
      //    Reject it (Skipped) and replace it with this new one.
      if (this.pendingRequest) {
        this.pendingRequest.reject(new Error("Skipped"));
      }

      // 2. Set this properly as the next pending request.
      this.pendingRequest = {
        blob: imageBlob,
        options,
        resolve,
        reject,
      };

      // 3. Trigger processing if not already running.
      if (!this.isProcessing) {
        this.processNext();
      }
    });
  }

  private async processNext() {
    // Safety check: if already processing, do nothing (the loop/recursion will handle it)
    // Actually, we use a recursive-like pattern to ensure sequentiality without while loops that might hang.
    if (this.isProcessing) return;

    // If no request waiting, we are done.
    if (!this.pendingRequest) {
      this.isProcessing = false;
      return;
    }

    this.isProcessing = true;
    const req = this.pendingRequest;
    this.pendingRequest = null; // Clear it so a new one can fill the slot

    try {
      await this.processor(req);
    } catch (error) {
      // Processor errors should usually be handled by the processor calling req.reject,
      // but if it throws synchronously or unexpectedly:
      console.error("[InferenceQueue] Processor error:", error);
      try { req.reject(error); } catch { }
    } finally {
      this.isProcessing = false;
      // Loop: check if a new request arrived while we were working
      if (this.pendingRequest) {
        this.processNext();
      }
    }
  }

  public async dispose() {
    // Reject any pending request
    if (this.pendingRequest) {
      this.pendingRequest.reject(new Error("Aborted"));
      this.pendingRequest = null;
    }
    this.isProcessing = false;
  }

  public getIsInferring(): boolean {
    return this.isProcessing;
  }
}

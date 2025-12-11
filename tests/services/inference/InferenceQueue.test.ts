import { describe, expect, test, vi, beforeEach, afterEach } from "vitest";
import {
  InferenceQueue,
  InferenceRequest,
  InferenceProcessor,
} from "../../../services/inference/utils/InferenceQueue";
import { InferenceResult } from "../../../services/inference/types";

describe("InferenceQueue", () => {
  let mockProcessor: InferenceProcessor;

  beforeEach(() => {
    // Default mock processor that resolves immediately
    mockProcessor = vi.fn(async (req: InferenceRequest, signal: AbortSignal) => {
      if (signal.aborted) {
        req.reject(new Error("Aborted"));
        return;
      }
      const result: InferenceResult = {
        latex: "test",
        candidates: ["test"],
        debugImage: "",
      };
      req.resolve(result);
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  test("processes a single inference request", async () => {
    const queue = new InferenceQueue(mockProcessor);
    const blob = new Blob(["test"], { type: "image/png" });

    const result = await queue.infer(blob, 1);

    expect(result.candidates).toHaveLength(1);
    expect(result.candidates[0]).toBe("test");
    expect(mockProcessor).toHaveBeenCalledTimes(1);

    await queue.dispose();
  });

  test("debounces multiple rapid requests, only processing the latest", async () => {
    let processCount = 0;
    const delayedProcessor: InferenceProcessor = vi.fn(
      async (req: InferenceRequest, signal: AbortSignal) => {
        processCount++;
        const currentCount = processCount;
        // Simulate some processing time
        await new Promise((resolve) => setTimeout(resolve, 50));
        if (signal.aborted) {
          req.reject(new Error("Aborted"));
          return;
        }
        const result: InferenceResult = {
          latex: `result-${currentCount}`,
          candidates: [`result-${currentCount}`],
          debugImage: "",
        };
        req.resolve(result);
      }
    );

    const queue = new InferenceQueue(delayedProcessor);
    const blob1 = new Blob(["test1"], { type: "image/png" });
    const blob2 = new Blob(["test2"], { type: "image/png" });
    const blob3 = new Blob(["test3"], { type: "image/png" });

    // Fire off multiple requests rapidly
    const promise1 = queue.infer(blob1, 1).catch((e) => e);
    const promise2 = queue.infer(blob2, 1).catch((e) => e);
    const promise3 = queue.infer(blob3, 1);

    // First two should be skipped/rejected, third should succeed
    const result1 = await promise1;
    const result2 = await promise2;
    const result3 = await promise3;

    // First request may have started processing and been aborted, or may have been skipped
    // before even starting. Both are valid debounce behaviors.
    expect(result1).toBeInstanceOf(Error);
    expect(["Skipped", "Aborted"]).toContain((result1 as Error).message);
    expect(result2).toBeInstanceOf(Error);
    expect((result2 as Error).message).toBe("Skipped");
    expect(result3.candidates[0]).toContain("result");

    await queue.dispose();
  });

  test("aborts current inference when new request arrives", async () => {
    let abortedCount = 0;
    const slowProcessor: InferenceProcessor = vi.fn(
      async (req: InferenceRequest, signal: AbortSignal) => {
        // Wait for a while or until aborted
        await new Promise((resolve) => {
          const timeout = setTimeout(resolve, 500);
          signal.addEventListener("abort", () => {
            clearTimeout(timeout);
            resolve(undefined);
          });
        });

        if (signal.aborted) {
          abortedCount++;
          req.reject(new Error("Aborted"));
          return;
        }

        const result: InferenceResult = {
          latex: "completed",
          candidates: ["completed"],
          debugImage: "",
        };
        req.resolve(result);
      }
    );

    const queue = new InferenceQueue(slowProcessor);
    const blob1 = new Blob(["test1"], { type: "image/png" });
    const blob2 = new Blob(["test2"], { type: "image/png" });

    // Start first request
    const promise1 = queue.infer(blob1, 1).catch((e) => e);

    // Wait a bit for the first request to start processing
    await new Promise((resolve) => setTimeout(resolve, 20));

    // Send second request which should abort the first
    const promise2 = queue.infer(blob2, 1);

    const result1 = await promise1;
    const result2 = await promise2;

    expect(result1).toBeInstanceOf(Error);
    expect((result1 as Error).message).toBe("Aborted");
    expect(abortedCount).toBe(1);
    expect(result2.candidates[0]).toBe("completed");

    await queue.dispose();
  });

  test("getIsInferring returns correct state", async () => {
    let resolveProcessor: (() => void) | null = null;
    const controlledProcessor: InferenceProcessor = vi.fn(
      async (req: InferenceRequest, signal: AbortSignal) => {
        await new Promise<void>((resolve) => {
          resolveProcessor = resolve;
        });
        if (signal.aborted) {
          req.reject(new Error("Aborted"));
          return;
        }
        const result: InferenceResult = {
          latex: "done",
          candidates: ["done"],
          debugImage: "",
        };
        req.resolve(result);
      }
    );

    const queue = new InferenceQueue(controlledProcessor);
    const blob = new Blob(["test"], { type: "image/png" });

    expect(queue.getIsInferring()).toBe(false);

    const promise = queue.infer(blob, 1);

    // Wait for the processor to start
    await new Promise((resolve) => setTimeout(resolve, 10));
    expect(queue.getIsInferring()).toBe(true);

    // Complete the processor
    resolveProcessor!();
    await promise;

    // Wait a bit for cleanup
    await new Promise((resolve) => setTimeout(resolve, 10));
    expect(queue.getIsInferring()).toBe(false);

    await queue.dispose();
  });

  test("dispose aborts pending requests and current inference", async () => {
    const controlledProcessor: InferenceProcessor = vi.fn(
      async (req: InferenceRequest, signal: AbortSignal) => {
        await new Promise<void>((resolve) => {
          signal.addEventListener("abort", () => resolve());
        });
        if (signal.aborted) {
          req.reject(new Error("Aborted"));
          return;
        }
        const result: InferenceResult = {
          latex: "done",
          candidates: ["done"],
          debugImage: "",
        };
        req.resolve(result);
      }
    );

    const queue = new InferenceQueue(controlledProcessor);
    const blob = new Blob(["test"], { type: "image/png" });

    const promise = queue.infer(blob, 1).catch((e) => e);

    // Wait for processing to start
    await new Promise((resolve) => setTimeout(resolve, 10));

    // Dispose should abort
    await queue.dispose();

    const result = await promise;
    expect(result).toBeInstanceOf(Error);
    expect((result as Error).message).toBe("Aborted");
  });

  test("passes numCandidates to processor", async () => {
    let receivedCandidates = 0;
    const trackingProcessor: InferenceProcessor = vi.fn(
      async (req: InferenceRequest, signal: AbortSignal) => {
        receivedCandidates = req.numCandidates;
        const result: InferenceResult = {
          latex: "test",
          candidates: ["test"],
          debugImage: "",
        };
        req.resolve(result);
      }
    );

    const queue = new InferenceQueue(trackingProcessor);
    const blob = new Blob(["test"], { type: "image/png" });

    await queue.infer(blob, 5);

    expect(receivedCandidates).toBe(5);

    await queue.dispose();
  });
});

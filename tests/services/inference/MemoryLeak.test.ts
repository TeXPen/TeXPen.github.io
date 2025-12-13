import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { InferenceService } from '../../../services/inference/InferenceService';

describe('InferenceService Memory Leak / Race Condition', () => {
  let inferenceService: InferenceService;

  beforeEach(() => {
    // Reset instance for isolation
    // We access the private singleton via any cast or just get it if public
    // But since it's a singleton, we might need to reset it.
    // The code has `getOrCreateInstance`.
    // We can assume `getInstance` returns the same one.
    // We might need to manually dispose it first.
    inferenceService = InferenceService.getInstance();

    // Clear any previous state
    inferenceService.disposeSync();
    vi.clearAllMocks();
  });

  afterEach(async () => {
    await inferenceService.dispose();
  });

  it('should not load model if disposed immediately after init call (queued init race)', async () => {
    // 1. Mock the ModelLoader to be slow, so we can control the timing
    const { modelLoader } = await import('../../../services/inference/ModelLoader');

    let resolveLoad: (value: any) => void;
    const loadPromise = new Promise((resolve) => {
      resolveLoad = resolve;
    });

    vi.spyOn(modelLoader, 'loadModelWithFallback').mockImplementation(async () => {
      await loadPromise;
      return {
        model: {
          dispose: vi.fn(),
          generate: vi.fn(),
        } as any,
        device: 'cpu'
      };
    });

    vi.spyOn(modelLoader, 'preDownloadModels').mockResolvedValue(undefined);

    // 2. Call init (this will get queued and start waiting on our slow load)
    const initPromise = inferenceService.init(undefined, { device: 'cpu' as any });

    // 3. IMMEDIATELY call dispose.
    // This increments the generation BEFORE the actual "loadModelWithFallback" finishes
    // (or even starts, depending on mutex timing, but definitely before it returns).
    await inferenceService.dispose();

    // 4. Now release the lock (finish the load)
    resolveLoad!({});

    // 5. Wait for init to "finish" (it should handle the silent rejection/return)
    await initPromise;

    // 6. Assertions
    // The model should NOT be set on the service
    // @ts-ignore - accessing private field
    expect(inferenceService.model).toBeNull();
    // @ts-ignore - accessing private field
    expect(inferenceService.tokenizer).toBeNull();

  });
});

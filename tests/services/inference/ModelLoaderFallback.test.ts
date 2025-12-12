
import { describe, it, expect, vi, beforeEach } from "vitest";
import { ModelLoader } from "../../../services/inference/ModelLoader";
import { MODEL_CONFIG } from "../../../services/inference/config";

const { mockAutoModel, mockDownloadManager } = vi.hoisted(() => {
  return {
    mockAutoModel: { from_pretrained: vi.fn() },
    mockDownloadManager: { downloadFile: vi.fn(), checkCacheIntegrity: vi.fn() },
  };
});

// Mock the modules
vi.mock("@huggingface/transformers", () => ({
  AutoModelForVision2Seq: mockAutoModel,
}));

vi.mock("../../../services/inference/downloader/DownloadManager", () => ({
  downloadManager: mockDownloadManager,
}));

describe("ModelLoader Fallback Logic", () => {
  let modelLoader: ModelLoader;

  beforeEach(() => {
    // Reset singleton if possible, or just get instance
    modelLoader = ModelLoader.getInstance();
    mockAutoModel.from_pretrained.mockReset();
    mockDownloadManager.downloadFile.mockReset();
  });

  it("should fallback to WASM when WebGPU throws a known memory error", async () => {
    // Setup initial partial failure for WebGPU
    mockAutoModel.from_pretrained
      .mockRejectedValueOnce(new Error("pre-allocated shape... too large for the implementation")) // fail first
      .mockResolvedValueOnce({ config: { device: "wasm" } }); // succeed second (WASM)

    const result = await modelLoader.loadModelWithFallback(
      "test-model",
      MODEL_CONFIG.PROVIDERS.WEBGPU,
      "fp32"
    );

    expect(result.device).toBe(MODEL_CONFIG.FALLBACK.PROVIDER);
    expect(mockAutoModel.from_pretrained).toHaveBeenCalledTimes(2);
  });

  it("should fallback to WASM when WebGPU throws a generic Session error", async () => {
    // This is the new case we want to support
    mockAutoModel.from_pretrained
      .mockRejectedValueOnce(new Error("Failed to create the session. Error: Validation Error")) // fail first
      .mockResolvedValueOnce({ config: { device: "wasm" } }); // succeed second

    const result = await modelLoader.loadModelWithFallback(
      "test-model",
      MODEL_CONFIG.PROVIDERS.WEBGPU,
      "fp32"
    );

    expect(result.device).toBe(MODEL_CONFIG.FALLBACK.PROVIDER);
    expect(mockAutoModel.from_pretrained).toHaveBeenCalledTimes(2);
  });

  it("should NOT fallback for unrelated errors", async () => {
    mockAutoModel.from_pretrained.mockRejectedValue(new Error("Network connection lost"));

    // Unlike bun:test, vitest expects might need to handle async rejections slightly differently if it's not a promise
    // But loadModelWithFallback returns a promise, so expect(...).rejects.toThrow() is correct
    await expect(
      modelLoader.loadModelWithFallback(
        "test-model",
        MODEL_CONFIG.PROVIDERS.WEBGPU,
        "fp32"
      )
    ).rejects.toThrow("Network connection lost");

    expect(mockAutoModel.from_pretrained).toHaveBeenCalledTimes(1);
  });
});


import { AutoModelForVision2Seq, PreTrainedModel } from '@huggingface/transformers';
import { VisionEncoderDecoderModel } from './types';
import { getSessionOptions } from './config';

export class ModelLoader {
  private static instance: ModelLoader;

  private constructor() { }

  public static getInstance(): ModelLoader {
    if (!ModelLoader.instance) {
      ModelLoader.instance = new ModelLoader();
    }
    return ModelLoader.instance;
  }

  public async preDownloadModels(modelId: string, sessionOptions: any, onProgress?: (status: string, progress?: number) => void): Promise<void> {
    const { downloadManager } = await import('../downloader/DownloadManager');
    const commonFiles = [
      `onnx/${sessionOptions.encoder_model_file_name}`,
      `onnx/${sessionOptions.decoder_model_file_name}`,
    ];

    // State for tracking progress across multiple files
    const progressState: Record<string, { loaded: number, total: number }> = {};

    // Initialize state
    commonFiles.forEach(f => {
      progressState[f] = { loaded: 0, total: 0 };
    });

    const toMB = (bytes: number) => (bytes / 1024 / 1024).toFixed(2);

    const updateProgress = () => {
      if (!onProgress) return;

      let totalLoaded = 0;
      let totalSize = 0;
      const parts: string[] = [];

      Object.entries(progressState).forEach(([f, s]) => {
        totalLoaded += s.loaded;
        totalSize += s.total;

        // Identify file type
        const name = f.includes('encoder') ? 'Enc' : (f.includes('decoder') ? 'Dec' : 'File');
        const loadedMB = toMB(s.loaded);
        const totalMB = toMB(s.total);

        if (s.total > 0) {
          const pct = Math.round((s.loaded / s.total) * 100);
          parts.push(`${name}: ${loadedMB}/${totalMB} MB (${pct}%)`);
        } else {
          parts.push(`${name}: ${loadedMB} MB`);
        }
      });

      if (totalSize === 0) return;

      // Calculate total percentage for the progress bar width
      const totalPercentage = Math.round((totalLoaded / totalSize) * 100);

      // Pass the detailed string as the status text
      onProgress(`${parts.join(' | ')}`, totalPercentage);
    };

    if (onProgress) onProgress(`Loading models...`, 0);

    // Download sequentially to avoid OOM
    const downloadPromises = commonFiles.map(async (file) => {
      const fileUrl = `https://huggingface.co/${modelId}/resolve/main/${file}`;
      try {
        await downloadManager.downloadFile(fileUrl, (p) => {
          progressState[file] = { loaded: p.loaded, total: p.total };
          updateProgress();
        });
      } catch (e) {
        console.warn(`[ModelLoader] Pre-download skipped for ${file}:`, e);
      }
    });

    await Promise.all(downloadPromises);
  }

  public async loadModelWithFallback(
    modelId: string,
    initialDevice: string,
    initialDtype: string,
    onProgress?: (status: string, progress?: number) => void
  ): Promise<{ model: VisionEncoderDecoderModel, device: string, dtype: string }> {
    let device = initialDevice;
    let dtype = initialDtype;
    let sessionOptions = getSessionOptions(device, dtype);

    try {
      const model = await AutoModelForVision2Seq.from_pretrained(modelId, sessionOptions as any) as VisionEncoderDecoderModel;
      return { model, device, dtype };
    } catch (loadError: any) {
      // Check if this is a WebGPU buffer size / memory error OR generic unsupported device error (common in Node env)
      const isWebGPUMemoryError = loadError?.message?.includes('createBuffer') ||
        loadError?.message?.includes('mappedAtCreation') ||
        loadError?.message?.includes('too large for the implementation') ||
        loadError?.message?.includes('GPUDevice');

      const isUnsupportedDeviceError = loadError?.message?.includes('Unsupported device');

      if ((isWebGPUMemoryError || isUnsupportedDeviceError) && device === 'webgpu') {
        if (isWebGPUMemoryError) {
          console.warn('[ModelLoader] WebGPU buffer allocation failed, falling back to WASM...');
          if (onProgress) onProgress('WebGPU memory limit hit. Switching to WASM...');
        } else {
          console.warn('[ModelLoader] WebGPU not supported in this environment, falling back to WASM...');
          if (onProgress) onProgress('WebGPU unavailable. Switching to WASM...');
        }

        // Retry with WASM
        device = 'wasm';
        dtype = 'q8';
        sessionOptions = getSessionOptions(device, dtype);

        // Explicitly download the WASM model files so the user sees progress
        await this.preDownloadModels(modelId, sessionOptions, onProgress);

        const model = await AutoModelForVision2Seq.from_pretrained(modelId, sessionOptions as any) as VisionEncoderDecoderModel;
        return { model, device, dtype };
      } else {
        throw loadError;
      }
    }
  }
}

export const modelLoader = ModelLoader.getInstance();

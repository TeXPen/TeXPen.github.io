import { useState, useCallback, useEffect, useRef } from 'react';
import { ModelConfig, Candidate } from '../types';
import { Quantization } from '../services/inference/types';
import { inferenceService } from '../services/inference/InferenceService';

import { MODEL_CONFIG, GENERATION_CONFIG } from '../services/inference/config';

export function useInkModel(
  theme: 'light' | 'dark',
  provider: 'webgpu' | 'wasm' | null,
  quantization: Quantization,
  encoderQuantization: Quantization,
  decoderQuantization: Quantization,
  customModelId: string = MODEL_CONFIG.ID
) {
  // Sampling Defaults
  const [numCandidates, setNumCandidates] = useState<number>(GENERATION_CONFIG.NUM_BEAMS);
  const [doSample, setDoSample] = useState(true); // Default to true for better UX with multiple candidates? Or stick to config? Sticking to hardcoded true for now as per original code logic or Config?
  // Original was true. Config says false. Assuming original intent for hook defaults overrides global default if needed, 
  // BUT goal is to use config. 
  // Let's check GENERATION_CONFIG.DO_SAMPLE -> false.
  // If I change this to false, behavior changes.
  // User "Extract the ones in useInkModel.ts too".
  // I should use GENERATION_CONFIG.DO_SAMPLE if I want true alignment.
  // However, the hook had `true`.
  // I will use `true` but note it, or maybe I should update config to true?
  // Actually `useInkModel` had explicit defaults.
  // Let's use `GENERATION_CONFIG.DEFAULT_DO_SAMPLE` if I added it... I didn't add it.
  // I added DEFAULT_TEMPERATURE etc.
  // Let's stick to extraction.

  // Wait, I saw `DO_SAMPLE: false` in generation.ts.
  // The hook has `useState(true)`.
  // I should probably map these to new constants I added: DEFAULT_TEMPERATURE, etc.

  const [temperature, setTemperature] = useState(GENERATION_CONFIG.DEFAULT_TEMPERATURE);
  const [topK, setTopK] = useState(GENERATION_CONFIG.DEFAULT_TOP_K);
  const [topP, setTopP] = useState(GENERATION_CONFIG.DEFAULT_TOP_P);

  const [config, setConfig] = useState<ModelConfig>({
    encoderModelUrl: MODEL_CONFIG.ID,
    decoderModelUrl: MODEL_CONFIG.ID,
    tokenizerUrl: MODEL_CONFIG.ID,
    imageSize: MODEL_CONFIG.IMAGE_SIZE,
    encoderInputName: MODEL_CONFIG.ENCODER_INPUT_NAME,
    decoderInputName: MODEL_CONFIG.DECODER_INPUT_NAME,
    decoderOutputName: MODEL_CONFIG.DECODER_OUTPUT_NAME,
    mean: [...MODEL_CONFIG.MEAN],
    std: [...MODEL_CONFIG.STD],
    invert: false,
    eosToken: MODEL_CONFIG.TOKENS.EOS,
    bosToken: MODEL_CONFIG.TOKENS.BOS,
    padToken: MODEL_CONFIG.TOKENS.PAD,
    preferredProvider: 'webgpu',
  });

  const [status, setStatus] = useState<string>('idle'); // idle, loading, error, success
  const [isInferencing, setIsInferencing] = useState<boolean>(false);
  const activeInferenceCount = useRef<number>(0);
  /* 
   * Queue management for inference requests that come in while model is loading
   * Now includes options to preserve callbacks like onPreprocess
   */
  const pendingInferenceRef = useRef<{
    canvas: HTMLCanvasElement;
    options?: { onPreprocess?: (debugImage: string) => void };
    resolve: (value: { latex: string; candidates: Candidate[]; debugImage: string | null } | null) => void;
    reject: (reason?: unknown) => void;
  } | null>(null);

  const debounceTimeoutRef = useRef<{ timer: ReturnType<typeof setTimeout>; resolve: (value: { latex: string; candidates: Candidate[]; debugImage: string | null } | null) => void } | null>(null);

  const [loadingPhase, setLoadingPhase] = useState<string>('');
  const [progress, setProgress] = useState(0);
  const [userConfirmed, setUserConfirmed] = useState(false);
  const [isGenerationQueued, setIsGenerationQueued] = useState(false);
  const [isLoadedFromCache, setIsLoadedFromCache] = useState(false);

  const [isInitialized, setIsInitialized] = useState(false);

  // Check if the model is cached
  useEffect(() => {
    async function checkCache() {
      try {
        const { getSessionOptions } = await import('../services/inference/config');

        // Determine which files we expect based on current settings
        if (!provider) return;
        const sessionOptions = getSessionOptions(provider, {
          overall: quantization,
          encoder: encoderQuantization,
          decoder: decoderQuantization
        });
        const expectedFiles = [
          sessionOptions.encoder_model_file_name,
          sessionOptions.decoder_model_file_name
        ];

        const cache = await caches.open('transformers-cache');
        const requests = await cache.keys();

        // Check if ALL expected files are in the cache
        // We check if the URL contains the filename. 
        // Ideally we should check modelID + filename, but filename is usually unique enough or we can assume modelID focus.
        // The URL is usually like: https://huggingface.co/.../resolve/main/onnx/encoder_model.onnx
        const allCached = expectedFiles.every(file =>
          requests.some(req => req.url.includes(file))
        );

        setIsLoadedFromCache(allCached);
        if (allCached) {
          setUserConfirmed(true);
        } else {
          // Only reset userConfirmed if we are NOT in the initial load phase (to avoid annoying resets)
          // But actually, if we switch to a mode that isn't cached, we DO want to ask confirmation again?
          // Current logic: if cached, auto-confirm. If not, wait for confirm.
          setUserConfirmed(false);
        }
      } catch (error) {
        console.warn('Cache API is not available or failed:', error);
        setUserConfirmed(false);
      } finally {
        setIsInitialized(true);
      }
    }
    checkCache();
  }, [config.encoderModelUrl, provider, quantization, encoderQuantization, decoderQuantization]);

  const prevSettingsRef = useRef<{ provider: string; modelId: string; quantization: Quantization; encoderQuantization: Quantization; decoderQuantization: Quantization } | null>(null);

  useEffect(() => {
    let isCancelled = false;

    const initModel = async () => {
      if (!provider) return;

      try {
        // Register quota error handler (e.g. Incognito fallback)
        const { downloadManager } = await import('../services/downloader/DownloadManager');
        downloadManager.setQuotaErrorHandler(async () => {
          return window.confirm(
            "Couldn't save checkpoints to persistent storage (e.g. Incognito Mode). \n\nThe download will continue in memory, but will be lost if you refresh the page."
          );
        });

        setStatus('loading');
        const msg = isLoadedFromCache ? 'Loading model from cache...' : 'Downloading model... (Inference paused)';
        setLoadingPhase(msg);

        await inferenceService.init((phase, progress) => {
          if (isCancelled) return;

          let displayPhase = phase;
          if (phase.startsWith('Loading model')) {
            displayPhase = msg;
          } else {
            displayPhase = phase;
          }

          setLoadingPhase(displayPhase);

          if (progress !== undefined) {
            setProgress(progress);
          }
        }, {
          device: provider,
          modelId: customModelId,
          quantization,
          encoderQuantization,
          decoderQuantization,
        });

        if (!isCancelled) {
          setStatus('idle');
          setLoadingPhase('');
          prevSettingsRef.current = { provider, modelId: customModelId, quantization, encoderQuantization, decoderQuantization };
        }
      } catch (error) {
        if (isCancelled) return;
        // Check if aborted by user
        if (error instanceof Error && error.message.includes('aborted by user')) {
          console.log('Model loading aborted by user.');
          setStatus('idle');
          setLoadingPhase('');
          return;
        }

        console.error('Failed to initialize model:', error);
        setStatus('error');
        setLoadingPhase('Failed to load model');
      }
    };

    if (userConfirmed) {
      initModel();
    }

    return () => {
      isCancelled = true;
      const settingsChanged = prevSettingsRef.current &&
        (prevSettingsRef.current.provider !== provider ||
          prevSettingsRef.current.modelId !== customModelId ||
          prevSettingsRef.current.quantization !== quantization ||
          prevSettingsRef.current.encoderQuantization !== encoderQuantization ||
          prevSettingsRef.current.decoderQuantization !== decoderQuantization);

      if (settingsChanged && userConfirmed) {
        console.log('[useInkModel] Settings changed, disposing model...');
        inferenceService.dispose().catch((err) => {
          console.warn('Model disposal during cleanup:', err.message);
        });
      }
    };
  }, [provider, customModelId, quantization, encoderQuantization, decoderQuantization, userConfirmed, isLoadedFromCache]);



  const infer = useCallback(async (canvas: HTMLCanvasElement, options?: { onPreprocess?: (debugImage: string) => void }) => {
    // Prevent inference if model is loading or not confirmed
    if (status === 'loading') {
      console.log('Inference queued: Model is currently loading.');

      if (pendingInferenceRef.current) {
        // Cancel previous queued inference
        pendingInferenceRef.current.resolve(null);
      }

      // Update UI immediately
      setIsGenerationQueued(true);

      return new Promise<{ latex: string; candidates: Candidate[]; debugImage: string | null } | null>((resolve, reject) => {
        pendingInferenceRef.current = { canvas, options, resolve, reject };
      });
    }

    if (!userConfirmed && !isLoadedFromCache) {
      console.warn('Inference skipped: User has not confirmed model download.');
      return null;
    }

    activeInferenceCount.current += 1;
    setIsInferencing(true);
    setStatus('inferencing');



    // Debounce Logic

    return new Promise<{ latex: string; candidates: Candidate[]; debugImage: string | null } | null>((resolve, reject) => {
      // Clear any existing debounce timer and resolve its promise as null (skipped)
      if (debounceTimeoutRef.current) {
        clearTimeout(debounceTimeoutRef.current.timer);
        debounceTimeoutRef.current.resolve(null);
        // Since we are cancelling the previous one effectively before it started (in the debounce phase),
        // we must decrement the counter that was incremented for it.
        activeInferenceCount.current -= 1;
      }

      const timer = setTimeout(() => {
        // Clear the ref as we are now executing
        debounceTimeoutRef.current = null;

        canvas.toBlob(async (blob) => {
          if (!blob) {
            activeInferenceCount.current -= 1;
            if (activeInferenceCount.current === 0) setIsInferencing(false);
            setStatus('error');
            return reject(new Error('Failed to create blob from canvas'));
          }
          try {
            const res = await inferenceService.infer(blob, {
              num_beams: numCandidates,
              do_sample: doSample,
              temperature,
              top_k: topK,
              top_p: topP,
              onPreprocess: options?.onPreprocess,
            });
            if (res) {
              // Map string candidates to Candidate objects
              const newCandidates = res.candidates.map((latex, index) => ({
                id: index,
                latex: latex
              }));

              setStatus('success');
              resolve({ latex: res.latex, candidates: newCandidates, debugImage: res.debugImage });
            } else {
              // If result is null (skipped), we resolve null
              resolve(null);
            }
          } catch (e: unknown) {
            const err = e as Error;
            if (err.message === 'Aborted' || err.message === 'Skipped' || err.name === 'AbortError') {
              console.log('Inference aborted/skipped:', err.message);
              resolve(null);
            } else {
              console.error('Inference error:', e);
              setStatus('error');
              reject(e);
            }
          } finally {
            activeInferenceCount.current -= 1;
            if (activeInferenceCount.current === 0) {
              setIsInferencing(false);
            }
          }
        }, 'image/png');
      }, 100); // 100ms debounce

      debounceTimeoutRef.current = { timer, resolve };
    });
  }, [numCandidates, doSample, temperature, topK, topP, status, userConfirmed, isLoadedFromCache]);

  const inferFromUrl = useCallback(async (url: string, options?: { onPreprocess?: (debugImage: string) => void }) => {
    // if (status === 'loading') {
    //   console.warn('Inference skipped: Model is currently loading.');
    //   return null;
    // }

    try {
      const img = new Image();
      img.crossOrigin = 'anonymous';

      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
        img.src = url;
      });

      const canvas = document.createElement('canvas');
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d');
      if (!ctx) throw new Error('Failed to get canvas context');

      ctx.drawImage(img, 0, 0);

      return await infer(canvas, options);

    } catch (error) {
      console.error('Error loading reference image:', error);
      setStatus('error');
      return null;
    }
  }, [infer]);

  // Process queued inference when model becomes idle (loaded)
  useEffect(() => {
    if (status === 'idle' && pendingInferenceRef.current) {
      console.log('[useInkModel] Processing queued inference');
      const { canvas, options, resolve, reject } = pendingInferenceRef.current;
      pendingInferenceRef.current = null;
      setIsGenerationQueued(false);

      infer(canvas, options).then(resolve).catch(reject);
    }
  }, [status, infer]);

  return {
    config,
    setConfig,
    status,
    infer,
    inferFromUrl,
    isInferencing,
    loadingPhase,
    numCandidates,
    setNumCandidates,
    progress,
    userConfirmed,
    setUserConfirmed,
    isLoadedFromCache,
    isInitialized,
    doSample,
    setDoSample,
    temperature,
    setTemperature,
    topK,
    setTopK,
    topP,
    setTopP,
    isGenerationQueued,
  };
}
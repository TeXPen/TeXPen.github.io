import { useState, useCallback, useEffect, useRef } from 'react';
import { ModelConfig, Candidate, ParagraphInferenceResult } from '../types';
import { inferenceService } from '../services/inference/InferenceService';

import { MODEL_CONFIG, GENERATION_CONFIG } from '../services/inference/config';

export function useInkModel(theme: 'light' | 'dark', provider: 'webgpu' | 'wasm' | null, customModelId: string = MODEL_CONFIG.ID, initOptions?: { skipLatex?: boolean }) {
  // Sampling Defaults
  const [numCandidates, setNumCandidates] = useState<number>(GENERATION_CONFIG.NUM_BEAMS);
  const [doSample, setDoSample] = useState(true);

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
        const sessionOptions = getSessionOptions(provider);
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
  }, [config.encoderModelUrl, provider]);

  const prevSettingsRef = useRef<{ provider: string; modelId: string } | null>(null);

  useEffect(() => {
    let isCancelled = false;

    const initModel = async () => {
      if (!provider) return;

      try {
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
        }, { device: provider, modelId: customModelId, ...initOptions });

        if (!isCancelled) {
          setStatus('idle');
          setLoadingPhase('');
          prevSettingsRef.current = { provider, modelId: customModelId };
        }
      } catch (error) {
        if (isCancelled) return;
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
          prevSettingsRef.current.modelId !== customModelId);

      if (settingsChanged && userConfirmed) {
        console.log('[useInkModel] Settings changed, disposing model...');
        inferenceService.dispose().catch((err) => {
          console.warn('Model disposal during cleanup:', err.message);
        });
      }
    };
  }, [provider, customModelId, userConfirmed, isLoadedFromCache]);



  const infer = useCallback(async (canvas: HTMLCanvasElement, options?: { onPreprocess?: (debugImage: string) => void }) => {
    if (status === 'loading') {
      console.log('Inference queued: Model is currently loading.');

      if (pendingInferenceRef.current) {
        pendingInferenceRef.current.resolve(null);
      }

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

    return new Promise<{ latex: string; candidates: Candidate[]; debugImage: string | null } | null>((resolve, reject) => {
      if (debounceTimeoutRef.current) {
        clearTimeout(debounceTimeoutRef.current.timer);
        debounceTimeoutRef.current.resolve(null);
        activeInferenceCount.current -= 1;
      }

      const timer = setTimeout(() => {
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
              const newCandidates = res.candidates.map((latex, index) => ({
                id: index,
                latex: latex
              }));

              setStatus('success');
              resolve({ latex: res.latex, candidates: newCandidates, debugImage: res.debugImage });
            } else {
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
      }, 100);

      debounceTimeoutRef.current = { timer, resolve };
    });
  }, [numCandidates, doSample, temperature, topK, topP, status, userConfirmed, isLoadedFromCache]);

  const inferFromUrl = useCallback(async (url: string, options?: { onPreprocess?: (debugImage: string) => void }) => {
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

  const inferParagraph = useCallback(async (canvas: HTMLCanvasElement, options?: { onPreprocess?: (debugImage: string) => void }) => {
    if (status === 'loading') {
      console.log('Inference queued: Model is currently loading.');
      return null;
    }

    if (!userConfirmed && !isLoadedFromCache) {
      console.warn('Inference skipped: User has not confirmed model download.');
      return null;
    }

    setIsInferencing(true);
    setStatus('inferencing');

    return new Promise<ParagraphInferenceResult | null>((resolve, reject) => {
      canvas.toBlob(async (blob) => {
        if (!blob) {
          setIsInferencing(false);
          setStatus('error');
          return reject(new Error('Failed to create blob'));
        }
        try {
          const res = await inferenceService.inferParagraph(blob, {
            onPreprocess: options?.onPreprocess
          });
          setStatus('success');
          resolve(res);
        } catch (e: unknown) {
          const err = e as Error;
          if (err.message === 'Aborted' || err.message === 'Skipped' || err.name === 'AbortError') {
            console.log('Paragraph inference aborted/skipped:', err.message);
            resolve(null);
          } else {
            console.error('Paragraph Inference error:', e);
            setStatus('error');
            reject(e);
          }
        } finally {
          setIsInferencing(false);
        }
      });
    });
  }, [status, userConfirmed, isLoadedFromCache]);

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
    inferParagraph,
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
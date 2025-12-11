import { useState, useCallback, useEffect, useRef } from 'react';
import { ModelConfig, Candidate } from '../types';
import { inferenceService } from '../services/inference/InferenceService';

import { INFERENCE_CONFIG } from '../services/inference/config';

export function useInkModel(theme: 'light' | 'dark', quantization: string = INFERENCE_CONFIG.DEFAULT_QUANTIZATION, provider: 'webgpu' | 'wasm', customModelId: string = INFERENCE_CONFIG.MODEL_ID) {
  const [numCandidates, setNumCandidates] = useState<number>(1);
  const [config, setConfig] = useState<ModelConfig>({
    encoderModelUrl: 'Ji-Ha/TexTeller3-ONNX-dynamic',
    decoderModelUrl: 'Ji-Ha/TexTeller3-ONNX-dynamic',
    tokenizerUrl: 'Ji-Ha/TexTeller3-ONNX-dynamic',
    imageSize: 448,
    encoderInputName: 'pixel_values',
    decoderInputName: 'decoder_input_ids',
    decoderOutputName: 'logits',
    mean: [0.9545467],
    std: [0.15394445],
    invert: false,
    eosToken: '</s>',
    bosToken: '<s>',
    padToken: '<pad>',
    preferredProvider: 'webgpu',
  });

  const [status, setStatus] = useState<string>('idle'); // idle, loading, error, success
  const [isInferencing, setIsInferencing] = useState<boolean>(false);
  const activeInferenceCount = useRef<number>(0);
  const [loadingPhase, setLoadingPhase] = useState<string>('');
  const [progress, setProgress] = useState(0);
  const [userConfirmed, setUserConfirmed] = useState(false);
  const [isLoadedFromCache, setIsLoadedFromCache] = useState(false);

  const [isInitialized, setIsInitialized] = useState(false);

  // Check if the model is cached
  useEffect(() => {
    async function checkCache() {
      try {
        const cache = await caches.open('transformers-cache');
        const requests = await cache.keys();
        const isCached = requests.some(req => req.url.includes(config.encoderModelUrl));
        setIsLoadedFromCache(isCached);
        if (isCached) {
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
  }, [config.encoderModelUrl]);

  const prevSettingsRef = useRef<{ quantization: string; provider: string; modelId: string } | null>(null);

  useEffect(() => {
    let isCancelled = false;

    const initModel = async () => {
      try {
        setStatus('loading');
        const msg = isLoadedFromCache ? 'Loading model from cache...' : 'Downloading model... (Inference paused)';
        setLoadingPhase(msg);

        await inferenceService.init((phase, progress) => {
          if (isCancelled) return;
          if (phase.startsWith('Loading model')) {
            setLoadingPhase(msg);
          } else {
            setLoadingPhase(phase);
          }
          if (progress !== undefined) {
            setProgress(progress);
          }
        }, { dtype: quantization, device: provider, modelId: customModelId });

        if (!isCancelled) {
          setStatus('idle');
          setLoadingPhase('');
          prevSettingsRef.current = { quantization, provider, modelId: customModelId };
        }
      } catch (error) {
        if (isCancelled) return;
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
        (prevSettingsRef.current.quantization !== quantization ||
          prevSettingsRef.current.provider !== provider ||
          prevSettingsRef.current.modelId !== customModelId);

      if (settingsChanged && userConfirmed) {
        console.log('[useInkModel] Settings changed, disposing model...');
        inferenceService.dispose().catch((err) => {
          console.warn('Model disposal during cleanup:', err.message);
        });
      }
    };
  }, [quantization, provider, customModelId, userConfirmed, isLoadedFromCache]);

  const infer = useCallback(async (canvas: HTMLCanvasElement) => {
    // Prevent inference if model is loading or not confirmed
    // if (status === 'loading') {
    //   console.warn('Inference skipped: Model is currently loading.');
    //   return null;
    // }
    if (!userConfirmed && !isLoadedFromCache) {
      console.warn('Inference skipped: User has not confirmed model download.');
      return null;
    }

    activeInferenceCount.current += 1;
    setIsInferencing(true);
    setStatus('inferencing');

    return new Promise<{ latex: string; candidates: Candidate[]; debugImage: string | null } | null>((resolve, reject) => {
      canvas.toBlob(async (blob) => {
        if (!blob) {
          activeInferenceCount.current -= 1;
          if (activeInferenceCount.current === 0) {
            setIsInferencing(false);
          }
          setStatus('error');
          return reject(new Error('Failed to create blob from canvas'));
        }
        try {
          const res = await inferenceService.infer(blob, numCandidates);
          if (res) {
            // Map string candidates to Candidate objects
            const newCandidates = res.candidates.map((latex, index) => ({
              id: index,
              latex: latex
            }));

            setStatus('success');
            resolve({ latex: res.latex, candidates: newCandidates, debugImage: res.debugImage });
          } else {
            setStatus('idle');
            resolve(null);
          }
        } catch (e: any) {
          if (e.message === 'Aborted' || e.message === 'Skipped' || e.name === 'AbortError') {
            console.log('Inference aborted/skipped:', e.message);
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
    });
  }, [numCandidates, status, userConfirmed, isLoadedFromCache]);

  const inferFromUrl = useCallback(async (url: string) => {
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

      return await infer(canvas);

    } catch (error) {
      console.error('Error loading reference image:', error);
      setStatus('error');
      return null;
    }
  }, [infer, status]);

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
  };
}
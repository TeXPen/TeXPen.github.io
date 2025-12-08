import { useState, useCallback, useEffect, useRef } from 'react';
import { ModelConfig, Candidate } from '../types';
import { inferenceService } from '../services/inference/InferenceService';

import { INFERENCE_CONFIG } from '../services/inference/config';

export function useInkModel(theme: 'light' | 'dark', quantization: string = INFERENCE_CONFIG.DEFAULT_QUANTIZATION, provider: 'webgpu' | 'wasm' | 'webgl', customModelId: string = INFERENCE_CONFIG.MODEL_ID) {
  const [numCandidates, setNumCandidates] = useState<number>(1);
  const [config, setConfig] = useState<ModelConfig>({
    encoderModelUrl: 'onnx-community/TexTeller3-ONNX',
    decoderModelUrl: 'onnx-community/TexTeller3-ONNX',
    tokenizerUrl: 'onnx-community/TexTeller3-ONNX',
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

  const [latex, setLatex] = useState<string>('');
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [status, setStatus] = useState<string>('idle'); // idle, loading, error, success
  const [isInferencing, setIsInferencing] = useState<boolean>(false);
  // Counter to track active inference requests - prevents race condition when one is aborted while another starts
  const activeInferenceCount = useRef<number>(0);
  const [loadingPhase, setLoadingPhase] = useState<string>('');
  const [debugImage, setDebugImage] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [userConfirmed, setUserConfirmed] = useState(false);
  const [isLoadedFromCache, setIsLoadedFromCache] = useState(false);

  // Check if the model is cached
  useEffect(() => {
    async function checkCache() {
      try {
        const cache = await caches.open('transformers-cache');
        const requests = await cache.keys();
        const isCached = requests.some(req => req.url.includes(config.encoderModelUrl));
        setIsLoadedFromCache(isCached);
        // If it's cached, we auto-load (userConfirmed = true)
        // If NOT cached, we wait for user to confirm (userConfirmed = false)
        if (isCached) {
          setUserConfirmed(true);
        } else {
          setUserConfirmed(false);
        }
      } catch (error) {
        console.warn('Cache API is not available or failed:', error);
        // Fallback: assume not cached, ask user
        setUserConfirmed(false);
      }
    }
    checkCache();
  }, [config.encoderModelUrl]);

  // Track previous settings to detect actual changes vs StrictMode re-runs
  const prevSettingsRef = useRef<{ quantization: string; provider: string; modelId: string } | null>(null);

  // Initialize model on mount, dispose on unmount or settings change
  useEffect(() => {
    let isCancelled = false;

    const initModel = async () => {
      try {
        setStatus('loading');
        // Better message based on cache status
        const msg = isLoadedFromCache ? 'Loading model from cache...' : 'Downloading model... (this may take a while)';
        setLoadingPhase(msg);

        await inferenceService.init((phase, progress) => {
          if (isCancelled) return; // Don't update state if cancelled

          // If the service sends a generic 'Loading model...' message, override it with our more specific one
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
          // Track that we successfully loaded with these settings
          prevSettingsRef.current = { quantization, provider, modelId: customModelId };
        }
      } catch (error) {
        if (isCancelled) return; // Ignore errors if cancelled
        console.error('Failed to initialize model:', error);
        setStatus('error');
        setLoadingPhase('Failed to load model');
      }
    };

    if (userConfirmed) {
      initModel();
    }

    // Cleanup: only dispose if settings actually changed (not just StrictMode re-run)
    return () => {
      isCancelled = true;
      // Check if settings actually changed - if same settings, DON'T dispose
      // The InferenceService singleton will reuse the existing model
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

  // Note: beforeunload cleanup is now handled directly in InferenceService.ts

  const infer = useCallback(async (canvas: HTMLCanvasElement) => {
    // Increment counter and set inferencing state
    activeInferenceCount.current += 1;
    setIsInferencing(true);
    setStatus('inferencing'); // Use different status to avoid showing full overlay

    return new Promise<{ latex: string; candidates: Candidate[] } | null>((resolve, reject) => {
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
            setLatex(res.latex);
            setDebugImage(res.debugImage);

            // Map string candidates to Candidate objects
            const newCandidates = res.candidates.map((latex, index) => ({
              id: index,
              latex: latex
            }));

            setCandidates(newCandidates);
            setStatus('success');
            resolve({ latex: res.latex, candidates: newCandidates });
          } else {
            setStatus('idle');
            resolve(null);
          }
        } catch (e: any) {
          if (e.message === 'Aborted' || e.message === 'Skipped' || e.name === 'AbortError') {
            console.log('Inference aborted/skipped:', e.message);
            // Don't update status or latex - another inference is likely pending
            // Just resolve with null to indicate no result from this attempt
            resolve(null);
          } else {
            console.error('Inference error:', e);
            setStatus('error');
            reject(e);
          }
        } finally {
          // Decrement counter, only set isInferencing to false if no more active inferences
          activeInferenceCount.current -= 1;
          if (activeInferenceCount.current === 0) {
            setIsInferencing(false);
          }
        }
      }, 'image/png');
    });
  }, [numCandidates]);

  const inferFromUrl = useCallback(async (url: string) => {
    try {
      // Load image from URL
      const img = new Image();
      img.crossOrigin = 'anonymous'; // Handle CORS

      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
        img.src = url;
      });

      // Create a canvas and draw the image
      const canvas = document.createElement('canvas');
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d');
      if (!ctx) throw new Error('Failed to get canvas context');

      ctx.drawImage(img, 0, 0);

      // Run inference
      return await infer(canvas);

    } catch (error) {
      console.error('Error loading reference image:', error);
      setStatus('error');
      return null;
    }
  }, [infer]);

  const clear = useCallback(() => {
    setLatex('');
    setCandidates([]);
    setDebugImage(null);
    setStatus('idle');
  }, []);

  return {
    config,
    setConfig,
    status,
    latex,
    setLatex,
    candidates,
    infer,
    inferFromUrl,
    clear,
    isInferencing,
    loadingPhase,
    debugImage,
    numCandidates,
    setNumCandidates,
    progress,
    userConfirmed,
    setUserConfirmed,
    isLoadedFromCache,
  };
}
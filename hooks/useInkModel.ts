import { useState, useCallback, useEffect } from 'react';
import { ModelConfig, Candidate } from '../types';
import { inferenceService } from '../services/inference/InferenceService';

import { INFERENCE_CONFIG } from '../services/inference/config';

export function useInkModel(theme: 'light' | 'dark', quantization: string = INFERENCE_CONFIG.DEFAULT_QUANTIZATION, provider: 'webgpu' | 'wasm' | 'webgl') {
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
        }, { dtype: quantization, device: provider });

        if (!isCancelled) {
          setStatus('idle');
          setLoadingPhase('');
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

    // Cleanup: dispose model when settings change or component unmounts
    return () => {
      isCancelled = true;
      // Only dispose if we were confirmed (model was potentially loaded)
      if (userConfirmed) {
        inferenceService.dispose().catch((err) => {
          // Ignore disposal errors (e.g., if inference is in progress)
          console.warn('Model disposal during cleanup:', err.message);
        });
      }
    };
  }, [quantization, provider, userConfirmed, isLoadedFromCache]);

  // Cleanup on page unload (refresh/close)
  useEffect(() => {
    const handleBeforeUnload = () => {
      // Attempt to dispose model before page unloads
      // Note: async operations may not complete, but we try
      inferenceService.dispose().catch(() => { });
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
    };
  }, []);

  const infer = useCallback(async (canvas: HTMLCanvasElement) => {
    setIsInferencing(true);
    setStatus('inferencing'); // Use different status to avoid showing full overlay

    return new Promise<{ latex: string; candidates: Candidate[] } | null>((resolve, reject) => {
      canvas.toBlob(async (blob) => {
        if (!blob) {
          setIsInferencing(false);
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
            // Do not set status to error, just silently fail
            // We might want to set status back to idle if it was inferencing?
            // Usually another inference immediately takes over, setting it to inferencing.
            // If we set it to idle it might flash. kept as is or conditional?
            // If this was the last one, maybe idle? But hard to know.
            // Safest is to just NOT set error.
          } else {
            console.error('Inference error:', e);
            setStatus('error');
            reject(e);
          }
        } finally {
          setIsInferencing(false);
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
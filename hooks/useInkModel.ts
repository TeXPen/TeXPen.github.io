import { useState, useCallback, useEffect } from 'react';
import { ModelConfig, Candidate } from '../types';
import { inferenceService } from '../services/inference/InferenceService';

export function useInkModel(theme: 'light' | 'dark', quantization: string, provider: 'webgpu' | 'wasm' | 'webgl') {
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
        // If it's cached, we don't need user confirmation to start downloading
        if (isCached) {
          setUserConfirmed(true);
        }
      } catch (error) {
        console.warn('Cache API is not available or failed:', error);
      }
    }
    checkCache();
  }, [config.encoderModelUrl]);

  // Initialize model on mount
  useEffect(() => {
    const initModel = async () => {
      try {
        setStatus('loading');
        setLoadingPhase('Initializing model...');
        await inferenceService.init((phase, progress) => {
          setLoadingPhase(phase);
          if (progress !== undefined) {
            setProgress(progress);
          }
        }, { dtype: quantization, device: provider });
        setStatus('idle');
        setLoadingPhase('');
      } catch (error) {
        console.error('Failed to initialize model:', error);
        setStatus('error');
        setLoadingPhase('Failed to load model');
      }
    };
    if (userConfirmed) {
      initModel();
    }
  }, [quantization, provider, userConfirmed]);

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
        } catch (e) {
          console.error('Inference error:', e);
          setStatus('error');
          reject(e);
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
      await infer(canvas);

    } catch (error) {
      console.error('Error loading reference image:', error);
      setStatus('error');
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
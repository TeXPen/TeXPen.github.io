import { useState, useCallback, useEffect } from 'react';
import { ModelConfig, Candidate } from '../types';
import { inferenceService } from '../services/inferenceService';

export function useInkModel(theme: 'light' | 'dark') {
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
    preferredProvider: 'webgpu'
  });

  const [latex, setLatex] = useState<string>('');
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [status, setStatus] = useState<string>('idle'); // idle, loading, error, success
  const [isInferencing, setIsInferencing] = useState<boolean>(false);
  const [loadingPhase, setLoadingPhase] = useState<string>('');

  // Initialize model on mount
  useEffect(() => {
    const initModel = async () => {
      try {
        setStatus('loading');
        setLoadingPhase('Initializing model...');
        await inferenceService.init((phase) => setLoadingPhase(phase));
        setStatus('idle');
        setLoadingPhase('');
      } catch (error) {
        console.error('Failed to initialize model:', error);
        setStatus('error');
        setLoadingPhase('Failed to load model');
      }
    };
    initModel();
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
          const res = await inferenceService.infer(blob);
          if (res) {
            setLatex(res);
            setCandidates([{ id: 0, latex: res }]);
            setStatus('success');
            resolve({ latex: res, candidates: [{ id: 0, latex: res }] });
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
  }, []);

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
    loadingPhase
  };
}

import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { ModelConfig, ModelStatus, Candidate } from '../types';
import { DEFAULT_CONFIG, initModel, runInference, generateVariations, clearModelCache } from '../services/onnxService';
import { areModelsCached } from '../services/cacheService';

export const useInkModel = (theme: 'dark' | 'light') => {
  const [config, setConfig] = useState<ModelConfig>(DEFAULT_CONFIG);
  const [status, setStatus] = useState<ModelStatus>('loading');
  const [latex, setLatex] = useState<string>('');
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [progress, setProgress] = useState<number>(0);
  const [loadingPhase, setLoadingPhase] = useState<string>('Initializing');
  const [isLoadedFromCache, setIsLoadedFromCache] = useState<boolean>(false);
  
  const [userConfirmed, setUserConfirmedState] = useState<boolean>(
    () => localStorage.getItem('userConfirmed') === 'true'
  );

  const isInitializing = useRef(false);
  const isInitialized = useRef(false);

  // Check cache status
  useEffect(() => {
    const checkCache = async () => {
      try {
        const cached = await areModelsCached([
          DEFAULT_CONFIG.encoderModelUrl,
          DEFAULT_CONFIG.decoderModelUrl,
        ]);
        
        setIsLoadedFromCache(cached);

        // If models are cached, we can auto-confirm if not already confirmed
        if (cached && !userConfirmed) {
          setUserConfirmedState(true);
          localStorage.setItem('userConfirmed', 'true');
        }
      } catch (e) {
        console.warn("Cache check failed:", e);
      }
    };
    
    checkCache();
    // We only need to run this on mount or if userConfirmed changes 
    // (though logically it mostly matters on mount)
  }, []);

  const setUserConfirmed = (value: boolean) => {
    setUserConfirmedState(value);
    if (value) localStorage.setItem('userConfirmed', 'true');
  };

  // Initialize Model
  useEffect(() => {
    if (!userConfirmed) return;
    if (isInitializing.current || isInitialized.current) return;

    isInitializing.current = true;

    const load = async () => {
      try {
        await initModel(config, (phase, pct) => {
          setLoadingPhase(phase);
          setProgress(pct);
        });
        setStatus('ready');
        isInitialized.current = true;
      } catch (e) {
        console.error("Initialization Error:", e);
        setStatus('error');
      } finally {
        isInitializing.current = false;
      }
    };

    load();
  }, [config, userConfirmed]);

  const offscreenCanvas = useRef<HTMLCanvasElement | null>(null);
  const offscreenCtx = useRef<CanvasRenderingContext2D | null>(null);

  useEffect(() => {
    const canvas = document.createElement('canvas');
    canvas.width = config.imageSize;
    canvas.height = config.imageSize;
    offscreenCanvas.current = canvas;
    offscreenCtx.current = canvas.getContext('2d', { willReadFrequently: true });
  }, [config.imageSize]);

  // Force invert to FALSE. We handle pixel normalization manually in infer().
  const runConfig = useMemo(() => ({ ...config, invert: false }), [config]);

  const infer = useCallback(async (sourceCanvas: HTMLCanvasElement) => {
    if ((status !== 'ready' && status !== 'inferencing') || !offscreenCtx.current) return null;

    setStatus('inferencing');
    try {
      const ctx = offscreenCtx.current;
      const size = config.imageSize;

      // 1. Clear the offscreen canvas (Transparent)
      ctx.clearRect(0, 0, size, size);
      
      // 2. Draw the user's raw strokes (scaled)
      // Note: We do NOT fill with white first. We want to preserve transparency 
      // so we can distinguish background from ink.
      ctx.drawImage(sourceCanvas, 0, 0, size, size);
      
      // 3. Get raw pixel data
      const imageData = ctx.getImageData(0, 0, size, size);
      const data = imageData.data;

      // 4. PIXEL PIPELINE: Normalize to "Black Ink on White Background"
      // This fixes the "Black on Black" issue in Dark Mode.
      for (let i = 0; i < data.length; i += 4) {
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];
        const a = data[i + 3];

        // Determine if pixel is effectively background
        const isTransparent = a < 50;
        const avg = (r + g + b) / 3;

        if (theme === 'dark') {
           // DARK MODE: Ink is White (High value), Background is Transparent or Dark.
           if (isTransparent) {
             // Transparent -> Force White Background
             data[i] = 255; data[i+1] = 255; data[i+2] = 255; data[i+3] = 255;
           } else if (avg > 100) {
             // Bright Pixel (Ink) -> Force BLACK
             data[i] = 0; data[i+1] = 0; data[i+2] = 0; data[i+3] = 255;
           } else {
             // Dark Pixel (Canvas artifacts) -> Force WHITE
             data[i] = 255; data[i+1] = 255; data[i+2] = 255; data[i+3] = 255;
           }
        } else {
           // LIGHT MODE: Ink is Black (Low value), Background is Transparent or White.
           if (isTransparent) {
             // Transparent -> Force White Background
             data[i] = 255; data[i+1] = 255; data[i+2] = 255; data[i+3] = 255;
           } else if (avg < 150) {
             // Dark Pixel (Ink) -> Force BLACK
             data[i] = 0; data[i+1] = 0; data[i+2] = 0; data[i+3] = 255;
           } else {
             // Bright Pixel (Background) -> Force WHITE
             data[i] = 255; data[i+1] = 255; data[i+2] = 255; data[i+3] = 255;
           }
        }
      }
      
      // 5. Put the normalized image back
      ctx.putImageData(imageData, 0, 0);

      // 6. Run Inference
      const resultLatex = await runInference(imageData, runConfig);
      
      const vars = generateVariations(resultLatex);
      const newCandidates = vars.map((l, i) => ({ id: i, latex: l }));
      setCandidates(newCandidates);
      setLatex(vars[0]);
      setStatus('ready');

      return { latex: vars[0], candidates: newCandidates };
    } catch (e) {
      console.error("Inference Error:", e);
      setStatus('error');
      return null;
    }
  }, [status, theme, config.imageSize, runConfig]);

  const clear = useCallback(() => {
    setLatex('');
    setCandidates([]);
  }, []);

  return {
    config,
    setConfig,
    status,
    latex,
    setLatex,
    candidates,
    setCandidates,
    infer,
    clear,
    progress,
    loadingPhase,
    userConfirmed,
    setUserConfirmed,
    resetCache: clearModelCache,
    isLoadedFromCache // Export the cache status
  };
};
import { useRef, useCallback, useState } from 'react';

const MAX_HISTORY = 50;

export const useCanvasHistory = () => {
  const historyRef = useRef<ImageData[]>([]);
  const historyIndexRef = useRef(-1);
  const [canUndo, setCanUndo] = useState(false);
  const [canRedo, setCanRedo] = useState(false);

  const updateState = useCallback(() => {
    setCanUndo(historyIndexRef.current > 0);
    setCanRedo(historyIndexRef.current < historyRef.current.length - 1);
  }, []);

  const saveSnapshot = useCallback((canvas: HTMLCanvasElement) => {
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;

    // Get raw image data
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    // Remove any redo states
    historyRef.current = historyRef.current.slice(0, historyIndexRef.current + 1);

    // Add new state
    historyRef.current.push(imageData);

    // Limit history size
    if (historyRef.current.length > MAX_HISTORY) {
      historyRef.current.shift();
    } else {
      historyIndexRef.current++;
    }

    updateState();
  }, [updateState]);

  const undo = useCallback((canvas: HTMLCanvasElement) => {
    if (historyIndexRef.current <= 0) return;

    historyIndexRef.current--;
    const imageData = historyRef.current[historyIndexRef.current];

    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx || !imageData) return;

    ctx.putImageData(imageData, 0, 0);
    updateState();
  }, [updateState]);

  const redo = useCallback((canvas: HTMLCanvasElement) => {
    if (historyIndexRef.current >= historyRef.current.length - 1) return;

    historyIndexRef.current++;
    const imageData = historyRef.current[historyIndexRef.current];

    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx || !imageData) return;

    ctx.putImageData(imageData, 0, 0);
    updateState();
  }, [updateState]);

  const clear = useCallback(() => {
    historyRef.current = [];
    historyIndexRef.current = -1;
    updateState();
  }, [updateState]);

  return {
    saveSnapshot,
    undo,
    redo,
    clear,
    canUndo,
    canRedo
  };
};

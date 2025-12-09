import React, { useRef, useState, useCallback } from 'react';
import CanvasBoard from './CanvasBoard';
import CanvasToolbar from './CanvasToolbar';
import { useCanvasHistory } from '../../hooks/useCanvasHistory';
import { ToolType, Stroke } from '../../types/canvas';

interface CanvasAreaProps {
    theme: 'dark' | 'light';
    onStrokeEnd: (canvas: HTMLCanvasElement, strokes: Stroke[]) => void;
    onClear: () => void;
}

const CanvasArea: React.FC<CanvasAreaProps> = ({ theme, onStrokeEnd, onClear }) => {
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const contentCanvasRef = useRef<HTMLCanvasElement | null>(null);
    const strokesRef = useRef<Stroke[]>([]);
    const [activeTool, setActiveTool] = useState<ToolType>('pen');
    const { saveSnapshot, undo, redo, clear: clearHistory, canUndo, canRedo } = useCanvasHistory();

    const setCanvasRef = useCallback((ref: HTMLCanvasElement | null) => {
        canvasRef.current = ref;
    }, []);

    const setContentCanvasRef = useCallback((ref: HTMLCanvasElement | null) => {
        contentCanvasRef.current = ref;
    }, []);

    const handleStrokeEnd = useCallback(() => {
        if (contentCanvasRef.current) {
            saveSnapshot(contentCanvasRef.current, strokesRef.current);
            onStrokeEnd(contentCanvasRef.current, strokesRef.current);
        }
    }, [onStrokeEnd, saveSnapshot]);

    const handleClear = () => {
        const canvas = canvasRef.current;
        const contentCanvas = contentCanvasRef.current;

        if (canvas) {
            const ctx = canvas.getContext('2d', { willReadFrequently: true });
            if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
        }

        if (contentCanvas) {
            const ctx = contentCanvas.getContext('2d', { willReadFrequently: true });
            if (ctx) ctx.clearRect(0, 0, contentCanvas.width, contentCanvas.height);

            // Allow Undo of Clear: Save the empty state
            strokesRef.current = [];
            saveSnapshot(contentCanvas, []);
            onClear();
        }
    };

    const handleUndo = useCallback(() => {
        const canvas = canvasRef.current;
        const contentCanvas = contentCanvasRef.current;
        if (contentCanvas && canvas) {
            const restoredStrokes = undo(contentCanvas);
            if (restoredStrokes) {
                strokesRef.current = [...restoredStrokes];
            }

            // Copy to visible
            const ctx = canvas.getContext('2d', { willReadFrequently: true });
            if (ctx) {
                ctx.save();
                ctx.resetTransform();
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(contentCanvas, 0, 0);
                ctx.restore();
            }
        }
    }, [undo]);

    const handleRedo = useCallback(() => {
        const canvas = canvasRef.current;
        const contentCanvas = contentCanvasRef.current;
        if (contentCanvas && canvas) {
            const restoredStrokes = redo(contentCanvas);
            if (restoredStrokes) {
                strokesRef.current = [...restoredStrokes];
            }

            // Copy to visible
            const ctx = canvas.getContext('2d', { willReadFrequently: true });
            if (ctx) {
                ctx.save();
                ctx.resetTransform();
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(contentCanvas, 0, 0);
                ctx.restore();
            }
        }
    }, [redo]);

    return (
        <div className="flex-1 relative bg-transparent group cursor-crosshair overflow-hidden transition-colors duration-500">
            <CanvasToolbar
                activeTool={activeTool}
                onToolChange={setActiveTool}
                onUndo={handleUndo}
                onRedo={handleRedo}
                canUndo={canUndo}
                canRedo={canRedo}
            />

            <CanvasBoard
                theme={theme}
                activeTool={activeTool}
                strokesRef={strokesRef}
                onStrokeEnd={handleStrokeEnd}
                refCallback={setCanvasRef}
                contentRefCallback={setContentCanvasRef}
            />

            {/* Clear Button */}
            <button
                onClick={handleClear}
                className="absolute bottom-6 right-6 p-4 rounded-full bg-white/80 dark:bg-[#1a1a1a] border border-black/5 dark:border-white/10 text-slate-400 dark:text-white/40 hover:text-red-500 dark:hover:text-red-400 hover:scale-105 transition-all shadow-lg hover:shadow-xl z-20"
                title="Clear Canvas"
            >
                <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="3 6 5 6 21 6" />
                    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                    <line x1="10" y1="11" x2="10" y2="17" />
                    <line x1="14" y1="11" x2="14" y2="17" />
                </svg>
            </button>
        </div>
    );
};

export default CanvasArea;
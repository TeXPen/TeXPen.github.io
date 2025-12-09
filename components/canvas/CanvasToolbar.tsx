import React, { useState } from 'react';
import { ToolType } from '../../types/canvas';

interface CanvasToolbarProps {
    activeTool: ToolType;
    onToolChange: (tool: ToolType) => void;
    onUndo: () => void;
    onRedo: () => void;
    canUndo: boolean;
    canRedo: boolean;
}

const PenIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M17 3a2.828 2.828 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3z" /></svg>
);

const EraserIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m7 21-4.3-4.3c-1-1-1-2.5 0-3.4l9.6-9.6c1-1 2.5-1 3.4 0l5.6 5.6c1 1 1 2.5 0 3.4L13 21" /><path d="M22 21H7" /><path d="m5 11 9 9" /></svg>
);

const CircleIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10" /></svg>
);

const LineIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="5" y1="19" x2="19" y2="5" /></svg>
);

const UndoIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 7v6h6" /><path d="M21 17a9 9 0 0 0-9-9 9 9 0 0 0-6 2.3L3 13" /></svg>
);

const RedoIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 7v6h-6" /><path d="M3 17a9 9 0 0 1 9-9 9 9 0 0 1 6 2.3L21 13" /></svg>
);

const CanvasToolbar: React.FC<CanvasToolbarProps> = ({
    activeTool,
    onToolChange,
    onUndo,
    onRedo,
    canUndo,
    canRedo
}) => {
    const [showEraserMenu, setShowEraserMenu] = useState(false);
    const isEraserActive = activeTool === 'eraser-radial' || activeTool === 'eraser-line';

    const handleEraserClick = () => {
        if (isEraserActive) {
            setShowEraserMenu(!showEraserMenu);
        } else {
            onToolChange('eraser-line');
            setShowEraserMenu(true);
        }
    };

    const selectEraserType = (type: 'eraser-radial' | 'eraser-line') => {
        onToolChange(type);
        setShowEraserMenu(false);
    };

    return (
        <div className="absolute bottom-24 right-6 flex flex-col items-center gap-2 z-20">
            {/* Undo/Redo */}
            <div className="flex flex-col items-center gap-1 p-1.5 bg-white/80 dark:bg-[#1a1a1a] backdrop-blur-sm border border-black/5 dark:border-white/10 rounded-full shadow-lg">
                <button
                    onClick={onUndo}
                    disabled={!canUndo}
                    className={`p-2 rounded-full transition-all ${!canUndo
                        ? 'text-slate-300 dark:text-white/10 cursor-not-allowed'
                        : 'text-slate-500 dark:text-white/40 hover:bg-black/5 dark:hover:bg-white/5 hover:text-slate-900 dark:hover:text-white'
                        }`}
                    title="Undo"
                >
                    <UndoIcon />
                </button>
                <button
                    onClick={onRedo}
                    disabled={!canRedo}
                    className={`p-2 rounded-full transition-all ${!canRedo
                        ? 'text-slate-300 dark:text-white/10 cursor-not-allowed'
                        : 'text-slate-500 dark:text-white/40 hover:bg-black/5 dark:hover:bg-white/5 hover:text-slate-900 dark:hover:text-white'
                        }`}
                    title="Redo"
                >
                    <RedoIcon />
                </button>
            </div>

            {/* Tools */}
            <div className="relative flex flex-col items-center gap-1 p-1.5 bg-white/80 dark:bg-[#1a1a1a] backdrop-blur-sm border border-black/5 dark:border-white/10 rounded-full shadow-lg">
                <button
                    onClick={() => { onToolChange('pen'); setShowEraserMenu(false); }}
                    className={`p-2 rounded-full transition-all ${activeTool === 'pen'
                        ? 'bg-black text-white dark:bg-white dark:text-black shadow-sm'
                        : 'text-slate-500 dark:text-white/40 hover:bg-black/5 dark:hover:bg-white/5'
                        }`}
                    title="Pen"
                >
                    <PenIcon />
                </button>

                {/* Eraser with submenu */}
                <div className="relative">
                    <button
                        onClick={handleEraserClick}
                        className={`p-2 rounded-full transition-all ${isEraserActive
                            ? 'bg-black text-white dark:bg-white dark:text-black shadow-sm'
                            : 'text-slate-500 dark:text-white/40 hover:bg-black/5 dark:hover:bg-white/5'
                            }`}
                        title="Eraser"
                    >
                        <EraserIcon />
                    </button>

                    {/* Eraser Type Menu */}
                    {showEraserMenu && (
                        <div className="absolute right-full mr-2 top-1/2 -translate-y-1/2 flex items-center gap-1 p-1 bg-white/90 dark:bg-[#1a1a1a] backdrop-blur-sm border border-black/5 dark:border-white/10 rounded-full shadow-lg animate-in slide-in-from-right-2 duration-150">
                            <button
                                onClick={() => selectEraserType('eraser-radial')}
                                className={`p-2 rounded-full transition-all ${activeTool === 'eraser-radial'
                                    ? 'bg-black text-white dark:bg-white dark:text-black shadow-sm'
                                    : 'text-slate-500 dark:text-white/40 hover:bg-black/5 dark:hover:bg-white/5'
                                    }`}
                                title="Radial Eraser"
                            >
                                <CircleIcon />
                            </button>
                            <button
                                onClick={() => selectEraserType('eraser-line')}
                                className={`p-2 rounded-full transition-all ${activeTool === 'eraser-line'
                                    ? 'bg-black text-white dark:bg-white dark:text-black shadow-sm'
                                    : 'text-slate-500 dark:text-white/40 hover:bg-black/5 dark:hover:bg-white/5'
                                    }`}
                                title="Stroke Eraser"
                            >
                                <LineIcon />
                            </button>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default CanvasToolbar;

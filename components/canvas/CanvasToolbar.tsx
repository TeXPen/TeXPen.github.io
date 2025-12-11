import React, { useState } from 'react';
import { ToolType } from '../../types/canvas';
import {
    PenIcon,
    EraserIcon,
    CircleIcon,
    LineIcon,
    UndoIcon,
    RedoIcon,

    SelectIcon,
    RectSelectIcon,
    LassoSelectIcon
} from '../common/icons/ToolbarIcons';

interface CanvasToolbarProps {
    activeTool: ToolType;
    onToolChange: (tool: ToolType) => void;
    onUndo: () => void;
    onRedo: () => void;
    canUndo: boolean;
    canRedo: boolean;
}

const CanvasToolbar: React.FC<CanvasToolbarProps> = ({
    activeTool,
    onToolChange,
    onUndo,
    onRedo,
    canUndo,
    canRedo
}) => {
    const [showEraserMenu, setShowEraserMenu] = useState(false);
    const [showSelectMenu, setShowSelectMenu] = useState(false);
    const isEraserActive = activeTool === 'eraser-radial' || activeTool === 'eraser-line';
    const isSelectActive = activeTool === 'select-rect' || activeTool === 'select-lasso';

    const handleEraserClick = () => {
        if (isEraserActive) {
            setShowEraserMenu(!showEraserMenu);
            setShowSelectMenu(false);
        } else {
            onToolChange('eraser-line');
            setShowEraserMenu(true);
            setShowSelectMenu(false);
        }
    };

    const handleSelectClick = () => {
        if (isSelectActive) {
            setShowSelectMenu(!showSelectMenu);
            setShowEraserMenu(false);
        } else {
            // Default to rect select
            onToolChange('select-rect');
            setShowSelectMenu(true);
            setShowEraserMenu(false);
        }
    };

    const selectEraserType = (type: 'eraser-radial' | 'eraser-line') => {
        onToolChange(type);
        setShowEraserMenu(false);
    };

    const selectSelectType = (type: 'select-rect' | 'select-lasso') => {
        onToolChange(type);
        setShowSelectMenu(false);
    };

    return (
        <div className="absolute bottom-24 right-6 flex flex-col items-center gap-2 z-20">
            {/* Undo/Redo */}
            <div className="flex flex-col items-center gap-1 p-1.5 bg-white/80 dark:bg-[#1a1a1a] backdrop-blur-sm border border-black/5 dark:border-white/10 rounded-full shadow-lg">
                <button
                    onClick={onUndo}
                    disabled={!canUndo}
                    className={`p-3 md:p-2 rounded-full transition-all flex items-center justify-center min-w-[44px] min-h-[44px] md:min-w-0 md:min-h-0 ${!canUndo
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
                    className={`p-3 md:p-2 rounded-full transition-all flex items-center justify-center min-w-[44px] min-h-[44px] md:min-w-0 md:min-h-0 ${!canRedo
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
                <div className="relative">
                    <button
                        onClick={handleSelectClick}
                        className={`p-3 md:p-2 rounded-full transition-all flex items-center justify-center min-w-[44px] min-h-[44px] md:min-w-0 md:min-h-0 ${isSelectActive
                            ? 'bg-black text-white dark:bg-white dark:text-black shadow-sm'
                            : 'text-slate-500 dark:text-white/40 hover:bg-black/5 dark:hover:bg-white/5'
                            }`}
                        title="Select"
                    >
                        <SelectIcon />
                    </button>

                    {/* Select Type Menu */}
                    {showSelectMenu && (
                        <div className="absolute right-full mr-2 top-1/2 -translate-y-1/2 flex items-center gap-1 p-1 bg-white/90 dark:bg-[#1a1a1a] backdrop-blur-sm border border-black/5 dark:border-white/10 rounded-full shadow-lg animate-in slide-in-from-right-2 duration-150">
                            <button
                                onClick={() => selectSelectType('select-rect')}
                                className={`p-2 rounded-full transition-all ${activeTool === 'select-rect'
                                    ? 'bg-black text-white dark:bg-white dark:text-black shadow-sm'
                                    : 'text-slate-500 dark:text-white/40 hover:bg-black/5 dark:hover:bg-white/5'
                                    }`}
                                title="Rectangle Selection"
                            >
                                <RectSelectIcon />
                            </button>
                            <button
                                onClick={() => selectSelectType('select-lasso')}
                                className={`p-2 rounded-full transition-all ${activeTool === 'select-lasso'
                                    ? 'bg-black text-white dark:bg-white dark:text-black shadow-sm'
                                    : 'text-slate-500 dark:text-white/40 hover:bg-black/5 dark:hover:bg-white/5'
                                    }`}
                                title="Lasso Selection"
                            >
                                <LassoSelectIcon />
                            </button>
                        </div>
                    )}
                </div>

                <button
                    onClick={() => { onToolChange('pen'); setShowEraserMenu(false); setShowSelectMenu(false); }}
                    className={`p-3 md:p-2 rounded-full transition-all flex items-center justify-center min-w-[44px] min-h-[44px] md:min-w-0 md:min-h-0 ${activeTool === 'pen'
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
                        className={`p-3 md:p-2 rounded-full transition-all flex items-center justify-center min-w-[44px] min-h-[44px] md:min-w-0 md:min-h-0 ${isEraserActive
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

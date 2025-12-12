import React, { useState } from 'react';
import { EyeIcon } from '../common/icons/EyeIcon';
import { CloseIcon } from '../common/icons/CloseIcon';

interface PreviewModelInputProps {
    debugImage: string | null;
}

const CollapsedView: React.FC = () => (
    <div className="w-12 h-12 p-0 flex items-center justify-center cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-500 dark:text-gray-400">
        <EyeIcon />
    </div>
);

const ExpandedView: React.FC<{ debugImage: string, onClose: () => void }> = ({ debugImage, onClose }) => (
    <div className="w-64 p-3 flex flex-col gap-2">
        <div className="flex justify-between items-center border-b border-gray-200 dark:border-gray-700 pb-2">
            <span className="text-xs font-bold text-gray-500 uppercase tracking-wider">Preview Model Input</span>
            <button
                onClick={(e) => {
                    e.stopPropagation();
                    onClose();
                }}
                className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
            >
                <CloseIcon />
            </button>
        </div>
        <div className="relative aspect-square w-full bg-gray-100 dark:bg-black rounded border border-gray-200 dark:border-gray-800 overflow-hidden">
            <img
                src={debugImage}
                alt="Debug Input"
                className="w-full h-full object-contain"
                style={{ imageRendering: 'pixelated' }}
            />
        </div>
        <div className="text-[10px] text-gray-400 font-mono text-center">
            448x448 • Grayscale
        </div>
    </div>
);

const PreviewModelInput: React.FC<PreviewModelInputProps> = ({ debugImage }) => {
    const [isExpanded, setIsExpanded] = useState(false);

    if (!debugImage) return null;

    return (
        <div className="absolute bottom-4 left-4 z-50 flex flex-col items-start gap-2">
            <div
                className={`
                    bg-white/90 dark:bg-gray-900/90 backdrop-blur-md 
                    border border-gray-200 dark:border-gray-700 
                    rounded-lg shadow-xl overflow-hidden transition-all duration-300 ease-in-out
                    ${isExpanded ? '' : 'cursor-pointer'}
                `}
                title={!isExpanded ? "Preview Model Input" : undefined}
                onClick={() => !isExpanded && setIsExpanded(true)}
            >
                {isExpanded ? (
                    <ExpandedView debugImage={debugImage} onClose={() => setIsExpanded(false)} />
                ) : (
                    <CollapsedView />
                )}
            </div>
        </div>
    );
};

export default PreviewModelInput;

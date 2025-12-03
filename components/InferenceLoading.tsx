import React from 'react';

interface InferenceLoadingProps {
    isInferencing: boolean;
}

const InferenceLoading: React.FC<InferenceLoadingProps> = ({ isInferencing }) => {
    if (!isInferencing) return null;

    return (
        <div className="fixed bottom-4 left-1/2 -translate-x-1/2 z-40 pointer-events-none">
            <div className="bg-gradient-to-r from-blue-500 to-purple-600 text-white px-6 py-3 rounded-full shadow-2xl flex items-center gap-3 animate-slide-up">
                {/* Animated spinner */}
                <div className="relative w-5 h-5">
                    <div className="absolute inset-0 border-2 border-white/30 rounded-full"></div>
                    <div className="absolute inset-0 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                </div>

                {/* Text with pulse animation */}
                <span className="font-medium text-sm animate-pulse">
                    Generating LaTeX...
                </span>
            </div>
        </div>
    );
};

export default InferenceLoading;

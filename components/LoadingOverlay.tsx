import React from 'react';

interface LoadingOverlayProps {
    phase: string;
    progress: number;
    error?: string;
    onConfirm?: () => void;
    needsConfirmation?: boolean;
    isLoadedFromCache?: boolean;
}

const LoadingOverlay: React.FC<LoadingOverlayProps> = ({
    phase,
    progress,
    error,
    onConfirm,
    needsConfirmation = false,
    isLoadedFromCache = false
}) => {
    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
            <div className="bg-white dark:bg-gray-900 rounded-2xl p-8 shadow-2xl max-w-md w-full mx-4">
                <div className="text-center">
                    {needsConfirmation ? (
                        <>
                            <div className="text-6xl mb-4">📥</div>
                            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-3">
                                Download Required
                            </h2>
                            <div className="text-left bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4 mb-4">
                                <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                                    <strong>First-time setup:</strong> This app requires downloading AI models (~1.2GB) from HuggingFace.
                                </p>
                                <ul className="text-xs text-gray-600 dark:text-gray-400 space-y-1 ml-4 list-disc">
                                    <li>Models are cached in your browser</li>
                                    <li>Only downloaded once</li>
                                    <li>May take 2-5 minutes on slower connections</li>
                                </ul>
                            </div>
                            <div className="flex gap-3 justify-center">
                                <button
                                    onClick={onConfirm}
                                    className="px-6 py-3 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-colors font-medium"
                                >
                                    Download Models
                                </button>
                            </div>
                        </>
                    ) : error ? (
                        <>
                            <div className="text-red-500 text-6xl mb-4">⚠️</div>
                            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
                                Loading Failed
                            </h2>
                            <p className="text-gray-600 dark:text-gray-400 mb-4 text-sm">
                                {error}
                            </p>
                            <button
                                onClick={() => window.location.reload()}
                                className="px-6 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-colors"
                            >
                                Retry
                            </button>
                        </>
                    ) : (
                        <>
                            <div className="text-6xl mb-4 animate-pulse">🧠</div>
                            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
                                {phase}
                            </h2>
                            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3 mb-4 overflow-hidden">
                                <div
                                    className="bg-gradient-to-r from-blue-500 to-purple-500 h-3 rounded-full transition-all duration-300 ease-out"
                                    style={{ width: `${progress}% ` }}
                                />
                            </div>
                            <p className="text-gray-600 dark:text-gray-400 text-sm">
                                {progress}% complete
                            </p>
                            {phase.includes('Encoder') || phase.includes('Decoder') ? (
                                <p className="text-gray-500 dark:text-gray-500 text-xs mt-2">
                                    {isLoadedFromCache 
                                        ? "Restoring model from browser cache..." 
                                        : "Downloading large model files. This may take a few minutes..."
                                    }
                                </p>
                            ) : null}
                        </>
                    )}
                </div>
            </div>
        </div>
    );
};

export default LoadingOverlay;

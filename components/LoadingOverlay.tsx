import React from 'react';
import { useAppContext } from './contexts/AppContext';

const LoadingOverlay: React.FC = () => {
    const {
        status,
        loadingPhase,
        progress,
        userConfirmed,
        setUserConfirmed,
        isLoadedFromCache,
    } = useAppContext();

    const error = status === 'error' ? 'Failed to load models. Please check your internet connection and try again.' : undefined;
    const needsConfirmation = !userConfirmed && !isLoadedFromCache;
    const onConfirm = () => setUserConfirmed(true);

    // Only show full overlay for initial permission/confirmation or errors.
    // We NO LONGER show it for standard model loading (handled by Main.tsx toast).
    const showFullOverlay = (status === 'error') || (!userConfirmed && !isLoadedFromCache);

    if (!showFullOverlay) {
        return null;
    }

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
            <div className="bg-white dark:bg-gray-900 rounded-2xl p-8 shadow-2xl max-w-md w-full mx-4">
                <div className="text-center">
                    {error ? (
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
                    ) : needsConfirmation ? (
                        <>
                            <div className="text-6xl mb-4">⏳</div>
                            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
                                Confirm Model Download
                            </h2>
                            <p className="text-gray-600 dark:text-gray-400 mb-4 text-sm">
                                The AI model will be downloaded to your browser's cache (approx 30MB).
                                This will only happen once.
                            </p>
                            <button
                                onClick={onConfirm}
                                className="px-6 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-colors"
                            >
                                Start Download
                            </button>
                        </>
                    ) : (
                        /* Should not happen given showFullOverlay logic, but fallback */
                        null
                    )}
                </div>
            </div>
        </div>
    );
};

export default LoadingOverlay;

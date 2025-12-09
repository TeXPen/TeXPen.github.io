import React from 'react';
import { useAppContext } from '../../contexts/AppContext';

const LoadingOverlay: React.FC = () => {
    const {
        status,
        loadingPhase,
        progress,
        userConfirmed,
        setUserConfirmed,
        isLoadedFromCache,
    } = useAppContext();

    const [dismissed, setDismissed] = React.useState(false);

    const error = status === 'error' ? 'Failed to load models. Please check your internet connection and try again.' : undefined;
    const needsConfirmation = !userConfirmed && !isLoadedFromCache;
    const onConfirm = () => setUserConfirmed(true);
    const onClose = () => setDismissed(true);

    // Only show full overlay for initial permission/confirmation or errors.
    // We NO LONGER show it for standard model loading (handled by Main.tsx toast).
    // If dismissed, we also hide it (user can manually go to settings).
    const showFullOverlay = !dismissed && ((status === 'error') || (!userConfirmed && !isLoadedFromCache));

    if (!showFullOverlay) {
        return null;
    }

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm animate-in fade-in duration-200">
            <div className="bg-white dark:bg-gray-900 rounded-2xl p-8 shadow-2xl max-w-md w-full mx-4 relative transform animate-in zoom-in-95 duration-200">
                {/* Close Button for non-error state */}
                {!error && (
                    <button
                        onClick={onClose}
                        className="absolute top-4 right-4 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 transition-colors"
                        title="Close and configure manually"
                    >
                        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                    </button>
                )}

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
                            <p className="text-gray-600 dark:text-gray-400 mb-2 text-sm">
                                The AI model will be downloaded to your browser's cache (approx 30MB).
                                This will only happen once.
                            </p>
                            <p className="text-gray-500 dark:text-gray-500 mb-6 text-xs italic">
                                You can also choose your own custom models in Settings.
                            </p>
                            <div className="flex flex-col gap-3">
                                <button
                                    onClick={onConfirm}
                                    className="px-6 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-colors font-medium shadow-lg shadow-blue-500/20"
                                >
                                    Start Download
                                </button>
                                <button
                                    onClick={onClose}
                                    className="text-sm text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 transition-colors"
                                >
                                    Configure Manually
                                </button>
                            </div>
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

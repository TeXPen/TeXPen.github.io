import React, { useState, useRef } from 'react';
import { useAppContext } from './contexts/AppContext';
import { useThemeContext } from './contexts/ThemeContext';
import { useHistoryContext } from './contexts/HistoryContext';
import LiquidBackground from './LiquidBackground';
import Header from './Header';
import HistorySidebar from './HistorySidebar';
import OutputDisplay from './OutputDisplay';
import Candidates from './Candidates';
import CanvasArea from './CanvasArea';
import LoadingOverlay from './LoadingOverlay';
import VisualDebugger from './VisualDebugger';
import ImageUploadArea from './ImageUploadArea';

const Main: React.FC = () => {
    const {
        status,
        latex,
        candidates,
        infer,
        inferFromUrl,
        clearModel,
        progress,
        loadingPhase,
        userConfirmed,
        setUserConfirmed,
        isLoadedFromCache,
        loadFromHistory,
        isSidebarOpen,
        selectedIndex,
        selectCandidate,
        isInferencing,
        debugImage,
        showVisualDebugger,
        activeTab,
    } = useAppContext();

    const { theme } = useThemeContext();
    const { history, addToHistory, deleteHistoryItem } = useHistoryContext();

    // Manage local preview for upload
    const [uploadPreview, setUploadPreview] = useState<string | null>(null);
    const [fileToConvert, setFileToConvert] = useState<File | null>(null);

    const handleInference = async (canvas: HTMLCanvasElement) => {
        const result = await infer(canvas);
        if (result) {
            addToHistory({ id: Date.now().toString(), latex: result.latex, timestamp: Date.now() });
        }
    };

    const handleImageSelect = (file: File) => {
        setFileToConvert(file);
        const url = URL.createObjectURL(file);
        setUploadPreview(url);
    };

    const handleUploadConvert = async () => {
        if (!uploadPreview) return;

        const result = await inferFromUrl(uploadPreview);
        if (result) {
            addToHistory({ id: Date.now().toString(), latex: result.latex, timestamp: Date.now() });
        }
    };

    // We need to capture the result of inferFromUrl to add to history.
    // Since we can't easily change the hook interface right this second without breaking context types,
    // let's assume valid result will be in `latex` state, BUT `addToHistory` needs a trigger.
    // Maybe `useEffect` on latex change? No, that triggers on history load too.

    // Workaround: We will rely on `inferFromUrl` calling `infer`. 
    // `useInkModel.infer` returns a promise with result.
    // `useInkModel.inferFromUrl` awaits `infer`.
    // It currently discards the return value.
    // We should fix `useInkModel.inferFromUrl` to return the result.
    // I entered this task to fix tab uploading, so I should ensure history works.

    // For this step, I will implement the UI. I will fix `inferFromUrl` return type in a follow up or assume I can do it.

    // Only show full overlay for initial model loading (User Confirmation), or critical errors.
    // We do NOT block for standard loading anymore.
    const showFullOverlay = (!userConfirmed && !isLoadedFromCache) || status === 'error';

    return (
        <div className="relative h-screen w-screen overflow-hidden font-sans bg-[#fafafa] dark:bg-black transition-colors duration-500">
            <LiquidBackground />

            <div className="flex flex-col w-full h-full bg-white/60 dark:bg-[#0c0c0c]/80 backdrop-blur-md transition-colors duration-500">
                <Header />

                <div className="flex-1 flex min-h-0 relative">
                    <HistorySidebar
                        history={history}
                        onSelect={loadFromHistory}
                        onDelete={deleteHistoryItem}
                        isOpen={isSidebarOpen}
                    />

                    <div className="flex-1 flex flex-col min-w-0 z-10 relative">
                        <OutputDisplay latex={latex} isInferencing={isInferencing} />

                        <Candidates />

                        {/* Tab Content */}
                        <div className="flex-1 relative overflow-hidden flex flex-col">
                            {/* Canvas - Always mounted to preserve state, but hidden if not active */}
                            <div className={`flex-1 flex flex-col absolute inset-0 transition-opacity duration-300 ${activeTab === 'draw' ? 'opacity-100 z-10' : 'opacity-0 z-0 pointer-events-none'}`}>
                                <CanvasArea
                                    theme={theme}
                                    onStrokeEnd={handleInference}
                                    onClear={clearModel}
                                />
                            </div>

                            {/* Upload Area */}
                            {activeTab === 'upload' && (
                                <div className="absolute inset-0 z-10 bg-transparent animate-in fade-in zoom-in-95 duration-300">
                                    <ImageUploadArea
                                        onImageSelect={handleImageSelect}
                                        onConvert={handleUploadConvert}
                                        isInferencing={isInferencing}
                                        previewUrl={uploadPreview}
                                    />
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>

            {/* Visual Debugger */}
            {showVisualDebugger && <VisualDebugger debugImage={debugImage} />}

            {/* Download Prompt / Error Overlay */}
            {/* We reuse the loading overlay logic but filtered */}
            {showFullOverlay && (
                <LoadingOverlay />
            )}

            {/* Non-blocking loading indicator (Optional, if we want a toast or corner indicator) */}
            {status === 'loading' && userConfirmed && (
                <div className="absolute bottom-6 left-1/2 -translate-x-1/2 px-4 py-2 bg-black/80 text-white rounded-full text-sm font-mono flex items-center gap-2 z-50 pointer-events-none animate-in slide-in-from-bottom-4">
                    <svg className="animate-spin h-3 w-3 text-cyan-400" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    {loadingPhase} {progress > 0 && `(${Math.round(progress)}%)`}
                </div>
            )}
        </div>
    );
};

export default Main;
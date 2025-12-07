import React, { useState } from 'react';
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
        infer,
        inferFromUrl,
        clearModel,
        progress,
        loadingPhase,
        userConfirmed,
        isLoadedFromCache,
        loadFromHistory,
        isSidebarOpen,
        isInferencing,
        debugImage,
        showVisualDebugger,
        activeTab,
        setActiveTab,
        toggleSidebar,
        sessionId,
        refreshSession,
    } = useAppContext();

    const { theme } = useThemeContext();
    const { history, addToHistory, deleteHistoryItem } = useHistoryContext();

    // Store upload preview in state to persist if we want (currently not persisting between modes for simplicity, or we could)
    // To match "seamless", let's strictly switch views.
    const [showUploadResult, setShowUploadResult] = useState(false);
    const [uploadPreview, setUploadPreview] = useState<string | null>(null);

    // Reset upload state when switching tabs
    React.useEffect(() => {
        if (activeTab === 'draw') {
            setShowUploadResult(false);
            setUploadPreview(null);
        }
    }, [activeTab]);

    const handleInference = async (canvas: HTMLCanvasElement) => {
        const result = await infer(canvas);
        if (result) {
            addToHistory({ id: Date.now().toString(), latex: result.latex, timestamp: Date.now(), source: 'draw', sessionId });
        }
    };

    const handleImageSelect = (file: File) => {
        const url = URL.createObjectURL(file);
        setUploadPreview(url);
        setShowUploadResult(false);
    };

    const handleUploadConvert = async () => {
        if (!uploadPreview) return;
        const result = await inferFromUrl(uploadPreview);
        if (result) {
            addToHistory({ id: Date.now().toString(), latex: result.latex, timestamp: Date.now(), source: 'upload', sessionId });
            // Don't clear preview immediately, keep it for context or clear if preferred.
            // User flow: Convert -> Show Result.
            setShowUploadResult(true);
        }
    };

    const handleUploadAnother = () => {
        setUploadPreview(null);
        setShowUploadResult(false);
    };

    // Only show full overlay for initial model loading (User Confirmation), or critical errors.
    const showFullOverlay = (!userConfirmed && !isLoadedFromCache) || status === 'error';

    // Logic for Full Screen Upload Mode
    // We hide Output and Candidates if we are in Upload tab AND we haven't formulated a result yet (or user wants to see full screen input)
    // Actually, "Preview" state (before convert) should also be full screen.
    // So: activeTab === 'upload' && !showUploadResult
    const isFullPageUpload = activeTab === 'upload' && !showUploadResult;

    // Helper for loading overlay content
    const renderLoadingOverlay = () => (
        <div className="absolute inset-x-0 bottom-0 z-20 flex flex-col animate-in slide-in-from-bottom-5 duration-300">
            {/* Bottom bar with loading status */}
            <div className="flex-none px-6 py-4 bg-white/95 dark:bg-[#111]/95 backdrop-blur-md border-t border-black/5 dark:border-white/5 flex items-center gap-4 shadow-lg">
                <div className="relative w-5 h-5 flex-none">
                    <div className="absolute inset-0 border-2 border-cyan-500/30 rounded-full"></div>
                    <div className="absolute inset-0 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin"></div>
                </div>
                <span className="text-sm font-medium text-slate-700 dark:text-white/80 whitespace-nowrap">
                    {loadingPhase} {progress > 0 && `(${Math.round(progress)}%)`}
                </span>
            </div>
        </div>
    );

    return (
        <div className="relative h-screen w-full overflow-hidden font-sans bg-[#fafafa] dark:bg-black transition-colors duration-500 flex flex-row">
            <LiquidBackground />

            {/* Global glass background wrapper */}
            <div className="absolute inset-0 z-0 bg-white/60 dark:bg-[#0c0c0c]/80 backdrop-blur-md transition-colors duration-500 pointer-events-none" />

            {/* Main Content Area (z-10) */}
            <div className="relative z-10 flex w-full h-full">

                <div className="flex-1 flex min-h-0 relative">
                    <HistorySidebar
                        history={history}
                        onSelect={loadFromHistory}
                        onDelete={deleteHistoryItem}
                        isOpen={isSidebarOpen}
                    />

                    <div className="flex-1 flex flex-col min-w-0 relative">
                        {/* Top Settings Bar (formerly Header) */}
                        <Header />

                        {/* Conditionally Render Output and Candidates */}
                        {!isFullPageUpload && (
                            <>
                                <OutputDisplay latex={latex} isInferencing={isInferencing} />
                                <Candidates />
                            </>
                        )}

                        {/* Workspace */}
                        <div className="flex-1 relative overflow-hidden flex flex-col">

                            {/* Draw Mode */}
                            <div className={`flex-1 flex flex-col absolute inset-0 transition-opacity duration-300 ${activeTab === 'draw' ? 'opacity-100 z-10' : 'opacity-0 z-0 pointer-events-none'}`}>
                                <CanvasArea
                                    theme={theme}
                                    onStrokeEnd={handleInference}
                                    onClear={() => {
                                        if (latex && latex.trim()) {
                                            addToHistory({
                                                id: Date.now().toString(),
                                                latex: latex,
                                                timestamp: Date.now(),
                                                source: 'draw',
                                                sessionId
                                            });
                                        }
                                        clearModel();
                                        refreshSession();
                                    }}
                                />
                                {status === 'loading' && userConfirmed && renderLoadingOverlay()}
                            </div>

                            {/* Upload Mode */}
                            {activeTab === 'upload' && (
                                <div className="absolute inset-0 z-10 bg-transparent animate-in fade-in zoom-in-95 duration-200 p-4 flex flex-col overflow-hidden">
                                    <div className={`flex-1 bg-white/50 dark:bg-black/20 rounded-2xl overflow-hidden backdrop-blur-sm w-full h-full flex flex-col items-center justify-center ${showUploadResult ? 'relative' : ''}`}>

                                        {!showUploadResult ? (
                                            /* Input / Preview State (Full Screen) */
                                            <ImageUploadArea
                                                onImageSelect={handleImageSelect}
                                                onConvert={handleUploadConvert}
                                                isInferencing={isInferencing}
                                                previewUrl={uploadPreview}
                                            />
                                        ) : (
                                            /* Result State with "Upload Another" */
                                            <div className="flex flex-col items-center gap-6 animate-in fade-in slide-in-from-bottom-5 duration-500">
                                                {uploadPreview && (
                                                    <div className="relative h-48 w-full max-w-md aspect-video rounded-xl overflow-hidden shadow-lg border border-black/10 dark:border-white/10 opacity-60 hover:opacity-100 transition-opacity">
                                                        <img src={uploadPreview} alt="Original" className="w-full h-full object-contain bg-white dark:bg-black/50" />
                                                    </div>
                                                )}

                                                <button
                                                    onClick={handleUploadAnother}
                                                    className="px-8 py-3 bg-cyan-500 hover:bg-cyan-400 text-white font-bold rounded-full shadow-lg shadow-cyan-500/20 active:scale-95 transition-all flex items-center gap-2"
                                                >
                                                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                                                    </svg>
                                                    Upload Another Image
                                                </button>
                                            </div>
                                        )}
                                    </div>
                                    {status === 'loading' && userConfirmed && renderLoadingOverlay()}
                                </div>
                            )}

                        </div>
                    </div>
                </div>
            </div>

            {/* Visual Debugger */}
            {showVisualDebugger && <VisualDebugger debugImage={debugImage} />}

            {/* Download Prompt / Error Overlay */}
            {showFullOverlay && <LoadingOverlay />}


        </div>
    );
};

export default Main;
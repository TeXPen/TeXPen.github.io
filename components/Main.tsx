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
import NavRail from './NavRail';

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
    } = useAppContext();

    const { theme } = useThemeContext();
    const { history, addToHistory, deleteHistoryItem } = useHistoryContext();

    // Store upload preview in state to persist if we want (currently not persisting between modes for simplicity, or we could)
    // To match "seamless", let's strictly switch views.
    const [uploadPreview, setUploadPreview] = useState<string | null>(null);

    const handleInference = async (canvas: HTMLCanvasElement) => {
        const result = await infer(canvas);
        if (result) {
            // History add handled by global context? Context doesn't auto-add?
            // Checking original code: Main.tsx calls addToHistory
            // I need addToHistory from context
            addToHistory({ id: Date.now().toString(), latex: result.latex, timestamp: Date.now() });
        }
    };
    // Wait, I missed addToHistory in destructuring above. Adding it back.

    const handleImageSelect = (file: File) => {
        const url = URL.createObjectURL(file);
        setUploadPreview(url);
    };

    const handleUploadConvert = async () => {
        if (!uploadPreview) return;
        const result = await inferFromUrl(uploadPreview);
        if (result) {
            addToHistory({ id: Date.now().toString(), latex: result.latex, timestamp: Date.now() });
            setUploadPreview(null); // Clear preview after conversion
        }
        // Result is adding to history? I need to check how inferFromUrl works.
        // Previous code assumed we need to add manually?
        // Let's grab addToHistory.
    };

    // Only show full overlay for initial model loading (User Confirmation), or critical errors.
    const showFullOverlay = (!userConfirmed && !isLoadedFromCache) || status === 'error';

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

                {/* Navigation Rail */}
                <NavRail activeMode={activeTab as 'draw' | 'upload'} onModeChange={(mode) => setActiveTab(mode)} />

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

                        <OutputDisplay latex={latex} isInferencing={isInferencing} />

                        <Candidates />

                        {/* Workspace */}
                        <div className="flex-1 relative overflow-hidden flex flex-col">

                            {/* Draw Mode */}
                            <div className={`flex-1 flex flex-col absolute inset-0 transition-opacity duration-300 ${activeTab === 'draw' ? 'opacity-100 z-10' : 'opacity-0 z-0 pointer-events-none'}`}>
                                <CanvasArea
                                    theme={theme}
                                    onStrokeEnd={handleInference}
                                    onClear={clearModel}
                                />
                                {status === 'loading' && userConfirmed && renderLoadingOverlay()}
                            </div>

                            {activeTab === 'upload' && (
                                <div className="absolute inset-0 z-10 bg-transparent animate-in fade-in zoom-in-95 duration-200 p-4 flex flex-col overflow-hidden">
                                    <div className="flex-1 bg-white/50 dark:bg-black/20 rounded-2xl overflow-hidden backdrop-blur-sm w-full h-full">
                                        <ImageUploadArea
                                            onImageSelect={handleImageSelect}
                                            onConvert={handleUploadConvert}
                                            isInferencing={isInferencing}
                                            previewUrl={uploadPreview}
                                        />
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
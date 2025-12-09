import React, { useState, useRef, useCallback } from 'react';
import { useAppContext } from '../../contexts/AppContext';
import { useThemeContext } from '../../contexts/ThemeContext';
import { useHistoryContext } from '../../contexts/HistoryContext';
import { Stroke } from '../../types/canvas';
import LiquidBackground from '../common/LiquidBackground';
import Header from './Header';
import HistorySidebar from './HistorySidebar';
import OutputDisplay from '../display/OutputDisplay';
import Candidates from '../display/Candidates';
import CanvasArea from '../canvas/CanvasArea';
import LoadingOverlay from '../common/LoadingOverlay';
import VisualDebugger from '../debug/VisualDebugger';
import ImageUploadArea from '../upload/ImageUploadArea';
import { MobileBottomNav } from './MobileBottomNav';


const Main: React.FC = () => {
    // Local state for the download prompt
    const [isPromptDismissed, setIsPromptDismissed] = useState(false);

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
        uploadPreview,
        showUploadResult,
        setUploadPreview,
        setShowUploadResult,
        activeInferenceTab
    } = useAppContext();

    const { theme } = useThemeContext();
    const { history, addToHistory, deleteHistoryItem, clearHistory } = useHistoryContext();

    const uploadFileInputRef = useRef<HTMLInputElement>(null);

    // Derived inference states
    const isDrawInferencing = isInferencing && activeInferenceTab === 'draw';
    const isUploadInferencing = isInferencing && activeInferenceTab === 'upload';

    // Handle file selection from the hidden input
    const handleFileInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file && file.type.startsWith('image/')) {
            const url = URL.createObjectURL(file);
            setUploadPreview(url);
            // Go back to upload page so user can start inference
            setShowUploadResult(false);
        }
        // Reset input so same file can be selected again
        e.target.value = '';
    }, [setUploadPreview, setShowUploadResult]);

    // Handle paste in upload mode (works in both preview and result views)
    React.useEffect(() => {
        if (activeTab !== 'upload') return;

        const handlePaste = (e: ClipboardEvent) => {
            if (e.clipboardData && e.clipboardData.files.length > 0) {
                const file = e.clipboardData.files[0];
                if (file.type.startsWith('image/')) {
                    e.preventDefault();
                    const url = URL.createObjectURL(file);
                    setUploadPreview(url);
                    // Go back to upload page so user can start inference
                    setShowUploadResult(false);
                }
            }
        };

        window.addEventListener('paste', handlePaste);
        return () => window.removeEventListener('paste', handlePaste);
    }, [activeTab, setUploadPreview, setShowUploadResult]);

    const handleInference = async (canvas: HTMLCanvasElement, strokes: Stroke[]) => {
        // Enforce confirmation
        if (!userConfirmed && !isLoadedFromCache) {
            setIsPromptDismissed(false); // Re-show prompt
            return; // Cancel implicit inference
        }

        const result = await infer(canvas);
        if (result) {
            addToHistory({
                id: Date.now().toString(),
                latex: result.latex,
                timestamp: Date.now(),
                source: 'draw',
                sessionId,
                strokes
            });
        }
    };

    const handleImageSelect = (file: File) => {
        const url = URL.createObjectURL(file);
        setUploadPreview(url);
        setShowUploadResult(false);
    };

    // Convert Blob URL to Base64
    const blobUrlToBase64 = async (url: string): Promise<string> => {
        const response = await fetch(url);
        const blob = await response.blob();
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onloadend = () => resolve(reader.result as string);
            reader.onerror = reject;
            reader.readAsDataURL(blob);
        });
    };

    const handleUploadConvert = async () => {
        if (!uploadPreview) return;

        // Enforce confirmation
        if (!userConfirmed && !isLoadedFromCache) {
            setIsPromptDismissed(false); // Re-show prompt
            return; // Cancel implicit inference
        }

        const result = await inferFromUrl(uploadPreview);
        if (result) {
            // Save image data to history
            const base64Image = await blobUrlToBase64(uploadPreview);
            addToHistory({
                id: Date.now().toString(),
                latex: result.latex,
                timestamp: Date.now(),
                source: 'upload',
                sessionId,
                imageData: base64Image
            });
            // Don't clear preview immediately, keep it for context or clear if preferred.
            // User flow: Convert -> Show Result.
            setShowUploadResult(true);
        }
    };

    const handleHistorySelect = (item: any) => {
        if (item.source === 'upload' && item.imageData) {
            setUploadPreview(item.imageData);
            setShowUploadResult(true);
            setActiveTab('upload');
        }
        loadFromHistory(item);
    };

    const handleUploadAnother = () => {
        // Directly open file picker instead of going back to upload page
        uploadFileInputRef.current?.click();
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
        <div className="relative h-[100dvh] w-full overflow-hidden font-sans bg-[#fafafa] dark:bg-black transition-colors duration-500 flex flex-row">
            <LiquidBackground />

            {/* Hidden file input for "Upload Another" functionality */}
            <input
                type="file"
                ref={uploadFileInputRef}
                className="hidden"
                accept="image/*"
                onChange={handleFileInputChange}
            />

            {/* Global glass background wrapper */}
            <div className="absolute inset-0 z-0 bg-white/60 dark:bg-[#0c0c0c]/80 backdrop-blur-md transition-colors duration-500 pointer-events-none" />

            {/* Main Content Area (z-10) */}
            <div className="relative z-10 flex w-full h-full">

                <div className="flex-1 flex min-h-0 relative">
                    <HistorySidebar
                        history={history}
                        onSelect={handleHistorySelect}
                        onDelete={deleteHistoryItem}
                        onClearAll={clearHistory}
                        isOpen={isSidebarOpen}
                    />

                    <div className="flex-1 flex flex-col min-w-0 relative">
                        {/* Top Settings Bar (formerly Header) */}
                        <Header />

                        {/* Conditionally Render Output and Candidates (Standard View for Draw Mode) */}
                        {activeTab === 'draw' && (
                            <div className="flex-none h-1/4 md:h-2/5 flex flex-col w-full relative z-10 shrink-0">
                                <OutputDisplay
                                    latex={latex}
                                    isInferencing={isDrawInferencing}
                                    className="flex-1 w-full"
                                />
                                <Candidates />
                            </div>
                        )}

                        {/* Workspace */}
                        <div className="flex-1 relative overflow-hidden flex flex-col">

                            {/* Draw Mode */}
                            <div className={`flex-1 flex flex-col absolute inset-0 ${activeTab === 'draw' ? 'z-10' : 'z-0 pointer-events-none hidden'}`}>
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
                                <div className="absolute inset-0 z-10 bg-transparent p-4 flex flex-col overflow-hidden">
                                    <div className={`flex-1 bg-white/50 dark:bg-black/20 rounded-2xl overflow-hidden backdrop-blur-sm w-full h-full flex flex-col ${showUploadResult ? 'relative' : 'items-center justify-center'}`}>

                                        {!showUploadResult ? (
                                            /* Input / Preview State (Full Screen center) */
                                            <ImageUploadArea
                                                onImageSelect={handleImageSelect}
                                                onConvert={handleUploadConvert}
                                                isInferencing={isUploadInferencing}
                                                previewUrl={uploadPreview}
                                            />
                                        ) : (
                                            /* Result Split View */
                                            <div className="flex-1 flex w-full h-full animate-in fade-in duration-500 flex-col divide-y divide-black/5 dark:divide-white/5">

                                                {/* Top Panel: Result & Controls (Fixed Height 50%) */}
                                                <div className="h-1/2 flex flex-col relative bg-white/40 dark:bg-transparent min-h-0">
                                                    {/* Custom styled OutputDisplay - Flex grow to fill available space */}
                                                    <OutputDisplay
                                                        latex={latex}
                                                        isInferencing={isUploadInferencing}
                                                        className="flex-1 w-full"
                                                    />
                                                    <Candidates />


                                                </div>

                                                {/* Bottom Panel: Source Image (Rest of height) */}
                                                <div
                                                    onClick={handleUploadAnother}
                                                    className="flex-1 relative bg-black/5 dark:bg-white/5 flex items-center justify-center p-4 min-h-0 cursor-pointer group hover:bg-black/10 dark:hover:bg-white/10 transition-colors"
                                                >
                                                    {uploadPreview && (
                                                        <img
                                                            src={uploadPreview}
                                                            alt="Original"
                                                            className="max-w-full max-h-full object-contain shadow-lg rounded-lg transition-transform duration-300 group-hover:scale-[1.02]"
                                                        />
                                                    )}
                                                    <div className="absolute top-4 left-4 inline-flex items-center px-2 py-1 bg-black/20 backdrop-blur-sm rounded text-xs text-white/50">
                                                        Original Image
                                                    </div>

                                                    {/* Floating Upload Button */}
                                                    <button
                                                        onClick={(e) => {
                                                            e.stopPropagation();
                                                            handleUploadAnother();
                                                        }}
                                                        className="absolute bottom-6 left-1/2 -translate-x-1/2 px-8 py-3 bg-cyan-500/10 dark:bg-cyan-500/10 hover:bg-cyan-500/20 dark:hover:bg-cyan-500/20 text-cyan-600 dark:text-cyan-400 font-bold rounded-full border border-cyan-500/50 dark:border-cyan-400/50 hover:border-cyan-500 dark:hover:border-cyan-400 hover:shadow-lg hover:shadow-cyan-500/20 active:scale-95 transition-all flex items-center gap-2 backdrop-blur-md z-10"
                                                    >
                                                        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                                                        </svg>
                                                        Upload Another Image
                                                    </button>
                                                </div>

                                            </div>
                                        )}
                                    </div>
                                    {status === 'loading' && userConfirmed && renderLoadingOverlay()}
                                </div>
                            )}

                        </div>

                        {/* Mobile Bottom Navigation */}
                        <MobileBottomNav />
                    </div>
                </div>
            </div>

            {/* Visual Debugger */}
            {showVisualDebugger && <VisualDebugger debugImage={debugImage} />}

            {/* Download Prompt / Error Overlay */}
            {showFullOverlay && (
                <LoadingOverlay
                    isDismissed={isPromptDismissed}
                    onDismiss={() => setIsPromptDismissed(true)}
                />
            )}


        </div>
    );
};

export default Main;
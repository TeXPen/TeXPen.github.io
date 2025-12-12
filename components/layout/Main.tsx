import React, { useState, useRef, useCallback } from 'react';
import { useAppContext } from '../../contexts/AppContext';

import { useHistoryContext } from '../../contexts/HistoryContext';
import { Stroke } from '../../types/canvas';
import { HistoryItem } from '../../types';
import LiquidBackground from '../common/LiquidBackground';
import Header from './Header';
import HistorySidebar from './HistorySidebar';
import LoadingOverlay from '../common/LoadingOverlay';
import { ProgressToast } from '../common/ProgressToast';
import PreviewModelInput from '../debug/PreviewModelInput';
import DrawTab from './DrawTab';
import UploadTab from './UploadTab';
import { MobileBottomNav } from './MobileBottomNav';
import { ConfirmationDialog } from '../common/ConfirmationDialog';

const Main: React.FC = () => {
    const [isPromptDismissed, setIsPromptDismissed] = useState(false);

    const {
        status,
        infer,
        inferFromUrl,
        progress,
        loadingPhase,
        isGenerationQueued,
        userConfirmed,
        isLoadedFromCache,
        loadFromHistory,
        isSidebarOpen,
        debugImage,
        showPreviewInput,
        activeTab,
        setActiveTab,
        sessionId,
        uploadPreview,
        setUploadPreview,
        setShowUploadResult,
        customNotification,
        dialogConfig,
        closeDialog
    } = useAppContext();

    // Theme not currently used in Main directly, passed down to components or used in CSS
    // const { theme } = useThemeContext();
    const { history, addToHistory, deleteHistoryItem, clearHistory } = useHistoryContext();

    const uploadFileInputRef = useRef<HTMLInputElement>(null);

    // Handle file selection from the hidden input
    const handleFileInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file && file.type.startsWith('image/')) {
            const url = URL.createObjectURL(file);
            setUploadPreview(url);
            setShowUploadResult(false);
        }
        e.target.value = '';
    }, [setUploadPreview, setShowUploadResult]);

    // Handle paste in upload mode
    React.useEffect(() => {
        if (activeTab !== 'upload') return;

        const handlePaste = (e: ClipboardEvent) => {
            if (e.clipboardData && e.clipboardData.files.length > 0) {
                const file = e.clipboardData.files[0];
                if (file.type.startsWith('image/')) {
                    e.preventDefault();
                    const url = URL.createObjectURL(file);
                    setUploadPreview(url);
                    setShowUploadResult(false);
                }
            }
        };

        window.addEventListener('paste', handlePaste);
        return () => window.removeEventListener('paste', handlePaste);
    }, [activeTab, setUploadPreview, setShowUploadResult]);

    // Inference handler for Draw tab
    const handleInference = async (canvas: HTMLCanvasElement, strokes: Stroke[]) => {
        if (!userConfirmed && !isLoadedFromCache) {
            setIsPromptDismissed(false);
            return;
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

    // Image handlers for Upload tab
    const handleImageSelect = (file: File) => {
        const url = URL.createObjectURL(file);
        setUploadPreview(url);
        setShowUploadResult(false);
    };

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

        if (!userConfirmed && !isLoadedFromCache) {
            setIsPromptDismissed(false);
            return;
        }

        const result = await inferFromUrl(uploadPreview);
        if (result) {
            const base64Image = await blobUrlToBase64(uploadPreview);
            addToHistory({
                id: Date.now().toString(),
                latex: result.latex,
                timestamp: Date.now(),
                source: 'upload',
                sessionId,
                imageData: base64Image
            });
            setShowUploadResult(true);
        }
    };

    const handleHistorySelect = (item: HistoryItem) => {
        if (item.source === 'upload' && item.imageData) {
            setUploadPreview(item.imageData);
            setShowUploadResult(true);
            setActiveTab('upload');
        }
        loadFromHistory(item);
    };

    const handleUploadAnother = () => {
        uploadFileInputRef.current?.click();
    };

    // Shared loading overlay for both tabs
    // Shared loading overlay for both tabs
    const renderLoadingOverlay = () => {
        if (customNotification) {
            return (
                <ProgressToast
                    phase={customNotification.message}
                    progress={customNotification.progress ?? 0}
                    isQueued={false}
                    isLoading={customNotification.isLoading}
                />
            );
        }
        return <ProgressToast phase={loadingPhase} progress={progress} isQueued={isGenerationQueued} />;
    };

    const showFullOverlay = (!userConfirmed && !isLoadedFromCache) || status === 'error';

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

            {/* Main Content Area */}
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
                        <Header />

                        {/* Tab Content */}
                        {activeTab === 'draw' ? (
                            <DrawTab
                                onInference={handleInference}
                                renderLoadingOverlay={renderLoadingOverlay}
                            />
                        ) : (
                            <UploadTab
                                onImageSelect={handleImageSelect}
                                onConvert={handleUploadConvert}
                                onUploadAnother={handleUploadAnother}
                                renderLoadingOverlay={renderLoadingOverlay}
                            />
                        )}

                        <MobileBottomNav />
                    </div>
                </div>
            </div>

            {/* Preview Model Input */}
            {showPreviewInput && <PreviewModelInput debugImage={debugImage} />}

            {/* Download Prompt / Error Overlay */}
            {showFullOverlay && (
                <LoadingOverlay
                    isDismissed={isPromptDismissed}
                    onDismiss={() => setIsPromptDismissed(true)}
                />
            )}

            <ConfirmationDialog
                isOpen={dialogConfig.isOpen}
                onCancel={closeDialog}
                onConfirm={dialogConfig.onConfirm}
                title={dialogConfig.title}
                message={dialogConfig.message}
                confirmText={dialogConfig.confirmText}
                isDangerous={dialogConfig.isDangerous}
            />
        </div>
    );
};

export default Main;
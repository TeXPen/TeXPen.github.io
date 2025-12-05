import React from 'react';
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

const Main: React.FC = () => {
    const {
        status,
        latex,
        candidates,
        infer,
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
    } = useAppContext();

    const { theme } = useThemeContext();
    const { history, addToHistory, deleteHistoryItem } = useHistoryContext();

    const handleInference = async (canvas: HTMLCanvasElement) => {
        const result = await infer(canvas);
        if (result) {
            addToHistory({ id: Date.now().toString(), latex: result.latex, timestamp: Date.now() });
        }
    };

    // Only show full overlay for initial model loading, not during inference
    const isInitialLoading = status === 'loading' && loadingPhase.includes('model');
    const showFullOverlay = isInitialLoading || status === 'error';

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

                        <CanvasArea
                            theme={theme}
                            onStrokeEnd={handleInference}
                            onClear={clearModel}
                        />

                        {/* <DebugTest
                            onTest={inferFromUrl}
                            status={status}
                        /> */}
                    </div>
                </div>
            </div>

            {/* Visual Debugger (shows preprocessed image when enabled) */}
            {showVisualDebugger && <VisualDebugger debugImage={debugImage} />}

            {/* Full overlay only for initial model loading or errors */}
            {showFullOverlay && (
                <LoadingOverlay />
            )}
        </div>
    );
};

export default Main;
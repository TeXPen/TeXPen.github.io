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
import DebugTest from './DebugTest';

const Main: React.FC = () => {
    const {
        config,
        setConfig,
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
        toggleSidebar,
        selectedIndex,
        selectCandidate,
        isInferencing,
    } = useAppContext();

    const { theme, toggleTheme } = useThemeContext();
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
                <Header
                    theme={theme}
                    toggleTheme={toggleTheme}
                    config={config}
                    setConfig={setConfig}
                    isSidebarOpen={isSidebarOpen}
                    toggleSidebar={toggleSidebar}
                />

                <div className="flex-1 flex min-h-0 relative">
                    <HistorySidebar
                        history={history}
                        onSelect={loadFromHistory}
                        onDelete={deleteHistoryItem}
                        isOpen={isSidebarOpen}
                    />

                    <div className="flex-1 flex flex-col min-w-0 z-10 relative">
                        <OutputDisplay latex={latex} isInferencing={isInferencing} />

                        <Candidates
                            candidates={candidates}
                            selectedIndex={selectedIndex}
                            onSelect={selectCandidate}
                            status={status}
                        />

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

            {/* Full overlay only for initial model loading or errors */}
            {showFullOverlay && (
                <LoadingOverlay
                    phase={loadingPhase}
                    progress={progress}
                    error={status === 'error' ? 'Failed to load models. Please check your internet connection and try again.' : undefined}
                    needsConfirmation={!userConfirmed}
                    onConfirm={() => setUserConfirmed(true)}
                    isLoadedFromCache={isLoadedFromCache}
                />
            )}
        </div>
    );
};

export default Main;
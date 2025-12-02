import React, { useState } from 'react';
import { HistoryItem } from './types';
import { useTheme } from './hooks/useTheme';
import { useInkModel } from './hooks/useInkModel';
import LiquidBackground from './components/LiquidBackground';
import Candidates from './components/Candidates';
import Header from './components/Header';
import HistorySidebar from './components/HistorySidebar';
import OutputDisplay from './components/OutputDisplay';
import CanvasArea from './components/CanvasArea';
import LoadingOverlay from './components/LoadingOverlay';
import DebugTest from './components/DebugTest'; // <--- Import the debug component

const App: React.FC = () => {
    const { theme, toggleTheme } = useTheme();
    
    // Destructure inferFromUrl here
    const {
        config,
        setConfig,
        status,
        latex,
        setLatex,
        candidates,
        infer,
        inferFromUrl, // <--- Get the new test function
        clear: clearModel,
        progress,
        loadingPhase,
        userConfirmed,
        setUserConfirmed,
        isLoadedFromCache
    } = useInkModel(theme);

    const [history, setHistory] = useState<HistoryItem[]>([]);
    const [selectedIndex, setSelectedIndex] = useState<number>(0);
    const [isSidebarOpen, setIsSidebarOpen] = useState(true);

    const handleInference = async (canvas: HTMLCanvasElement) => {
        const result = await infer(canvas);
        if (result) {
            setSelectedIndex(0);
            setHistory(prev => [
                { id: Date.now().toString(), latex: result.latex, timestamp: Date.now() },
                ...prev
            ].slice(0, 20));
        }
    };

    const handleLoadFromHistory = (item: HistoryItem) => {
        setLatex(item.latex);
        setSelectedIndex(0);
    };

    const handleSelectCandidate = (idx: number) => {
        setSelectedIndex(idx);
        setLatex(candidates[idx].latex);
    };

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
                    toggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)}
                />

                <div className="flex-1 flex min-h-0 relative">

                    <HistorySidebar
                        history={history}
                        onSelect={handleLoadFromHistory}
                        isOpen={isSidebarOpen}
                    />

                    <div className="flex-1 flex flex-col min-w-0 z-10 relative">

                        <OutputDisplay latex={latex} />

                        <Candidates
                            candidates={candidates}
                            selectedIndex={selectedIndex}
                            onSelect={handleSelectCandidate}
                            status={status}
                        />

                        <CanvasArea
                            theme={theme}
                            onStrokeEnd={handleInference}
                            onClear={clearModel}
                        />

                        {/* 
                            --- DEBUG COMPONENT --- 
                            It will float in the bottom right corner of the canvas area 
                        */}
                        <DebugTest 
                            onTest={inferFromUrl} 
                            status={status} 
                        />

                    </div>

                </div>
            </div>

            {(status === 'loading' || status === 'error') && (
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

export default App;
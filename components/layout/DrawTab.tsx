import React from 'react';
import { useAppContext } from '../../contexts/AppContext';
import { useThemeContext } from '../../contexts/ThemeContext';
import { useHistoryContext } from '../../contexts/HistoryContext';
import { Stroke } from '../../types/canvas';
import CanvasArea from '../canvas/CanvasArea';
import OutputDisplay from '../display/OutputDisplay';
import Candidates from '../display/Candidates';

interface DrawTabProps {
    onInference: (canvas: HTMLCanvasElement, strokes: Stroke[]) => Promise<void>;
    renderLoadingOverlay: () => React.ReactNode;
}

const DrawTab: React.FC<DrawTabProps> = ({
    onInference,
    renderLoadingOverlay
}) => {
    const {
        status,
        latex,
        clearModel,
        userConfirmed,
        isInferencing,
        activeInferenceTab,
        loadedStrokes,
        sessionId,
        refreshSession,
        customNotification
    } = useAppContext();

    const { theme } = useThemeContext();
    const { addToHistory } = useHistoryContext();

    const isDrawInferencing = isInferencing && activeInferenceTab === 'draw';

    const handleClear = () => {
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
    };

    return (
        <>
            {/* Output Display Section */}
            <div className="flex-none h-1/4 md:h-2/5 flex flex-col w-full relative z-10 shrink-0">
                <OutputDisplay
                    latex={latex}
                    isInferencing={isDrawInferencing}
                    className="flex-1 w-full"
                />
                <Candidates />
            </div>

            {/* Canvas Workspace */}
            <div className="flex-1 relative overflow-hidden flex flex-col">
                <div className="flex-1 flex flex-col absolute inset-0 z-10">
                    <CanvasArea
                        theme={theme}
                        onStrokeEnd={onInference}
                        initialStrokes={loadedStrokes}
                        onClear={handleClear}
                    />
                    {((status === 'loading' && userConfirmed) || !!customNotification) && renderLoadingOverlay()}
                </div>
            </div>
        </>
    );
};

export default DrawTab;

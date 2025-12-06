import React, { createContext, useContext, useState, useEffect } from 'react';
import { ModelConfig, Candidate, HistoryItem } from '../../types';
import { useInkModel } from '../../hooks/useInkModel';
import { useThemeContext } from './ThemeContext';
import { isWebGPUAvailable } from '../../utils/env';
import { INFERENCE_CONFIG } from '../../services/inference/config';

type Provider = 'webgpu' | 'wasm' | 'webgl';

interface AppContextType {
    // InkModel
    config: ModelConfig;
    setConfig: (config: ModelConfig) => void;
    status: string;
    latex: string;
    setLatex: (latex: string) => void;
    candidates: Candidate[];
    infer: (canvas: HTMLCanvasElement) => Promise<{ latex: string; candidates: Candidate[] } | null>;
    inferFromUrl: (url: string) => Promise<{ latex: string; candidates: Candidate[] } | null>;
    clearModel: () => void;
    loadingPhase: string;
    isInferencing: boolean;
    debugImage: string | null;
    numCandidates: number;
    setNumCandidates: (n: number) => void;
    quantization: string;
    setQuantization: (q: string) => void;
    provider: Provider;
    setProvider: (p: Provider) => void;
    progress: number;
    userConfirmed: boolean;
    setUserConfirmed: (confirmed: boolean) => void;
    isLoadedFromCache: boolean;
    showVisualDebugger: boolean;
    setShowVisualDebugger: (show: boolean) => void;

    // Sidebar
    isSidebarOpen: boolean;
    toggleSidebar: () => void;

    // Selected Candidate
    selectedIndex: number;
    setSelectedIndex: (index: number) => void;
    selectCandidate: (index: number) => void;

    // History Actions
    loadFromHistory: (item: HistoryItem) => void;

    // Tab Interface
    activeTab: 'draw' | 'upload';
    setActiveTab: (tab: 'draw' | 'upload') => void;
}

export const AppContext = createContext<AppContextType | undefined>(undefined);

export const AppProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const { theme } = useThemeContext();
    const [quantization, setQuantization] = useState<string>(INFERENCE_CONFIG.DEFAULT_QUANTIZATION);
    const [provider, setProvider] = useState<Provider>(INFERENCE_CONFIG.DEFAULT_PROVIDER as Provider);
    const [activeTab, setActiveTab] = useState<'draw' | 'upload'>('draw');

    useEffect(() => {
        isWebGPUAvailable().then(available => {
            if (available) {
                setProvider('webgpu');
            }
        });
    }, []);

    const {
        config,
        setConfig,
        status,
        latex,
        setLatex,
        candidates,
        infer,
        inferFromUrl,
        clear: clearModel,
        loadingPhase,
        isInferencing,
        debugImage,
        numCandidates,
        setNumCandidates,
        progress,
        userConfirmed,
        setUserConfirmed,
        isLoadedFromCache,
    } = useInkModel(theme, quantization, provider);

    const [selectedIndex, setSelectedIndex] = useState<number>(0);
    const [isSidebarOpen, setIsSidebarOpen] = useState(true);
    const [showVisualDebugger, setShowVisualDebugger] = useState(false);

    const loadFromHistory = (item: HistoryItem) => {
        setLatex(item.latex);
        setSelectedIndex(0);
        // If history item is selected, we might want to switch to 'draw' tab?
        // Let's assume yes for now.
        setActiveTab('draw');
    };

    const toggleSidebar = () => {
        setIsSidebarOpen(prev => !prev);
    };

    const selectCandidate = (index: number) => {
        setSelectedIndex(index);
        setLatex(candidates[index].latex);
    };

    const value: AppContextType = {
        // InkModel
        config,
        setConfig,
        status,
        latex,
        setLatex,
        candidates,
        infer,
        inferFromUrl,
        clearModel,
        loadingPhase,
        isInferencing,
        debugImage,
        numCandidates,
        setNumCandidates,
        quantization,
        setQuantization,
        provider,
        setProvider,
        progress,
        userConfirmed,
        setUserConfirmed,
        isLoadedFromCache,
        showVisualDebugger,
        setShowVisualDebugger,

        // Sidebar
        isSidebarOpen,
        toggleSidebar,

        // Selected Candidate
        selectedIndex,
        setSelectedIndex,
        selectCandidate,

        // History
        loadFromHistory,

        // Tab
        activeTab,
        setActiveTab
    };

    return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
};

export const useAppContext = () => {
    const context = useContext(AppContext);
    if (!context) {
        throw new Error('useAppContext must be used within an AppProvider');
    }
    return context;
};
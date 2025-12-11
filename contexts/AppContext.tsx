import React, { createContext, useContext, useState, useEffect, useRef } from 'react';
import { ModelConfig, Candidate, HistoryItem } from '../types';
import { Stroke } from '../types/canvas';
import { useInkModel } from '../hooks/useInkModel';
import { useThemeContext } from './ThemeContext';
import { isWebGPUAvailable } from '../utils/env';
import { MODEL_CONFIG } from '../services/inference/config';
import { useTabState } from '../hooks/useTabState';

type Provider = 'webgpu' | 'wasm';

export interface AppContextType {
    // InkModel
    config: ModelConfig;
    setConfig: (config: ModelConfig) => void;
    status: string;
    latex: string;
    setLatex: (latex: string) => void;
    candidates: Candidate[];
    loadedStrokes?: Stroke[] | null;
    infer: (canvas: HTMLCanvasElement) => Promise<{ latex: string; candidates: Candidate[] } | null>;
    inferFromUrl: (url: string) => Promise<{ latex: string; candidates: Candidate[] } | null>;
    clearModel: () => void;
    loadingPhase: string;
    isInferencing: boolean;
    debugImage: string | null;
    numCandidates: number;
    setNumCandidates: (n: number) => void;
    doSample: boolean;
    setDoSample: (b: boolean) => void;
    temperature: number;
    setTemperature: (n: number) => void;
    topK: number;
    setTopK: (n: number) => void;
    topP: number;
    setTopP: (n: number) => void;
    quantization: string;
    setQuantization: (q: string) => void;
    provider: Provider;
    setProvider: (p: Provider) => void;
    progress: number;
    userConfirmed: boolean;
    setUserConfirmed: (confirmed: boolean) => void;

    // Custom Model
    customModelId: string;
    setCustomModelId: (id: string) => void;

    isLoadedFromCache: boolean;
    isInitialized: boolean;
    showVisualDebugger: boolean;
    setShowVisualDebugger: (show: boolean) => void;

    // Settings
    isSettingsOpen: boolean;
    settingsFocus: 'modelId' | null;
    openSettings: (focusTarget?: 'modelId') => void;
    closeSettings: () => void;

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

    // Session
    sessionId: string;
    refreshSession: () => void;

    // Upload State
    uploadPreview: string | null;
    showUploadResult: boolean;
    setUploadPreview: (url: string | null) => void;
    setShowUploadResult: (show: boolean) => void;

    // Inference State
    activeInferenceTab?: 'draw' | 'upload' | null;
}

export const AppContext = createContext<AppContextType | undefined>(undefined);

export const AppProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const { theme } = useThemeContext();
    const [quantization, setQuantization] = useState<string>(MODEL_CONFIG.DEFAULT_QUANTIZATION);
    const [provider, setProvider] = useState<Provider>(MODEL_CONFIG.DEFAULT_PROVIDER as Provider);
    const [customModelId, setCustomModelId] = useState<string>(MODEL_CONFIG.ID);
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
        infer: modelInfer,
        inferFromUrl: modelInferFromUrl,
        loadingPhase,
        isInferencing,
        numCandidates,
        setNumCandidates,
        doSample,
        setDoSample,
        temperature,
        setTemperature,
        topK,
        setTopK,
        topP,
        setTopP,
        progress,
        userConfirmed,
        setUserConfirmed,
        isLoadedFromCache,
        isInitialized,
    } = useInkModel(theme, quantization, provider, customModelId);

    // Use the extracted tab state hook
    const {
        latex,
        candidates,
        selectedIndex,
        debugImage,
        loadedStrokes,
        uploadPreview,
        showUploadResult,
        setLatex,
        setSelectedIndex,
        selectCandidate,
        setUploadPreview,
        setShowUploadResult,
        clearTabState,
        updateDrawResult,
        updateUploadResult,
        loadDrawState,
        setDrawState,
        activeInferenceTab,
        startDrawInference,
        endDrawInference,
        startUploadInference,
        endUploadInference,
    } = useTabState(activeTab);

    const clearModel = () => {
        clearTabState();
    };

    // Wrappers for inference to update the correct state
    const infer = async (canvas: HTMLCanvasElement) => {
        startDrawInference();

        try {
            const result = await modelInfer(canvas);
            if (result) {
                updateDrawResult(result);
                return result;
            }
            return null;
        } finally {
            endDrawInference();
        }
    };

    const inferFromUrl = async (url: string) => {
        startUploadInference();

        try {
            const result = await modelInferFromUrl(url);
            if (result) {
                updateUploadResult(result);
                return result;
            }
            return null;
        } finally {
            endUploadInference();
        }
    };

    const [isSidebarOpen, setIsSidebarOpen] = useState(() => {
        if (typeof window !== 'undefined') {
            return window.innerWidth >= 768;
        }
        return true;
    });
    const [showVisualDebugger, setShowVisualDebugger] = useState(false);
    const [sessionId, setSessionId] = useState<string>(Date.now().toString());

    // Settings State
    const [isSettingsOpen, setIsSettingsOpen] = useState(false);
    const [settingsFocus, setSettingsFocus] = useState<'modelId' | null>(null);

    const openSettings = (focusTarget?: 'modelId') => {
        setIsSettingsOpen(true);
        setSettingsFocus(focusTarget || null);
    };

    const closeSettings = () => {
        setIsSettingsOpen(false);
        setSettingsFocus(null);
    };

    const refreshSession = () => {
        setSessionId(Date.now().toString());
    };

    const loadFromHistory = (item: HistoryItem) => {
        loadDrawState(item.latex, item.strokes || null);
        setActiveTab('draw');
        refreshSession();
    };

    const toggleSidebar = () => {
        setIsSidebarOpen(prev => !prev);
    };

    const value: AppContextType = {
        // InkModel
        config,
        setConfig,
        status,
        latex,
        setLatex,
        candidates,
        loadedStrokes,
        infer,
        inferFromUrl,
        clearModel,
        loadingPhase,
        isInferencing,
        debugImage,
        numCandidates,
        setNumCandidates,
        doSample,
        setDoSample,
        temperature,
        setTemperature,
        topK,
        setTopK,
        topP,
        setTopP,
        quantization,
        setQuantization,
        provider,
        setProvider,
        progress,
        userConfirmed,
        setUserConfirmed,
        customModelId,
        setCustomModelId,
        isLoadedFromCache,
        isInitialized,
        showVisualDebugger,
        setShowVisualDebugger,

        // Settings
        isSettingsOpen,
        settingsFocus,
        openSettings,
        closeSettings,

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
        setActiveTab,

        // Session
        sessionId,
        refreshSession,

        // Upload State
        uploadPreview,
        showUploadResult,
        setUploadPreview,
        setShowUploadResult,

        // Inference State
        activeInferenceTab,
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
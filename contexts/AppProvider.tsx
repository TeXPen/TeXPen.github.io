import React, { useState, useEffect } from 'react';
import { AppContext, AppContextType, Provider } from './AppContext';
import { useInkModel } from '../hooks/useInkModel';
import { useThemeContext } from './ThemeContext';
import { isWebGPUAvailable } from '../utils/env';
import { MODEL_CONFIG } from '../services/inference/config';
import { useTabState } from '../hooks/useTabState';
import { HistoryItem } from '../types';

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
        isGenerationQueued,
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
        setUploadState,
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
            const result = await modelInfer(canvas, {
                onPreprocess: (debugImage) => {
                    setDrawState(prev => ({ ...prev, debugImage }));
                }
            });
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
            const result = await modelInferFromUrl(url, {
                onPreprocess: (debugImage) => {
                    setUploadState(prev => ({ ...prev, debugImage }));
                }
            });
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
    const [showPreviewInput, setShowPreviewInput] = useState(false);
    const [sessionId, setSessionId] = useState<string>(Date.now().toString());

    // Settings State
    const [isSettingsOpen, setIsSettingsOpen] = useState(false);
    const [settingsFocus, setSettingsFocus] = useState<'modelId' | null>(null);

    // Custom Notification
    const [customNotification, setCustomNotification] = useState<string | null>(null);

    // Dialog configuration
    const [dialogConfig, setDialogConfig] = useState<{
        isOpen: boolean;
        title: string;
        message: string;
        onConfirm: () => void;
        confirmText?: string;
        isDangerous?: boolean;
    }>({
        isOpen: false,
        title: '',
        message: '',
        onConfirm: () => { },
    });

    const openDialog = (config: {
        title: string;
        message: string;
        onConfirm: () => void;
        confirmText?: string;
        isDangerous?: boolean;
    }) => {
        setDialogConfig({
            isOpen: true,
            ...config
        });
    };

    const closeDialog = () => {
        setDialogConfig(prev => ({ ...prev, isOpen: false }));
    };

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
        isGenerationQueued,
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
        showPreviewInput,
        setShowPreviewInput,

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

        // Custom Notification
        customNotification,
        setCustomNotification,

        // Dialog
        dialogConfig,
        openDialog,
        closeDialog,
    };

    return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
};

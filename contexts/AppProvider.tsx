import React, { useState, useEffect } from 'react';
import { AppContext, AppContextType, Provider } from './AppContext';
import { useInkModel } from '../hooks/useInkModel';
import { useThemeContext } from './ThemeContext';
import { isWebGPUAvailable, getDefaultProfile } from '../utils/env';
import { MODEL_CONFIG } from '../services/inference/config';
import { useTabState } from '../hooks/useTabState';
import { HistoryItem } from '../types';
import { Quantization, PerformanceProfile } from '../services/inference/types';

export const AppProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const { theme } = useThemeContext();
    const [provider, setProvider] = useState<Provider | null>(null);
    const [quantization, setQuantization] = useState<Quantization>('int8');
    const [performanceProfile, setPerformanceProfile] = useState<PerformanceProfile>('balanced');
    const [encoderQuantization, setEncoderQuantization] = useState<Quantization>('int8');
    const [decoderQuantization, setDecoderQuantization] = useState<Quantization>('int8');
    const [customModelId, setCustomModelId] = useState<string>(MODEL_CONFIG.ID);
    const [activeTab, setActiveTab] = useState<'draw' | 'upload'>('draw');

    useEffect(() => {
        isWebGPUAvailable().then(available => {
            console.log('[AppProvider] WebGPU Available:', available);
            if (available) {
                setProvider('webgpu');
            } else {
                setProvider('wasm');
            }
        });

        getDefaultProfile().then(defaultProfile => {
            console.log('[AppProvider] Default Profile:', defaultProfile);
            setPerformanceProfile(defaultProfile);
        });
    }, []);

    // Auto-switch profile when provider changes to prevent invalid states
    useEffect(() => {
        if (!provider) return;

        // If switching to WASM (CPU)
        if (provider === 'wasm') {
            // If current profile is not widely compatible or slow, switch to balanced
            // Or if custom quantizations are FP16, we should probably reset/warn.
            if (performanceProfile === 'fast' || performanceProfile === 'high_quality') {
                console.log('[AppProvider] Switching to WASM: Resetting profile to Balanced');
                setPerformanceProfile('balanced');
            } else if (performanceProfile === 'custom') {
                // Check if custom settings are invalid for CPU (e.g. FP16)
                // This is harder since we don't want to override user intent if they really want it.
                // But FP16 on CPU is explicitly "very very slow so it should not appear".
                // Let's force reset to balanced if encoder is FP16.
                if (encoderQuantization === 'fp16' || decoderQuantization === 'fp16') { // decoder FP16 is gone anyway but for safety
                    console.log('[AppProvider] Switching to WASM: Custom profile has FP16, resetting to Balanced');
                    setPerformanceProfile('balanced');
                }
            }
        }
        // If switching to WebGPU
        else if (provider === 'webgpu') {
            // If current profile is balanced/low_memory, should we switch to High Quality?
            // User said: "default to Highest Quality" for GPU.
            // If user previously selected balanced on GPU, maybe keep it?
            // But they said "hide int8/int4 in GPU tab", implying they shouldn't even be there.
            // So if we are on GPU and have an int8 profile, we MUST switch.
            if (performanceProfile === 'balanced' || performanceProfile === 'low_memory') {
                console.log('[AppProvider] Switching to WebGPU: Resetting profile to High Quality');
                setPerformanceProfile('high_quality');
            } else if (performanceProfile === 'custom') {
                if (encoderQuantization === 'int8' || encoderQuantization === 'int4') {
                    console.log('[AppProvider] Switching to WebGPU: Custom profile has Int8/4, resetting to High Quality');
                    setPerformanceProfile('high_quality');
                }
            }
        }
    }, [provider, performanceProfile, encoderQuantization, decoderQuantization]);

    // Effect to update quantizations based on performance profile
    useEffect(() => {
        if (performanceProfile === 'custom') return;

        let enc: Quantization = 'int8';
        let dec: Quantization = 'int8';

        switch (performanceProfile) {
            case 'high_quality':
                enc = 'fp32';
                dec = 'fp32';
                break;
            case 'fast':
                enc = 'fp16';
                dec = 'fp32';
                break;
            case 'balanced':
                enc = 'int8';
                dec = 'int8';
                break;
            case 'low_memory':
                enc = 'int4';
                dec = 'int4';
                break;
        }

        console.log(`[AppProvider] Applying profile ${performanceProfile}: Enc=${enc}, Dec=${dec}`);
        setQuantization(enc);
        setEncoderQuantization(enc);
        setDecoderQuantization(dec);
    }, [performanceProfile]);

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
    } = useInkModel(theme, provider, quantization, encoderQuantization, decoderQuantization, customModelId);

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
    const [customNotification, setCustomNotification] = useState<{ message: string; progress?: number; isLoading?: boolean } | null>(null);

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
        provider: provider || 'wasm',
        setProvider,
        quantization,
        setQuantization,
        encoderQuantization,
        setEncoderQuantization,
        decoderQuantization,
        setDecoderQuantization,
        performanceProfile,
        setPerformanceProfile,
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

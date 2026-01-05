import React, { useState, useEffect, useCallback } from 'react';
import { AppContext, AppContextType, Provider } from './AppContext';
import { useInkModel } from '../hooks/useInkModel';
import { useThemeContext } from './ThemeContext';
import { getDeviceCapabilities } from '../utils/env';
import { MODEL_CONFIG } from '../services/inference/config';
import { useTabState } from '../hooks/useTabState';
import { HistoryItem } from '../types';
import { Quantization, PerformanceProfile } from '../services/inference/types';

const STORAGE_KEY = 'texpen_settings_v1';

interface SavedSettings {
    provider: Provider;
    performanceProfile: PerformanceProfile;
    encQuantization?: Quantization;
    decQuantization?: Quantization;
    customModelId?: string;
}

export const AppProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const { theme } = useThemeContext();

    // Initial states with safe defaults
    const [provider, setProviderState] = useState<Provider | null>(null);
    const [performanceProfile, setPerformanceProfileState] = useState<PerformanceProfile>('balanced');
    const [quantization, setQuantizationState] = useState<Quantization>('int8');
    const [encoderQuantization, setEncoderQuantizationState] = useState<Quantization>('int8');
    const [decoderQuantization, setDecoderQuantizationState] = useState<Quantization>('int8');
    const [customModelId, setCustomModelIdState] = useState<string>(MODEL_CONFIG.ID);

    // UI State
    const [activeTab, setActiveTab] = useState<'draw' | 'upload' | 'scan'>('draw');
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

    // Notification & Dialog
    const [customNotification, setCustomNotification] = useState<{ message: string; progress?: number; isLoading?: boolean } | null>(null);
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


    // Helper to resolve quantization from profile
    const resolveQuantization = (profile: PerformanceProfile): { enc: Quantization, dec: Quantization } | null => {
        switch (profile) {
            case 'high_quality': return { enc: 'fp32', dec: 'fp32' };
            case 'fast': return { enc: 'fp16', dec: 'fp32' };
            case 'balanced': return { enc: 'int8', dec: 'int8' };
            case 'low_memory': return { enc: 'int4', dec: 'int4' };
            default: return null; // 'custom' or unknown
        }
    };

    // Unified State Updater & Saver
    const updateSettings = useCallback((
        newProvider: Provider,
        newProfile: PerformanceProfile,
        customSettings?: { enc: Quantization, dec: Quantization, modelId?: string }
    ) => {
        setProviderState(newProvider);
        setPerformanceProfileState(newProfile);

        let enc: Quantization, dec: Quantization;

        if (newProfile === 'custom' && customSettings) {
            enc = customSettings.enc;
            dec = customSettings.dec;
        } else {
            const resolved = resolveQuantization(newProfile);
            if (resolved) {
                enc = resolved.enc;
                dec = resolved.dec;
            } else {
                // Fallback / Maintain current if feasible
                // Since this is inside a callback, we can't easily read current state 
                // unless we use refs or pass it in. 
                // For safety, default to balanced if undefined map.
                enc = 'int8';
                dec = 'int8';
            }
        }

        setQuantizationState(enc);
        setEncoderQuantizationState(enc);
        setDecoderQuantizationState(dec);

        const newModelId = customSettings?.modelId || customModelId || MODEL_CONFIG.ID;
        if (customSettings?.modelId) {
            setCustomModelIdState(customSettings.modelId);
        }

        // Save to storage
        try {
            const settingsToSave: SavedSettings = {
                provider: newProvider,
                performanceProfile: newProfile,
                encQuantization: enc,
                decQuantization: dec,
                customModelId: newModelId
            };
            localStorage.setItem(STORAGE_KEY, JSON.stringify(settingsToSave));
        } catch (e) {
            console.error("Failed to save settings", e);
        }

    }, [customModelId]);

    // Initialization Effect
    useEffect(() => {
        const init = async () => {
            const caps = await getDeviceCapabilities();
            const savedStr = localStorage.getItem(STORAGE_KEY);
            let saved: SavedSettings | null = null;
            if (savedStr) {
                try { saved = JSON.parse(savedStr); } catch (e) { console.error("Failed to parse settings", e); }
            }

            let loadedProvider: Provider = saved?.provider || (caps.hasGPU ? 'webgpu' : 'wasm');

            // Validate GPU availability
            if (loadedProvider === 'webgpu' && !caps.hasGPU) {
                console.log("[AppProvider] WebGPU preference found but not available, falling back to WASM");
                loadedProvider = 'wasm';
            } else {
                console.log("[AppProvider] Initializing Provider:", loadedProvider);
            }

            // Determine Profile
            let loadedProfile: PerformanceProfile = saved?.performanceProfile || (loadedProvider === 'webgpu' ? 'high_quality' : 'balanced');

            // Resolve Quantization
            let enc: Quantization = 'int8', dec: Quantization = 'int8';

            if (loadedProfile === 'custom' && saved?.encQuantization) {
                enc = saved.encQuantization;
                dec = saved.decQuantization || 'int8';
            } else {
                const res = resolveQuantization(loadedProfile);
                if (res) { enc = res.enc; dec = res.dec; }
                else {
                    // Fallback if profile invalid
                    loadedProfile = 'balanced';
                    enc = 'int8'; dec = 'int8';
                }
            }

            const loadedModelId = saved?.customModelId || MODEL_CONFIG.ID;

            // Apply all directly (first render will have defaults, then this updates)
            setProviderState(loadedProvider);
            setPerformanceProfileState(loadedProfile);
            setQuantizationState(enc);
            setEncoderQuantizationState(enc);
            setDecoderQuantizationState(dec);
            setCustomModelIdState(loadedModelId);
        };

        init();
    }, []);

    // Provider Setter Logic
    const setProvider = (p: Provider) => {
        // Optimization: if same, do nothing? 
        // Be careful if other states need sync, but generally yes.
        if (p === provider) return;

        // Auto-switch logic
        let newProfile = performanceProfile;

        if (p === 'webgpu') {
            // Upgrade to HQ if coming from balanced/low or if forced by logic
            // User wants: "if user has GPU... load fp32... load that and also display that"
            if (performanceProfile === 'balanced' || performanceProfile === 'low_memory') {
                console.log('[AppProvider] Switching to WebGPU: Upgrading profile to High Quality');
                newProfile = 'high_quality';
            }
        } else if (p === 'wasm') {
            // "When you switch from WebGPU to WASM it shouldn't display int8 and the UI and yet load fp32."
            // User requested NOT to force int8 if they have fp32 loaded/selected.
            // So we keep the current profile (even if it's High Quality/fp32)
            console.log('[AppProvider] Switching to WASM: Keeping current profile');
        }

        // Apply
        if (newProfile === 'custom') {
            updateSettings(p, newProfile, { enc: encoderQuantization, dec: decoderQuantization, modelId: customModelId });
        } else {
            updateSettings(p, newProfile, { enc: 'int8', dec: 'int8', modelId: customModelId }); // Enc/Dec ignored by updateSettings for non-custom, but passed for type safety if refactored
        }
    };

    const setPerformanceProfile = (p: PerformanceProfile) => {
        if (p === performanceProfile) return;
        updateSettings(provider || 'wasm', p,
            { enc: encoderQuantization, dec: decoderQuantization, modelId: customModelId }
        );
    };

    // Granular Setters
    const setQuantization = (q: Quantization) => {
        // Changing simple quantization usually implies 'enc' and 'dec' 
        // But in this app 'quantization' usually maps to encoder for display?
        // Let's assume this updates both for simplicity if calling the top level setter?
        // Or if we are in 'high_quality', this setter shouldn't be called?
        // We will assume this switches to 'custom' profile.

        updateSettings(provider || 'wasm', 'custom', {
            enc: q,
            dec: decoderQuantization === 'fp32' ? 'fp32' : q, // Try to keep decoder consistent? Or just update both?
            // Actually usually 'setQuantization' implies overall.
            // Let's update both to 'q' to be safe, or just encoder.
            modelId: customModelId
        });

        // Wait, 'setQuantization' in original code updated 'quantization' state but NOT enc/dec state directly in the Effect 
        // The effect (lines 106-107 original) updated enc/dec based on profile.
        // But what if user manually sets quantization?
        // We should treat it as 'custom'.
    };

    const setEncoderQuantization = (q: Quantization) => {
        updateSettings(provider || 'wasm', 'custom', {
            enc: q,
            dec: decoderQuantization,
            modelId: customModelId
        });
    };

    const setDecoderQuantization = (q: Quantization) => {
        updateSettings(provider || 'wasm', 'custom', {
            enc: encoderQuantization,
            dec: q,
            modelId: customModelId
        });
    };

    const setCustomModelId = (id: string) => {
        updateSettings(provider || 'wasm', performanceProfile, {
            enc: encoderQuantization,
            dec: decoderQuantization,
            modelId: id
        });
    };

    // Hooks
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

    // Methods
    const clearModel = () => clearTabState();

    const infer = async (canvas: HTMLCanvasElement) => {
        startDrawInference();
        try {
            const result = await modelInfer(canvas, {
                onPreprocess: (img) => setDrawState(prev => ({ ...prev, debugImage: img }))
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
                onPreprocess: (img) => setUploadState(prev => ({ ...prev, debugImage: img }))
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

    // Dialog & UI logic
    const openDialog = (cfg: { title: string; message: string; onConfirm: () => void; confirmText?: string; isDangerous?: boolean }) => {
        setDialogConfig({ isOpen: true, ...cfg });
    };
    const closeDialog = () => setDialogConfig(prev => ({ ...prev, isOpen: false }));

    const openSettings = (focusTarget?: 'modelId') => {
        setIsSettingsOpen(true);
        setSettingsFocus(focusTarget || null);
    };
    const closeSettings = () => {
        setIsSettingsOpen(false);
        setSettingsFocus(null);
    };

    const refreshSession = () => setSessionId(Date.now().toString());

    const loadFromHistory = (item: HistoryItem) => {
        loadDrawState(item.latex, item.strokes || null);
        setActiveTab('draw');
        refreshSession();
    };

    const toggleSidebar = () => setIsSidebarOpen(prev => !prev);

    const value: AppContextType = {
        config, setConfig,
        status, latex, setLatex, candidates, loadedStrokes,
        infer, inferFromUrl, clearModel,
        loadingPhase, isInferencing, isGenerationQueued, debugImage,
        numCandidates, setNumCandidates, doSample, setDoSample,
        temperature, setTemperature, topK, setTopK, topP, setTopP,
        provider: provider || 'wasm',
        setProvider,
        quantization, setQuantization,
        encoderQuantization, setEncoderQuantization,
        decoderQuantization, setDecoderQuantization,
        performanceProfile, setPerformanceProfile,
        progress, userConfirmed, setUserConfirmed,
        customModelId, setCustomModelId,
        isLoadedFromCache, isInitialized,
        showPreviewInput, setShowPreviewInput,
        isSettingsOpen, settingsFocus, openSettings, closeSettings,
        isSidebarOpen, toggleSidebar,
        selectedIndex, setSelectedIndex, selectCandidate,
        loadFromHistory,
        activeTab, setActiveTab,
        sessionId, refreshSession,
        uploadPreview, showUploadResult, setUploadPreview, setShowUploadResult,
        activeInferenceTab,
        customNotification, setCustomNotification,
        dialogConfig, openDialog, closeDialog
    };

    return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
};

// Imports cleaned up
import { createContext, useContext } from 'react';
import { ModelConfig, Candidate, HistoryItem } from '../types';
import { Quantization, PerformanceProfile } from '../services/inference/types';
import { Stroke } from '../types/canvas';

export type Provider = 'webgpu' | 'wasm';

export interface AppContextType {
    // InkModel
    config: ModelConfig;
    setConfig: (config: ModelConfig) => void;
    status: string;
    latex: string;
    setLatex: (latex: string) => void;
    candidates: Candidate[];
    loadedStrokes?: Stroke[] | null;
    infer: (canvas: HTMLCanvasElement, options?: { onPreprocess?: (debugImage: string) => void }) => Promise<{ latex: string; candidates: Candidate[] } | null>;
    inferFromUrl: (url: string, options?: { onPreprocess?: (debugImage: string) => void }) => Promise<{ latex: string; candidates: Candidate[] } | null>;
    clearModel: () => void;
    loadingPhase: string;
    isInferencing: boolean;
    isGenerationQueued: boolean;
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
    provider: Provider;
    setProvider: (p: Provider) => void;
    quantization: Quantization;
    setQuantization: (q: Quantization) => void;
    encoderQuantization: Quantization;
    setEncoderQuantization: (q: Quantization) => void;
    decoderQuantization: Quantization;
    setDecoderQuantization: (q: Quantization) => void;
    performanceProfile: PerformanceProfile;
    setPerformanceProfile: (p: PerformanceProfile) => void;
    progress: number;
    userConfirmed: boolean;
    setUserConfirmed: (confirmed: boolean) => void;

    // Custom Model
    customModelId: string;
    setCustomModelId: (id: string) => void;

    isLoadedFromCache: boolean;
    isInitialized: boolean;
    showPreviewInput: boolean;
    setShowPreviewInput: (show: boolean) => void;

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

    // Custom Notification
    customNotification: { message: string; progress?: number; isLoading?: boolean } | null;
    setCustomNotification: (notification: { message: string; progress?: number; isLoading?: boolean } | null) => void;

    // Global Confirmation Dialog
    dialogConfig: {
        isOpen: boolean;
        title: string;
        message: string;
        onConfirm: () => void;
        confirmText?: string;
        isDangerous?: boolean;
    };
    openDialog: (config: {
        title: string;
        message: string;
        onConfirm: () => void;
        confirmText?: string;
        isDangerous?: boolean;
    }) => void;
    closeDialog: () => void;
}

export const AppContext = createContext<AppContextType | undefined>(undefined);


export const useAppContext = () => {
    const context = useContext(AppContext);
    if (!context) {
        throw new Error('useAppContext must be used within an AppProvider');
    }
    return context;
};
import { useState, useRef } from 'react';
import { Candidate } from '../types';
import { Stroke } from '../types/canvas';

/**
 * State for a single tab (draw or upload)
 */
export interface TabState {
  latex: string;
  candidates: Candidate[];
  selectedIndex: number;
  debugImage: string | null;
  uploadPreview: string | null;
  showUploadResult: boolean;
  loadedStrokes: Stroke[] | null;
  paragraphResult: string;
  inferenceMode: 'formula' | 'paragraph';
}

const initialTabState: TabState = {
  latex: '',
  candidates: [],
  selectedIndex: 0,
  debugImage: null,
  uploadPreview: null,
  showUploadResult: false,
  loadedStrokes: null,
  paragraphResult: '',
  inferenceMode: 'formula'
};

/**
 * Hook for managing tab-specific state (draw/upload tabs have separate state)
 */
export function useTabState(activeTab: 'draw' | 'upload' | 'vlm') {
  const [drawState, setDrawState] = useState<TabState>(initialTabState);
  const [uploadState, setUploadState] = useState<TabState>(initialTabState);

  // Get current active state based on tab
  // For VLM (or other tabs), we default to drawState or manage separately. 
  // Since VLM demo is currently standalone, we can share drawState or ignore.
  // Using drawState as fallback for now.
  const currentState = activeTab === 'upload' ? uploadState : drawState;
  const setCurrentState = activeTab === 'upload' ? setUploadState : setDrawState;

  // Derived values for context consumers
  const latex = currentState.latex;
  const candidates = currentState.candidates;
  const selectedIndex = currentState.selectedIndex;
  const debugImage = currentState.debugImage;
  const loadedStrokes = drawState.loadedStrokes; // Always from draw state
  const paragraphResult = currentState.paragraphResult;
  const inferenceMode = currentState.inferenceMode;

  // Upload specific (always from uploadState for persistence)
  const uploadPreview = uploadState.uploadPreview;
  const showUploadResult = uploadState.showUploadResult;

  // Setters
  const setLatex = (val: string) => {
    setCurrentState(prev => ({ ...prev, latex: val }));
  };

  const setSelectedIndex = (val: number) => {
    setCurrentState(prev => ({ ...prev, selectedIndex: val }));
  };

  const selectCandidate = (index: number) => {
    setCurrentState(prev => ({
      ...prev,
      selectedIndex: index,
      latex: prev.candidates[index]?.latex || ''
    }));
  };

  const setUploadPreview = (url: string | null) => {
    setUploadState(prev => ({ ...prev, uploadPreview: url }));
  };

  const setShowUploadResult = (show: boolean) => {
    setUploadState(prev => ({ ...prev, showUploadResult: show }));
  };

  const setParagraphResult = (val: string) => {
    setCurrentState(prev => ({ ...prev, paragraphResult: val }));
  };

  const setInferenceMode = (mode: 'formula' | 'paragraph') => {
    setCurrentState(prev => ({ ...prev, inferenceMode: mode }));
  };

  const clearTabState = () => {
    setCurrentState(prev => ({
      ...initialTabState,
      inferenceMode: prev.inferenceMode
    }));
  };

  // Update draw state with inference result
  const updateDrawResult = (result: { latex: string; candidates: Candidate[]; debugImage: string | null }) => {
    setDrawState(prev => ({
      ...prev,
      latex: result.latex,
      candidates: result.candidates,
      selectedIndex: 0,
      debugImage: result.debugImage
    }));
  };

  // Update draw state with paragraph result
  const updateDrawParagraphResult = (result: { markdown: string; debugImage?: string }) => {
    setDrawState(prev => ({
      ...prev,
      paragraphResult: result.markdown,
      debugImage: result.debugImage || prev.debugImage
    }));
  };

  // Update upload state with inference result
  const updateUploadResult = (result: { latex: string; candidates: Candidate[]; debugImage: string | null }) => {
    setUploadState(prev => ({
      ...prev,
      latex: result.latex,
      candidates: result.candidates,
      selectedIndex: 0,
      debugImage: result.debugImage
    }));
  };

  // Load from history item
  const loadDrawState = (latex: string, strokes: Stroke[] | null) => {
    setDrawState(prev => ({
      latex,
      candidates: [],
      selectedIndex: 0,
      debugImage: null,
      uploadPreview: null,
      showUploadResult: false,
      loadedStrokes: strokes,
      paragraphResult: '',
      inferenceMode: prev.inferenceMode
    }));
  };

  // Track which tab is performing inference
  const [activeInferenceTab, setActiveInferenceTab] = useState<'draw' | 'upload' | null>(null);
  const activeRequestsRef = useRef<{ draw: number; upload: number }>({ draw: 0, upload: 0 });

  const startDrawInference = () => {
    activeRequestsRef.current.draw += 1;
    setActiveInferenceTab('draw');
  };

  const endDrawInference = () => {
    activeRequestsRef.current.draw -= 1;
    if (activeRequestsRef.current.draw === 0) {
      setActiveInferenceTab(prev => prev === 'draw' ? null : prev);
    }
  };

  const startUploadInference = () => {
    activeRequestsRef.current.upload += 1;
    setActiveInferenceTab('upload');
  };

  const endUploadInference = () => {
    activeRequestsRef.current.upload -= 1;
    if (activeRequestsRef.current.upload === 0) {
      setActiveInferenceTab(prev => prev === 'upload' ? null : prev);
    }
  };

  return {
    // Current state values
    latex,
    candidates,
    selectedIndex,
    debugImage,
    loadedStrokes,
    uploadPreview,
    showUploadResult,

    // Setters
    setLatex,
    setSelectedIndex,
    selectCandidate,
    setUploadPreview,
    setShowUploadResult,
    clearTabState,

    // Result updaters
    updateDrawResult,
    updateUploadResult,
    loadDrawState,

    // Raw state setters for advanced use
    setDrawState,
    setUploadState,

    // Inference tracking
    activeInferenceTab,
    startDrawInference,
    endDrawInference,
    startUploadInference,
    endUploadInference,
    paragraphResult,
    setParagraphResult,
    inferenceMode,
    setInferenceMode,
    updateDrawParagraphResult,
  };
}

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
}

const initialTabState: TabState = {
  latex: '',
  candidates: [],
  selectedIndex: 0,
  debugImage: null,
  uploadPreview: null,
  showUploadResult: false,
  loadedStrokes: null
};

/**
 * Hook for managing tab-specific state (draw/upload tabs have separate state)
 */
export function useTabState(activeTab: 'draw' | 'upload' | 'scan') {
  const [drawState, setDrawState] = useState<TabState>(initialTabState);
  const [uploadState, setUploadState] = useState<TabState>(initialTabState);
  const [scanState, setScanState] = useState<TabState>(initialTabState);

  // Get current active state based on tab
  const currentState = activeTab === 'draw' ? drawState : (activeTab === 'upload' ? uploadState : scanState);
  const setCurrentState = activeTab === 'draw' ? setDrawState : (activeTab === 'upload' ? setUploadState : setScanState);

  // Derived values for context consumers
  const latex = currentState.latex;
  const candidates = currentState.candidates;
  const selectedIndex = currentState.selectedIndex;
  const debugImage = currentState.debugImage;
  const loadedStrokes = drawState.loadedStrokes; // Always from draw state

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

  const clearTabState = () => {
    setCurrentState(initialTabState);
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
    setDrawState({
      latex,
      candidates: [],
      selectedIndex: 0,
      debugImage: null,
      uploadPreview: null,
      showUploadResult: false,
      loadedStrokes: strokes
    });
  };

  // Track which tab is performing inference
  const [activeInferenceTab, setActiveInferenceTab] = useState<'draw' | 'upload' | 'scan' | null>(null);
  const activeRequestsRef = useRef<{ draw: number; upload: number; scan: number }>({ draw: 0, upload: 0, scan: 0 });

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
  };
}

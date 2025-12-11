// @vitest-environment jsdom
import { renderHook, act, waitFor } from '@testing-library/react';
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { useInkModel } from '../../hooks/useInkModel';
import { inferenceService } from '../../services/inference/InferenceService';

// Mock the global caches API
const mockCache = {
  keys: vi.fn(),
  open: vi.fn(),
  match: vi.fn(),
  put: vi.fn(),
  delete: vi.fn(),
  addAll: vi.fn(),
  add: vi.fn(),
};

global.caches = mockCache as any;

// Mock InferenceService
vi.mock('../../services/inference/InferenceService', () => ({
  inferenceService: {
    init: vi.fn(),
    infer: vi.fn(),
    dispose: vi.fn().mockResolvedValue(undefined),
  }
}));

describe('useInkModel', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Default cache mock: Empty
    mockCache.open.mockResolvedValue({
      keys: vi.fn().mockResolvedValue([])
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('initializes in idle state', async () => {
    const { result } = renderHook(() => useInkModel('light', 'fp32', 'webgpu'));

    // Initial state is idle, not confirmed unless cached
    expect(result.current.status).toBe('idle');
    expect(result.current.userConfirmed).toBe(false);
    expect(result.current.isLoadedFromCache).toBe(false);

    // Wait for checkCache effect
    await waitFor(() => {
      expect(result.current.isInitialized).toBe(true);
    });
  });

  it('detects cached model and auto-confirms', async () => {
    // Mock cache hitting the model
    mockCache.open.mockResolvedValue({
      keys: vi.fn().mockResolvedValue([{ url: 'some-url/Ji-Ha/TexTeller3-ONNX-dynamic' }]) // Url match
    });

    const { result } = renderHook(() => useInkModel('light', 'fp32', 'webgpu'));

    await waitFor(() => {
      expect(result.current.isLoadedFromCache).toBe(true);
      expect(result.current.userConfirmed).toBe(true);
    });
  });

  it('starts loading when confirmed', async () => {
    const { result } = renderHook(() => useInkModel('light', 'fp32', 'webgpu'));

    // Manually confirm
    act(() => {
      result.current.setUserConfirmed(true);
    });

    // Should trigger initModel effect
    // However, initModel is async. status might flicker to 'loading' then 'idle' if mock resolves fast.
    // Let's make inferenceService.init simulate loading
    (inferenceService.init as any).mockImplementation(async (cb: any) => {
      cb('Loading model...', 0);
      await new Promise(r => setTimeout(r, 10));
      cb('Ready', 100);
    });

    await waitFor(() => {
      expect(inferenceService.init).toHaveBeenCalled();
    });

    // We can't easily catch the 'loading' state in a test without more control, 
    // but we can verify init was called with correct params
    expect(inferenceService.init).toHaveBeenCalledWith(expect.any(Function), {
      dtype: 'fp32',
      device: 'webgpu',
      modelId: expect.any(String)
    });
  });

  it('performs inference when infer is called', async () => {
    const { result } = renderHook(() => useInkModel('light', 'fp32', 'webgpu'));

    // Setup: Confirmed and Loaded
    act(() => {
      result.current.setUserConfirmed(true);
    });

    // Mock Canvas
    const mockCanvas = document.createElement('canvas');
    mockCanvas.toBlob = vi.fn((cb) => cb(new Blob([''], { type: 'image/png' })));

    // Mock Service Result
    (inferenceService.infer as any).mockResolvedValue({
      latex: 'LatexResult',
      candidates: ['LatexResult'],
      debugImage: 'data:image/png;base64,...'
    });

    let output;
    await act(async () => {
      output = await result.current.infer(mockCanvas);
    });

    expect(inferenceService.infer).toHaveBeenCalled();
    expect(output).toEqual({
      latex: 'LatexResult',
      candidates: [{ id: 0, latex: 'LatexResult' }],
      debugImage: 'data:image/png;base64,...'
    });
    expect(result.current.status).toBe('success');
  });

  it('skips inference if not confirmed and not cached', async () => {
    const { result } = renderHook(() => useInkModel('light', 'fp32', 'webgpu'));

    expect(result.current.userConfirmed).toBe(false);

    const mockCanvas = document.createElement('canvas');

    const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => { });

    let output;
    await act(async () => {
      output = await result.current.infer(mockCanvas);
    });

    expect(output).toBeNull();
    expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('User has not confirmed'));
    expect(inferenceService.infer).not.toHaveBeenCalled();
  });

  it('updates config when numCandidates changes', () => {
    // Technically this is just state update, but good to sanity check
    const { result } = renderHook(() => useInkModel('light', 'fp32', 'webgpu'));

    act(() => {
      result.current.setNumCandidates(3);
    });

    expect(result.current.numCandidates).toBe(3);
    // Note: numCandidates is passed to infer(), not to config state directly or init(). 
    // So checking state is enough.
  });
});

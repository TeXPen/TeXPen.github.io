// @vitest-environment jsdom
/* eslint-disable @typescript-eslint/no-explicit-any */
import { renderHook, act, waitFor } from '@testing-library/react';
import { useInkModel } from '../../hooks/useInkModel';
import { inferenceService } from '../../services/inference/InferenceService';
import { MODEL_CONFIG } from '../../services/inference/config';
import { describe, it, expect, vi, beforeEach } from 'vitest';

// Mock InferenceService
vi.mock('../../services/inference/InferenceService', () => ({
    inferenceService: {
        init: vi.fn(),
        infer: vi.fn(),
        dispose: vi.fn().mockResolvedValue(undefined),
    }
}));

// Mock utils/env
vi.mock('../../utils/env', () => ({
    isWebGPUAvailable: vi.fn().mockResolvedValue(false),
}));

// Mock caches
const mockCache = {
    keys: vi.fn().mockResolvedValue([]),
    open: vi.fn(),
    match: vi.fn(),
    put: vi.fn(),
    delete: vi.fn(),
};
global.caches = {
    open: vi.fn().mockResolvedValue(mockCache),
    has: vi.fn(),
    delete: vi.fn(),
    keys: vi.fn(),
    match: vi.fn(),
} as any;

// Mock services/downloader/DownloadManager
vi.mock('../../services/downloader/DownloadManager', () => ({
    downloadManager: {
        setQuotaErrorHandler: vi.fn(),
        downloadFile: vi.fn(),
    }
}));


describe('useInkModel', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        // Default to not cached
        mockCache.keys.mockResolvedValue([]);
    });

    it('initializes in idle state', async () => {
        const { result } = renderHook(() => useInkModel('light', MODEL_CONFIG.PROVIDERS.WASM, 'int8', 'int8', 'int8'));
        expect(result.current.status).toBe('idle');
        expect(result.current.isInitialized).toBe(false); // Initially false until cache check

        await waitFor(() => {
            expect(result.current.isInitialized).toBe(true);
        });
    });

    it('queues inference when model is loading', async () => {
        let resolveInit: (value: unknown) => void;
        (inferenceService.init as any).mockImplementation((callback: any) => {
            // Callback with "Loading model..." to trigger the status update logic
            callback('Loading model...', 0);
            return new Promise((resolve) => {
                resolveInit = resolve;
            });
        });

        const { result } = renderHook(() => useInkModel('light', MODEL_CONFIG.PROVIDERS.WASM, 'int8', 'int8', 'int8'));

        await waitFor(() => expect(result.current.isInitialized).toBe(true));

        // Start loading
        await act(async () => {
            result.current.setUserConfirmed(true);
        });

        // Verify we are loading
        await waitFor(() => {
            expect(result.current.loadingPhase).toContain('Downloading model');
        });

        const canvas = document.createElement('canvas');
        canvas.getContext('2d');
        canvas.toBlob = (cb) => cb(new Blob([''], { type: 'image/png' }));
        (inferenceService.infer as any).mockResolvedValue({
            latex: 'queue_test',
            candidates: ['queue_test'],
            debugImage: ''
        });

        // Call infer while loading
        let inferPromise: Promise<any>;
        act(() => {
            inferPromise = result.current.infer(canvas);
        });

        // Should be queued
        expect(inferenceService.infer).not.toHaveBeenCalled();
        expect(result.current.isGenerationQueued).toBe(true);

        // Finish loading
        await act(async () => {
            if (resolveInit) resolveInit(undefined);
        });

        // Now it should have processed the queue
        await waitFor(() => {
            expect(inferenceService.infer).toHaveBeenCalled();
        });

        const res = await inferPromise!;
        expect(res.latex).toBe('queue_test');
    });

    it('blocks inference when user has not confirmed and not cached', async () => {
        // Ensure cache is empty
        mockCache.keys.mockResolvedValue([]);

        const { result } = renderHook(() => useInkModel('light', MODEL_CONFIG.PROVIDERS.WASM, 'int8', 'int8', 'int8'));

        await waitFor(() => expect(result.current.isInitialized).toBe(true));

        expect(result.current.userConfirmed).toBe(false);
        expect(result.current.isLoadedFromCache).toBe(false);

        const canvas = document.createElement('canvas');

        // Attempt inference
        const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => { });
        const res = await result.current.infer(canvas);

        expect(res).toBeNull();
        expect(consoleSpy).toHaveBeenCalledWith('Inference skipped: User has not confirmed model download.');
        expect(inferenceService.infer).not.toHaveBeenCalled();
        consoleSpy.mockRestore();
    });

    it('allows inference when loaded from cache (auto confirmed)', async () => {
        // Mock cache hit with the expected Int8 files
        mockCache.keys.mockResolvedValue([
            { url: `https://huggingface.co/${MODEL_CONFIG.ID}/resolve/main/onnx/encoder_model_int8.onnx` },
            { url: `https://huggingface.co/${MODEL_CONFIG.ID}/resolve/main/onnx/decoder_model_merged_int8.onnx` }
        ]);

        // Mock init to resolve
        (inferenceService.init as any).mockImplementation(async () => {
            console.log('Mock init called');
            await new Promise(resolve => setTimeout(resolve, 50));
        });

        const { result } = renderHook(() => useInkModel('light', MODEL_CONFIG.PROVIDERS.WASM, 'int8', 'int8', 'int8'));

        await waitFor(() => {
            expect(result.current.isLoadedFromCache).toBe(true);
            expect(result.current.userConfirmed).toBe(true);
        });

        const canvas = document.createElement('canvas');
        canvas.getContext('2d'); // Ensure context exists
        // Mock toBlob
        canvas.toBlob = (cb) => cb(new Blob([''], { type: 'image/png' }));

        // Mock inference success
        (inferenceService.infer as any).mockResolvedValue({
            latex: 'x^2',
            candidates: ['x^2'],
            debugImage: ''
        });

        // Wait for init to occur
        await waitFor(() => {
            expect(inferenceService.init).toHaveBeenCalled();
        });

        // Wait for it to settle back to idle
        await waitFor(() => {
            expect(result.current.status).toBe('idle');
        });

        let res: any;
        await act(async () => {
            res = await result.current.infer(canvas);
        });

        await waitFor(() => {
            expect(inferenceService.infer).toHaveBeenCalled();
        });

        expect(res).toBeDefined();
        if (res) {
            expect(res.latex).toBe('x^2');
        }
    });
});

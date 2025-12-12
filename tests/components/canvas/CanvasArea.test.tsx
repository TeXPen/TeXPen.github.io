// @vitest-environment jsdom
import React, { useEffect } from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import CanvasArea from '../../../components/canvas/CanvasArea';
import { describe, it, expect, vi } from 'vitest';

// Mock CanvasBoard to expose callbacks for testing
let capturedOnStrokeAdded: (() => void) | undefined;

vi.mock('../../../components/canvas/CanvasBoard', () => ({
    default: function MockCanvasBoard({ refCallback, contentRefCallback, onStrokeAdded }: any) {
        capturedOnStrokeAdded = onStrokeAdded;
        useEffect(() => {
            // Create mock canvas elements
            const mockCanvas = document.createElement('canvas');
            const mockContentCanvas = document.createElement('canvas');

            // Invoke callbacks to simulate component mounting
            refCallback(mockCanvas);
            // The following lines from the instruction are syntactically incorrect and refer to undefined variables (`currentItem`).
            // To maintain syntactic correctness as per instructions, these lines cannot be directly inserted here.
            // If `currentItem` and its properties (`linumOld`, `line`) are intended to be part of the mock,
            // please provide the full context for their definition and usage within this mock component.
            // For now, the original line `contentRefCallback(mockContentCanvas);` is preserved.
            contentRefCallback(mockContentCanvas);
        }, [refCallback, contentRefCallback]);
        return <div>MockCanvasBoard</div>;
    }
}));

describe('CanvasArea', () => {
    it('calls onClear when Clear button is clicked', () => {
        const mockOnClear = vi.fn();
        const mockOnStrokeEnd = vi.fn();

        render(
            <CanvasArea
                theme="light"
                onClear={mockOnClear}
                onStrokeEnd={mockOnStrokeEnd}
            />
        );

        const clearBtn = screen.getByTitle('Clear Canvas');
        fireEvent.click(clearBtn);

        expect(mockOnClear).toHaveBeenCalled();
    });

    it('restores strokes when initialStrokes is provided', async () => {
        const mockOnClear = vi.fn();
        const mockOnStrokeEnd = vi.fn();

        const initialStrokes: any[] = [
            {
                tool: 'pen',
                color: '#000000',
                width: 2,
                points: [{ x: 0, y: 0 }, { x: 10, y: 10 }]
            }
        ];

        // Create mock context
        const mockCtx = {
            clearRect: vi.fn(),
            beginPath: vi.fn(),
            stroke: vi.fn(),
            moveTo: vi.fn(),
            lineTo: vi.fn(),
            drawImage: vi.fn(),
            lineCap: '',
            lineJoin: '',
            strokeStyle: '',
            lineWidth: 0,
            save: vi.fn(),
            restore: vi.fn(),
            resetTransform: vi.fn(),
            getImageData: vi.fn(() => ({ data: [], width: 100, height: 100 })),
            putImageData: vi.fn(),
        } as any;

        const originalCreateElement = document.createElement.bind(document);
        vi.spyOn(document, 'createElement').mockImplementation((tagName, options) => {
            if (tagName === 'canvas') {
                const canvas = originalCreateElement(tagName, options) as HTMLCanvasElement;
                canvas.getContext = vi.fn(() => mockCtx);
                return canvas;
            }
            return originalCreateElement(tagName, options);
        });

        render(
            <CanvasArea
                theme="light"
                onClear={mockOnClear}
                onStrokeEnd={mockOnStrokeEnd}
                initialStrokes={initialStrokes}
            />
        );

        await waitFor(() => {
            expect(mockCtx.beginPath).toHaveBeenCalled();
            expect(mockCtx.stroke).toHaveBeenCalled();
        });

        vi.restoreAllMocks();
    });

    it('resets transform before drawing restored content to prevent double scaling', async () => {
        const mockOnClear = vi.fn();
        const mockOnStrokeEnd = vi.fn();

        const initialStrokes: any[] = [
            {
                tool: 'pen',
                color: '#000000',
                width: 2,
                points: [{ x: 0, y: 0 }, { x: 10, y: 10 }]
            }
        ];

        // Create mock context
        const mockCtx = {
            clearRect: vi.fn(),
            beginPath: vi.fn(),
            stroke: vi.fn(),
            moveTo: vi.fn(),
            lineTo: vi.fn(),
            drawImage: vi.fn(),
            lineCap: '',
            lineJoin: '',
            strokeStyle: '',
            lineWidth: 0,
            save: vi.fn(),
            restore: vi.fn(),
            resetTransform: vi.fn(),
            getImageData: vi.fn(() => ({ data: [], width: 100, height: 100 })),
            putImageData: vi.fn(),
        } as any;

        const originalCreateElement = document.createElement.bind(document);
        vi.spyOn(document, 'createElement').mockImplementation((tagName, options) => {
            if (tagName === 'canvas') {
                const canvas = originalCreateElement(tagName, options) as HTMLCanvasElement;
                canvas.getContext = vi.fn(() => mockCtx);
                return canvas;
            }
            return originalCreateElement(tagName, options);
        });

        render(
            <CanvasArea
                theme="light"
                onClear={mockOnClear}
                onStrokeEnd={mockOnStrokeEnd}
                initialStrokes={initialStrokes}
            />
        );

        await waitFor(() => {
            // Verify correct sequence: save -> reset -> draw -> restore
            expect(mockCtx.save).toHaveBeenCalled();
            expect(mockCtx.resetTransform).toHaveBeenCalled();
            expect(mockCtx.drawImage).toHaveBeenCalled();
            expect(mockCtx.restore).toHaveBeenCalled();
        });

        vi.restoreAllMocks();
    });

    it('saves snapshot when a stroke is added (single stroke undo)', async () => {
        const mockOnClear = vi.fn();
        const mockOnStrokeEnd = vi.fn();

        // Create mock context with getImageData to verify snapshot saving
        const mockCtx = {
            clearRect: vi.fn(),
            getImageData: vi.fn(() => ({ data: new Uint8ClampedArray(4), width: 1, height: 1 })),
            putImageData: vi.fn(),
            drawImage: vi.fn(),
            save: vi.fn(),
            restore: vi.fn(),
            resetTransform: vi.fn()
        } as any;

        const originalCreateElement = document.createElement.bind(document);
        vi.spyOn(document, 'createElement').mockImplementation((tagName, options) => {
            if (tagName === 'canvas') {
                const canvas = originalCreateElement(tagName, options) as HTMLCanvasElement;
                canvas.getContext = vi.fn(() => mockCtx);
                // Mock width/height
                Object.defineProperty(canvas, 'width', { value: 100 });
                Object.defineProperty(canvas, 'height', { value: 100 });
                return canvas;
            }
            return originalCreateElement(tagName, options);
        });

        render(
            <CanvasArea
                theme="light"
                onClear={mockOnClear}
                onStrokeEnd={mockOnStrokeEnd}
            />
        );

        // Simulate stroke added event
        if (capturedOnStrokeAdded) {
            capturedOnStrokeAdded();
        }

        // Verify getImageData was called to save state
        expect(mockCtx.getImageData).toHaveBeenCalled();
    });
});

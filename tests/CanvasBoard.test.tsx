// @vitest-environment jsdom
import React from 'react';
import { render, fireEvent, act } from '@testing-library/react';
import CanvasBoard from '../components/canvas/CanvasBoard';
import { describe, it, expect, vi, beforeEach, afterEach, beforeAll } from 'vitest';
import { Stroke, ToolType } from '../types/canvas';

describe('CanvasBoard Selection & Move', () => {
    let mockCtx: any;

    let originalCreateElement: typeof document.createElement;

    beforeAll(() => {
        originalCreateElement = document.createElement.bind(document);
        global.ResizeObserver = class ResizeObserver {
            observe() { }
            unobserve() { }
            disconnect() { }
        };
    });

    afterEach(() => {
        vi.useRealTimers();
        vi.restoreAllMocks();
    });

    beforeEach(() => {
        vi.useFakeTimers();
        // Mock Canvas Context
        mockCtx = {
            clearRect: vi.fn(),
            beginPath: vi.fn(),
            moveTo: vi.fn(),
            lineTo: vi.fn(),
            stroke: vi.fn(),
            strokeRect: vi.fn(), // Key for verifying selection highlight
            fill: vi.fn(),
            fillRect: vi.fn(),
            save: vi.fn(),
            restore: vi.fn(),
            scale: vi.fn(),
            resetTransform: vi.fn(),
            setLineDash: vi.fn(),
            drawImage: vi.fn(),
            globalCompositeOperation: 'source-over',
            lineCap: 'round',
            lineJoin: 'round',
            lineWidth: 1,
            strokeStyle: '#000000',
        };

        vi.spyOn(document, 'createElement').mockImplementation((tagName, options) => {
            if (tagName === 'canvas') {
                // Use JSDOM canvas which has full DOM API (setAttribute, etc.) needed by React
                const canvas = document.implementation.createHTMLDocument().createElement('canvas');

                canvas.getContext = vi.fn((type) => {
                    if (type === '2d') return mockCtx;
                    return null;
                }) as any;

                // Mock getBoundingClientRect for coordinate calculations
                canvas.getBoundingClientRect = () => ({
                    left: 0,
                    top: 0,
                    width: 500,
                    height: 500,
                    right: 500,
                    bottom: 500,
                    x: 0,
                    y: 0,
                    toJSON: () => { }
                });
                return canvas;
            }
            return originalCreateElement(tagName, options);
        });
    });

    it('selects a stroke when clicked', () => {
        const strokesRef = { current: [] as Stroke[] };
        const onStrokeEnd = vi.fn();

        // 1. Initial Render with Pen
        const { rerender, container } = render(
            <CanvasBoard
                activeTool="pen"
                theme="light"
                onStrokeEnd={onStrokeEnd}
                refCallback={() => { }}
                contentRefCallback={() => { }}
                strokesRef={strokesRef}
            />
        );

        const canvas = container.querySelector('canvas')!;

        // 2. Draw a horizontal line from (10,10) to (50,10)
        fireEvent.mouseDown(canvas, { clientX: 10, clientY: 10 });
        fireEvent.mouseMove(canvas, { clientX: 50, clientY: 10 });
        fireEvent.mouseUp(canvas);

        // Wait for stroke to be saved (debounce 600ms in code)
        act(() => {
            vi.advanceTimersByTime(1000);
        });

        // Verify stroke was recorded
        expect(strokesRef.current.length).toBe(1);

        // 3. Switch to Select Tool
        rerender(
            <CanvasBoard
                activeTool="select"
                theme="light"
                onStrokeEnd={onStrokeEnd}
                refCallback={() => { }}
                contentRefCallback={() => { }}
                strokesRef={strokesRef}
            />
        );

        // Clear previous mock calls
        mockCtx.strokeRect.mockClear();

        // 4. Click on the stroke (around 30, 10)
        fireEvent.mouseDown(canvas, { clientX: 30, clientY: 10 });
        fireEvent.mouseUp(canvas);

        // Verify selection highlight (bounding box) was drawn
        // strokeRect is called inside redrawStrokes when selection exists
        expect(mockCtx.strokeRect).toHaveBeenCalled();
    });

    it('moves a selected stroke', () => {
        const strokesRef = { current: [] as Stroke[] };
        const onStrokeEnd = vi.fn();

        // 1. Pre-exist a stroke
        strokesRef.current = [{
            points: [{ x: 10, y: 10 }, { x: 50, y: 10 }],
            tool: 'pen',
            color: '#000000',
            width: 3
        }];

        const { container } = render(
            <CanvasBoard
                activeTool="select"
                theme="light"
                onStrokeEnd={onStrokeEnd}
                refCallback={() => { }}
                contentRefCallback={() => { }}
                strokesRef={strokesRef}
            />
        );

        const canvas = container.querySelector('canvas')!;

        // 2. Click to select (around 30,10)
        fireEvent.mouseDown(canvas, { clientX: 30, clientY: 10 });
        fireEvent.mouseUp(canvas);

        // 3. Start dragging (Click on selected stroke)
        fireEvent.mouseDown(canvas, { clientX: 30, clientY: 10 });

        // 4. Move +10px in X and Y
        fireEvent.mouseMove(canvas, { clientX: 40, clientY: 20 });

        // 5. Release
        fireEvent.mouseUp(canvas);

        // Verify points moved
        // Original: (10,10) -> (50,10)
        // Delta: +10, +10
        // Expected: (20,20) -> (60,20)

        const newPoints = strokesRef.current[0].points;
        expect(newPoints[0].x).toBe(20);
        expect(newPoints[0].y).toBe(20);
        expect(newPoints[1].x).toBe(60);
        expect(newPoints[1].y).toBe(20);
    });

    it('clears selection when clicking empty space', () => {
        const strokesRef = { current: [] as Stroke[] };

        // Pre-measure selection logic... easier to just test visual/state effect
        // But we can check internal state via behavior

        // Mock a stroke
        strokesRef.current = [{
            points: [{ x: 10, y: 10 }, { x: 20, y: 20 }],
            tool: 'pen',
            color: '#000',
            width: 3
        }];

        const { container } = render(
            <CanvasBoard
                activeTool="select"
                theme="light"
                onStrokeEnd={vi.fn()}
                refCallback={() => { }}
                contentRefCallback={() => { }}
                strokesRef={strokesRef}
            />
        );
        const canvas = container.querySelector('canvas')!;

        // Select it
        fireEvent.mouseDown(canvas, { clientX: 15, clientY: 15 });
        fireEvent.mouseUp(canvas);
        mockCtx.strokeRect.mockClear();

        // Click empty space (100, 100)
        fireEvent.mouseDown(canvas, { clientX: 100, clientY: 100 });
        fireEvent.mouseUp(canvas);

        // Should NOT draw selection box anymore
        // Note: Canvas redraws everything. If selection is cleared, strokeRect won't be called.
        // But redrawStrokes is called.
        // But redrawStrokes is called.
        // With box selection, clicking empty space starts a box (0x0 size)
        expect(mockCtx.strokeRect).toHaveBeenCalledWith(100, 100, 0, 0);
        // And importantly, it should NOT be called for the previously selected stroke (10,10)
        expect(mockCtx.strokeRect).not.toHaveBeenCalledWith(expect.objectContaining({
            x: expect.closeTo(10, 5) // Loose check for the stroke pos if we were checking args, 
            // but vitest check is usually exact args. 
            // Let's just check call count? 
            // If it was selected, it would be called twice (once for box, once for stroke).
        }));
        expect(mockCtx.strokeRect).toHaveBeenCalledTimes(1);
    });

    it('starts box selection when dragging on empty space', () => {
        const strokesRef = { current: [] as Stroke[] };
        const onStrokeEnd = vi.fn();

        const { container } = render(
            <CanvasBoard
                activeTool="select"
                theme="light"
                onStrokeEnd={onStrokeEnd}
                refCallback={() => { }}
                contentRefCallback={() => { }}
                strokesRef={strokesRef}
            />
        );
        const canvas = container.querySelector('canvas')!;

        // 1. Mouse down on empty space (100, 100)
        fireEvent.mouseDown(canvas, { clientX: 100, clientY: 100 });

        // 2. Drag to (200, 200)
        fireEvent.mouseMove(canvas, { clientX: 200, clientY: 200 });

        // Verify selection box drawing
        // Should draw fillRect and strokeRect for the box
        expect(mockCtx.fillRect).toHaveBeenCalled();
        expect(mockCtx.strokeRect).toHaveBeenCalled();
    });

    it('selects strokes within the box', () => {
        const strokesRef = { current: [] as Stroke[] };

        // Stroke at (50, 50) to (60, 60)
        strokesRef.current = [{
            points: [{ x: 50, y: 50 }, { x: 60, y: 60 }],
            tool: 'pen',
            color: '#000',
            width: 3
        }];

        const { container } = render(
            <CanvasBoard
                activeTool="select"
                theme="light"
                onStrokeEnd={vi.fn()}
                refCallback={() => { }}
                contentRefCallback={() => { }}
                strokesRef={strokesRef}
            />
        );
        const canvas = container.querySelector('canvas')!;

        mockCtx.strokeRect.mockClear();

        // 1. Box select from (40, 40) to (70, 70) - Encloses the stroke
        fireEvent.mouseDown(canvas, { clientX: 40, clientY: 40 });
        fireEvent.mouseMove(canvas, { clientX: 70, clientY: 70 });

        // 2. Release to finalize
        fireEvent.mouseUp(canvas);

        // Verify stroke is highlighted
        expect(mockCtx.strokeRect).toHaveBeenCalled();
    });

    it('allows dragging by clicking inside the bounding box', () => {
        const strokesRef = { current: [] as Stroke[] };

        // Pre-exist a stroke at (50, 50) to (60, 60)
        strokesRef.current = [{
            points: [{ x: 50, y: 50 }, { x: 60, y: 60 }],
            tool: 'pen',
            color: '#000',
            width: 3
        }];

        const { container } = render(
            <CanvasBoard
                activeTool="select"
                theme="light"
                onStrokeEnd={vi.fn()}
                refCallback={() => { }}
                contentRefCallback={() => { }}
                strokesRef={strokesRef}
            />
        );
        const canvas = container.querySelector('canvas')!;

        // 1. Click on stroke to select it
        fireEvent.mouseDown(canvas, { clientX: 55, clientY: 55 });
        fireEvent.mouseUp(canvas);

        // 2. Click inside bounding box but NOT on the stroke (near edge)
        // Bounding box is around (45, 45) to (65, 65) with 5px padding
        fireEvent.mouseDown(canvas, { clientX: 47, clientY: 47 });

        // 3. Drag to new position
        fireEvent.mouseMove(canvas, { clientX: 57, clientY: 57 });
        fireEvent.mouseUp(canvas);

        // Verify stroke moved by +10 in both X and Y
        const newPoints = strokesRef.current[0].points;
        expect(newPoints[0].x).toBe(60);
        expect(newPoints[0].y).toBe(60);
        expect(newPoints[1].x).toBe(70);
        expect(newPoints[1].y).toBe(70);
    });

    it('clears selection when switching away from select tool', () => {
        const strokesRef = { current: [] as Stroke[] };

        // Pre-exist a stroke
        strokesRef.current = [{
            points: [{ x: 50, y: 50 }, { x: 60, y: 60 }],
            tool: 'pen',
            color: '#000',
            width: 3
        }];

        const { container, rerender } = render(
            <CanvasBoard
                activeTool="select"
                theme="light"
                onStrokeEnd={vi.fn()}
                refCallback={() => { }}
                contentRefCallback={() => { }}
                strokesRef={strokesRef}
            />
        );
        const canvas = container.querySelector('canvas')!;

        // 1. Select the stroke
        fireEvent.mouseDown(canvas, { clientX: 55, clientY: 55 });
        fireEvent.mouseUp(canvas);

        // Verify selection highlight was drawn
        expect(mockCtx.strokeRect).toHaveBeenCalled();
        mockCtx.strokeRect.mockClear();
        mockCtx.setLineDash.mockClear();

        // 2. Switch to pen tool
        rerender(
            <CanvasBoard
                activeTool="pen"
                theme="light"
                onStrokeEnd={vi.fn()}
                refCallback={() => { }}
                contentRefCallback={() => { }}
                strokesRef={strokesRef}
            />
        );

        // 3. Verify selection highlight is NOT drawn (selection was cleared)
        // The setLineDash call is only made when drawing selection highlight
        // After clearing, strokeRect should not be called with the selection params
        expect(mockCtx.setLineDash).not.toHaveBeenCalledWith([5, 5]);
    });
});

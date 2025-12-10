import React, { useRef, useEffect, useState, useCallback } from 'react';
import { ToolType, Point, Stroke } from '../../types/canvas';
import { isPointNearStroke, isStrokeInRect, splitStrokes } from '../../utils/geometry';
import {
    drawAllStrokes,
    drawSelectionHighlight,
    drawSelectionBox,
    copyToCanvas
} from '../../utils/canvasRendering';

interface CanvasBoardProps {
    onStrokeEnd: () => void;
    refCallback: (ref: HTMLCanvasElement | null) => void;
    contentRefCallback: (ref: HTMLCanvasElement | null) => void;
    theme: 'dark' | 'light';
    activeTool: ToolType;
    strokesRef?: React.MutableRefObject<Stroke[]>;
}

const ERASER_SIZE = 20;

const CanvasBoard: React.FC<CanvasBoardProps> = ({ onStrokeEnd, refCallback, contentRefCallback, theme, activeTool, strokesRef: externalStrokesRef }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const [isDrawing, setIsDrawing] = useState(false);
    const [cursorPos, setCursorPos] = useState<{ x: number; y: number } | null>(null);
    const lastPos = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
    const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

    // Track strokes for line eraser
    const localStrokesRef = useRef<Stroke[]>([]);
    const strokesRef = externalStrokesRef || localStrokesRef;
    const currentStrokeRef = useRef<Point[]>([]);

    // Selection State
    const [selectedStrokeIndices, setSelectedStrokeIndices] = useState<number[]>([]);
    const dragStartPos = useRef<{ x: number; y: number } | null>(null);
    const isDragging = useRef<boolean>(false);

    // Box Selection State
    const selectionBoxRef = useRef<{ start: Point; current: Point } | null>(null);
    const isSelecting = useRef<boolean>(false);

    const contentCanvasRef = useRef<HTMLCanvasElement | null>(null);

    // Setup canvas size and style
    const setupCanvas = useCallback(() => {
        const canvas = canvasRef.current;
        const container = containerRef.current;
        if (!canvas || !container) return;

        const { width, height } = container.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const targetWidth = width * dpr;
        const targetHeight = height * dpr;

        // Initialize content canvas if it doesn't exist
        if (!contentCanvasRef.current) {
            contentCanvasRef.current = document.createElement('canvas');
            contentCanvasRef.current.width = targetWidth;
            contentCanvasRef.current.height = targetHeight;
            const contentCtx = contentCanvasRef.current.getContext('2d', { willReadFrequently: true });
            if (contentCtx) {
                contentCtx.drawImage(canvas, 0, 0);
            }
        }

        const contentCanvas = contentCanvasRef.current;

        // Check if resize is needed
        if (canvas.width !== targetWidth || canvas.height !== targetHeight) {

            // If the container is larger than our backing canvas, we need to resize it
            if (contentCanvas.width < targetWidth || contentCanvas.height < targetHeight) {
                const newContentCanvas = document.createElement('canvas');
                newContentCanvas.width = Math.max(contentCanvas.width, targetWidth);
                newContentCanvas.height = Math.max(contentCanvas.height, targetHeight);
                const newContentCtx = newContentCanvas.getContext('2d', { willReadFrequently: true });

                if (newContentCtx) {
                    newContentCtx.drawImage(contentCanvas, 0, 0);
                }
                contentCanvasRef.current = newContentCanvas;
            }

            // Resize the visible canvas
            canvas.width = targetWidth;
            canvas.height = targetHeight;
            canvas.style.width = '100%';
            canvas.style.height = '100%';
        }

        // Copy from backing canvas to visible canvas
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        if (ctx) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(contentCanvasRef.current, 0, 0);

            // Setup context properties for future drawing
            ctx.resetTransform();
            ctx.scale(dpr, dpr);
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            ctx.lineWidth = 3;
            ctx.strokeStyle = theme === 'dark' ? '#ffffff' : '#000000';
        }

        refCallback(canvas);
        contentRefCallback(contentCanvasRef.current);
    }, [refCallback, contentRefCallback, theme]);

    // Redraw all strokes
    const redrawStrokes = useCallback(() => {
        const canvas = contentCanvasRef.current; // Draw on the backing canvas
        if (!canvas) return;
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        if (!ctx) return;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw all strokes using utility
        drawAllStrokes(ctx, strokesRef.current, theme);

        // Draw unified selection highlight
        if (selectedStrokeIndices.length > 0) {
            drawSelectionHighlight(ctx, strokesRef.current, selectedStrokeIndices);
        }

        // Draw selection box (on top of everything)
        if (selectionBoxRef.current && isSelecting.current) {
            drawSelectionBox(ctx, selectionBoxRef.current.start, selectionBoxRef.current.current);
        }

        // Copy to visible canvas
        const visibleCanvas = canvasRef.current;
        if (visibleCanvas) {
            copyToCanvas(canvas, visibleCanvas);
        }
    }, [theme, selectedStrokeIndices]);

    useEffect(() => {
        redrawStrokes();
    }, [redrawStrokes]);

    // Handle Theme Changes: Recolors existing strokes
    useEffect(() => {
        const canvas = contentCanvasRef.current; // Modify the backing canvas
        if (!canvas) return;
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        if (!ctx) return;

        // We use composition to replace the color of existing non-transparent pixels
        ctx.save();
        const dpr = window.devicePixelRatio || 1;
        ctx.scale(dpr, dpr);
        ctx.globalCompositeOperation = 'source-in';
        ctx.fillStyle = theme === 'dark' ? '#ffffff' : '#000000';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.restore();

        // Update stroke style for future drawing
        ctx.strokeStyle = theme === 'dark' ? '#ffffff' : '#000000';

        // Copy to visible canvas
        const visibleCanvas = canvasRef.current;
        if (visibleCanvas) {
            const visibleCtx = visibleCanvas.getContext('2d', { willReadFrequently: true });
            if (visibleCtx) {
                visibleCtx.save();
                visibleCtx.resetTransform();
                visibleCtx.clearRect(0, 0, visibleCanvas.width, visibleCanvas.height);
                visibleCtx.drawImage(canvas, 0, 0);
                visibleCtx.restore();
            }
        }
    }, [theme]);

    useEffect(() => {
        setupCanvas();
        const handleResize = () => requestAnimationFrame(setupCanvas);
        window.addEventListener('resize', handleResize);

        const resizeObserver = new ResizeObserver(() => requestAnimationFrame(setupCanvas));
        if (containerRef.current) {
            resizeObserver.observe(containerRef.current);
        }

        return () => {
            window.removeEventListener('resize', handleResize);
            resizeObserver.disconnect();
        };
    }, [setupCanvas]);

    const getPos = (e: React.MouseEvent | React.TouchEvent) => {
        const canvas = canvasRef.current;
        if (!canvas) return { x: 0, y: 0 };
        const rect = canvas.getBoundingClientRect();

        let clientX, clientY;
        if ('touches' in e) {
            clientX = e.touches[0].clientX;
            clientY = e.touches[0].clientY;
        } else {
            clientX = (e as React.MouseEvent).clientX;
            clientY = (e as React.MouseEvent).clientY;
        }

        return {
            x: clientX - rect.left,
            y: clientY - rect.top
        };
    };



    const startDrawing = (e: React.MouseEvent | React.TouchEvent) => {
        setIsDrawing(true);
        const pos = getPos(e);
        const dpr = window.devicePixelRatio || 1;
        const scaledPos = { x: pos.x * dpr, y: pos.y * dpr };
        lastPos.current = scaledPos;

        if (activeTool === 'pen') {
            currentStrokeRef.current = [scaledPos];
        } else if (activeTool === 'select') {
            const clickedStrokeIndex = strokesRef.current.findIndex(stroke =>
                isPointNearStroke(scaledPos, stroke, 10 * dpr)
            );

            if (clickedStrokeIndex !== -1) {
                if (selectedStrokeIndices.includes(clickedStrokeIndex)) {
                    // Clicking on already selected stroke -> prepare drag
                    isDragging.current = true;
                    dragStartPos.current = scaledPos;
                } else {
                    // New selection
                    setSelectedStrokeIndices([clickedStrokeIndex]);
                    isDragging.current = true;
                    dragStartPos.current = scaledPos;
                }
            } else {
                // Clicked on empty space -> Start Selection Box
                isSelecting.current = true;
                selectionBoxRef.current = { start: scaledPos, current: scaledPos };
                // Also clear previous selection
                setSelectedStrokeIndices([]);
                isDragging.current = false;
                dragStartPos.current = null;
            }
        }

        if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };

    const draw = (e: React.MouseEvent | React.TouchEvent) => {
        const currentPos = getPos(e);
        setCursorPos(currentPos);

        if (!isDrawing) return;
        if ('touches' in e) e.preventDefault();

        const dpr = window.devicePixelRatio || 1;
        const scaledPos = { x: currentPos.x * dpr, y: currentPos.y * dpr };

        const canvas = contentCanvasRef.current; // Draw on the backing canvas
        const ctx = canvas?.getContext('2d', { willReadFrequently: true });
        if (!canvas || !ctx) return;

        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        if (activeTool === 'pen') {
            // Draw stroke
            ctx.beginPath();
            ctx.moveTo(lastPos.current.x, lastPos.current.y);
            ctx.lineTo(scaledPos.x, scaledPos.y);
            ctx.globalCompositeOperation = 'source-over';
            ctx.lineWidth = 3 * dpr;
            ctx.strokeStyle = theme === 'dark' ? '#ffffff' : '#000000';
            ctx.stroke();

            currentStrokeRef.current.push(scaledPos);

        } else if (activeTool === 'eraser-radial') {
            // Radial erase
            ctx.globalCompositeOperation = 'destination-out';
            ctx.lineWidth = ERASER_SIZE * dpr;
            ctx.beginPath();
            ctx.moveTo(lastPos.current.x, lastPos.current.y);
            ctx.lineTo(scaledPos.x, scaledPos.y);
            ctx.stroke();
            ctx.globalCompositeOperation = 'source-over';

            // Also remove from strokes data for consistency
            // Split strokes instead of deleting them entirely
            strokesRef.current = splitStrokes(strokesRef.current, scaledPos, (ERASER_SIZE / 2) * dpr);

        } else if (activeTool === 'eraser-line') {
            // Line erase - remove entire strokes
            const beforeCount = strokesRef.current.length;
            strokesRef.current = strokesRef.current.filter(stroke =>
                !isPointNearStroke(scaledPos, stroke, 10 * dpr)
            );

            if (strokesRef.current.length !== beforeCount) {
                redrawStrokes();
            }
        } else if (activeTool === 'select' && isDragging.current && dragStartPos.current) {
            // Move Logic
            const dx = scaledPos.x - lastPos.current.x;
            const dy = scaledPos.y - lastPos.current.y;

            selectedStrokeIndices.forEach(index => {
                const stroke = strokesRef.current[index];
                if (stroke) {
                    stroke.points = stroke.points.map(p => ({
                        x: p.x + dx,
                        y: p.y + dy,
                        pressure: p.pressure
                    }));
                }
            });
            redrawStrokes();
        } else if (activeTool === 'select' && isSelecting.current && selectionBoxRef.current) {
            // Box Selection Logic
            selectionBoxRef.current.current = scaledPos;

            // Real-time selection update
            const startStr = selectionBoxRef.current.start;
            const curStr = selectionBoxRef.current.current;

            const rect = {
                x: Math.min(startStr.x, curStr.x),
                y: Math.min(startStr.y, curStr.y),
                w: Math.abs(curStr.x - startStr.x),
                h: Math.abs(curStr.y - startStr.y)
            };

            const newSelectedIndices: number[] = [];
            strokesRef.current.forEach((stroke, index) => {
                if (isStrokeInRect(stroke, rect)) {
                    newSelectedIndices.push(index);
                }
            });

            setSelectedStrokeIndices(newSelectedIndices);
            redrawStrokes();
        }

        lastPos.current = scaledPos;

        // Copy to visible canvas
        const visibleCanvas = canvasRef.current;
        if (visibleCanvas) {
            const visibleCtx = visibleCanvas.getContext('2d', { willReadFrequently: true });
            if (visibleCtx) {
                visibleCtx.save();
                visibleCtx.resetTransform();
                visibleCtx.clearRect(0, 0, visibleCanvas.width, visibleCanvas.height);
                visibleCtx.drawImage(canvas, 0, 0);
                visibleCtx.restore();
            }
        }
    };

    const stopDrawing = () => {
        if (isDrawing) {
            setIsDrawing(false);
            isDragging.current = false;
            dragStartPos.current = null;

            if (isSelecting.current) {
                isSelecting.current = false;
                selectionBoxRef.current = null;
                redrawStrokes();
            }

            // Save stroke for line eraser
            if (activeTool === 'pen' && currentStrokeRef.current.length > 1) {
                strokesRef.current.push({
                    points: [...currentStrokeRef.current],
                    tool: 'pen',
                    color: theme === 'dark' ? '#ffffff' : '#000000',
                    width: 3
                });
            }
            currentStrokeRef.current = [];

            if (timeoutRef.current) clearTimeout(timeoutRef.current);
            timeoutRef.current = setTimeout(() => {
                onStrokeEnd();
            }, 600);
        }
    };

    const handleMouseMove = (e: React.MouseEvent) => {
        setCursorPos(getPos(e));
        draw(e);
    };

    const handleMouseLeave = () => {
        setCursorPos(null);
        stopDrawing();
    };

    const showEraserCursor = (activeTool === 'eraser-radial' || activeTool === 'eraser-line') && cursorPos;

    return (
        <div
            ref={containerRef}
            className="w-full h-full touch-none overflow-hidden transition-all duration-500 relative"
            style={{
                cursor: 'none',
                backgroundImage: `radial-gradient(${theme === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.06)'} 1px, transparent 1px)`,
                backgroundSize: '24px 24px',
                backgroundPosition: '0 0'
            }}
        >
            <canvas
                ref={canvasRef}
                className="block touch-none select-none"
                onMouseDown={startDrawing}
                onMouseMove={handleMouseMove}
                onMouseUp={stopDrawing}
                onMouseLeave={handleMouseLeave}
                onTouchStart={startDrawing}
                onTouchMove={draw}
                onTouchEnd={stopDrawing}
                onDragStart={(e) => e.preventDefault()}
                onContextMenu={(e) => e.preventDefault()}
            />

            {/* Custom cursor */}
            {cursorPos && (
                <div
                    className="pointer-events-none fixed z-50"
                    style={{
                        left: cursorPos.x + (containerRef.current?.getBoundingClientRect().left ?? 0),
                        top: cursorPos.y + (containerRef.current?.getBoundingClientRect().top ?? 0),
                        transform: 'translate(-50%, -50%)'
                    }}
                >
                    {showEraserCursor ? (
                        <div
                            className="rounded-full border-2"
                            style={{
                                width: activeTool === 'eraser-radial' ? ERASER_SIZE : 20,
                                height: activeTool === 'eraser-radial' ? ERASER_SIZE : 20,
                                borderColor: theme === 'dark' ? 'rgba(255,255,255,0.5)' : 'rgba(0,0,0,0.5)',
                                borderStyle: activeTool === 'eraser-line' ? 'dashed' : 'solid'
                            }}
                        />
                    ) : (
                        <div
                            className="rounded-full"
                            style={{
                                width: 6,
                                height: 6,
                                backgroundColor: theme === 'dark' ? '#fff' : '#000'
                            }}
                        />
                    )}
                </div>
            )}
        </div>
    );
};

export default CanvasBoard;
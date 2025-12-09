import React, { useRef, useEffect, useState, useCallback } from 'react';
import { ToolType, Point, Stroke } from '../../types/canvas';

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
        ctx.globalCompositeOperation = 'source-over';
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.lineWidth = 3;
        ctx.strokeStyle = theme === 'dark' ? '#ffffff' : '#000000';

        strokesRef.current.forEach(stroke => {
            if (stroke.points.length < 2) return;
            ctx.beginPath();
            ctx.moveTo(stroke.points[0].x, stroke.points[0].y);
            for (let i = 1; i < stroke.points.length; i++) {
                ctx.lineTo(stroke.points[i].x, stroke.points[i].y);
            }
            ctx.stroke();
        });

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

    // Check if a point is near a stroke segment
    const isPointNearStroke = (point: Point, stroke: Stroke, threshold: number): boolean => {
        for (let i = 0; i < stroke.points.length - 1; i++) {
            const a = stroke.points[i];
            const b = stroke.points[i + 1];

            const l2 = Math.pow(a.x - b.x, 2) + Math.pow(a.y - b.y, 2);
            if (l2 === 0) {
                if (Math.sqrt(Math.pow(point.x - a.x, 2) + Math.pow(point.y - a.y, 2)) < threshold) {
                    return true;
                }
                continue;
            }

            let t = ((point.x - a.x) * (b.x - a.x) + (point.y - a.y) * (b.y - a.y)) / l2;
            t = Math.max(0, Math.min(1, t));

            const proj = {
                x: a.x + t * (b.x - a.x),
                y: a.y + t * (b.y - a.y)
            };

            const dist = Math.sqrt(Math.pow(point.x - proj.x, 2) + Math.pow(point.y - proj.y, 2));
            if (dist < threshold) {
                return true;
            }
        }
        return false;
    };

    // Split strokes based on erasure point
    const splitStrokes = (strokes: Stroke[], erasePoint: Point, radius: number): Stroke[] => {
        const thresholdSq = radius * radius;
        const newStrokes: Stroke[] = [];

        strokes.forEach(stroke => {
            let currentPoints: Point[] = [];
            let modified = false;

            stroke.points.forEach((p, i) => {
                const distSq = Math.pow(p.x - erasePoint.x, 2) + Math.pow(p.y - erasePoint.y, 2);

                if (distSq > thresholdSq) {
                    currentPoints.push(p);
                } else {
                    modified = true;
                    // End current segment if we hit the eraser
                    if (currentPoints.length > 1) { // Filter single points
                        newStrokes.push({ ...stroke, points: [...currentPoints] });
                    }
                    currentPoints = [];
                }
            });

            // Push final segment
            if (currentPoints.length > 1) {
                newStrokes.push({ ...stroke, points: [...currentPoints] });
            } else if (!modified && currentPoints.length > 0) {
                // Determine if we kept the whole stroke (single point edge case or just no collision)
                // Actually if !modified, we should keep it.
                // But logic above empties currentPoints on collision.
                // If we are here and !modified, currentPoints == stroke.points
                // So checking currentPoints.length > 1 handles it mostly, 
                // but if stroke was 1 point? (Technically not a line, but generic handling)
                // Let's rely on the iteration.
            }
        });

        return newStrokes;
    };

    const startDrawing = (e: React.MouseEvent | React.TouchEvent) => {
        setIsDrawing(true);
        const pos = getPos(e);
        const dpr = window.devicePixelRatio || 1;
        const scaledPos = { x: pos.x * dpr, y: pos.y * dpr };
        lastPos.current = scaledPos;

        if (activeTool === 'pen') {
            currentStrokeRef.current = [scaledPos];
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
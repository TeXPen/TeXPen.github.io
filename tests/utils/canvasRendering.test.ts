import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  getSelectionBounds,
  drawStroke,
  drawAllStrokes,
  drawSelectionHighlight,
  drawSelectionBox,
  drawLassoPath,
} from '../../utils/canvasRendering';
import { Stroke, Point } from '../../types/canvas';

// Helper to create mock canvas context
function createMockContext(): CanvasRenderingContext2D {
  return {
    beginPath: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    closePath: vi.fn(),
    stroke: vi.fn(),
    fill: vi.fn(),
    fillRect: vi.fn(),
    strokeRect: vi.fn(),
    save: vi.fn(),
    restore: vi.fn(),
    setLineDash: vi.fn(),
    globalCompositeOperation: 'source-over',
    lineCap: 'round' as CanvasLineCap,
    lineJoin: 'round' as CanvasLineJoin,
    lineWidth: 1,
    strokeStyle: '#000000',
    fillStyle: '#000000',
  } as unknown as CanvasRenderingContext2D;
}

describe('Canvas Rendering Utilities', () => {
  let ctx: CanvasRenderingContext2D;

  beforeEach(() => {
    ctx = createMockContext();
  });

  describe('getSelectionBounds', () => {
    const strokes: Stroke[] = [
      { points: [{ x: 10, y: 10 }, { x: 20, y: 20 }], tool: 'pen', color: '#000', width: 3 },
      { points: [{ x: 30, y: 30 }, { x: 40, y: 40 }], tool: 'pen', color: '#000', width: 3 },
      { points: [{ x: 50, y: 50 }, { x: 60, y: 60 }], tool: 'pen', color: '#000', width: 3 },
    ];

    it('returns null for empty indices', () => {
      expect(getSelectionBounds(strokes, [])).toBeNull();
    });

    it('calculates bounds for single stroke', () => {
      const bounds = getSelectionBounds(strokes, [0]);
      expect(bounds).toEqual({ minX: 10, minY: 10, maxX: 20, maxY: 20 });
    });

    it('calculates bounds for multiple strokes', () => {
      const bounds = getSelectionBounds(strokes, [0, 2]);
      expect(bounds).toEqual({ minX: 10, minY: 10, maxX: 60, maxY: 60 });
    });

    it('handles invalid indices gracefully', () => {
      const bounds = getSelectionBounds(strokes, [0, 99]);
      expect(bounds).toEqual({ minX: 10, minY: 10, maxX: 20, maxY: 20 });
    });
  });

  describe('drawStroke', () => {
    it('does nothing for strokes with less than 2 points', () => {
      const stroke: Stroke = { points: [{ x: 0, y: 0 }], tool: 'pen', color: '#000', width: 3 };
      drawStroke(ctx, stroke, 'light');
      expect(ctx.beginPath).not.toHaveBeenCalled();
    });

    it('draws stroke path correctly', () => {
      const stroke: Stroke = {
        points: [{ x: 0, y: 0 }, { x: 10, y: 10 }, { x: 20, y: 0 }],
        tool: 'pen',
        color: '#ff0000',
        width: 5,
      };
      drawStroke(ctx, stroke, 'light');

      expect(ctx.beginPath).toHaveBeenCalled();
      expect(ctx.moveTo).toHaveBeenCalledWith(0, 0);
      expect(ctx.lineTo).toHaveBeenCalledWith(10, 10);
      expect(ctx.lineTo).toHaveBeenCalledWith(20, 0);
      expect(ctx.stroke).toHaveBeenCalled();
    });

    it('uses stroke color and width', () => {
      const stroke: Stroke = {
        points: [{ x: 0, y: 0 }, { x: 10, y: 10 }],
        tool: 'pen',
        color: '#ff0000',
        width: 5,
      };
      drawStroke(ctx, stroke, 'light');

      expect(ctx.strokeStyle).toBe('#ff0000');
      expect(ctx.lineWidth).toBe(5);
    });

    it('uses default color based on theme when not specified', () => {
      const stroke: Stroke = {
        points: [{ x: 0, y: 0 }, { x: 10, y: 10 }],
        tool: 'pen',
        color: undefined as any,
        width: undefined as any,
      };

      drawStroke(ctx, stroke, 'dark');
      expect(ctx.strokeStyle).toBe('#ffffff');

      drawStroke(ctx, stroke, 'light');
      expect(ctx.strokeStyle).toBe('#000000');
    });
  });

  describe('drawAllStrokes', () => {
    it('draws all strokes', () => {
      const strokes: Stroke[] = [
        { points: [{ x: 0, y: 0 }, { x: 10, y: 10 }], tool: 'pen', color: '#000', width: 3 },
        { points: [{ x: 20, y: 20 }, { x: 30, y: 30 }], tool: 'pen', color: '#000', width: 3 },
      ];
      drawAllStrokes(ctx, strokes, 'light');

      // beginPath is called for each stroke with 2+ points
      expect(ctx.beginPath).toHaveBeenCalledTimes(2);
    });
  });

  describe('drawSelectionHighlight', () => {
    const strokes: Stroke[] = [
      { points: [{ x: 10, y: 10 }, { x: 20, y: 20 }], tool: 'pen', color: '#000', width: 3 },
    ];

    it('draws dashed rectangle around selected strokes', () => {
      drawSelectionHighlight(ctx, strokes, [0]);

      expect(ctx.save).toHaveBeenCalled();
      expect(ctx.setLineDash).toHaveBeenCalledWith([5, 5]);
      expect(ctx.strokeRect).toHaveBeenCalled();
      expect(ctx.restore).toHaveBeenCalled();
    });

    it('does nothing for empty selection', () => {
      drawSelectionHighlight(ctx, strokes, []);
      expect(ctx.strokeRect).not.toHaveBeenCalled();
    });
  });

  describe('drawSelectionBox', () => {
    it('draws selection box with fill and stroke', () => {
      const start: Point = { x: 10, y: 10 };
      const current: Point = { x: 50, y: 50 };

      drawSelectionBox(ctx, start, current);

      expect(ctx.save).toHaveBeenCalled();
      expect(ctx.fillRect).toHaveBeenCalledWith(10, 10, 40, 40);
      expect(ctx.strokeRect).toHaveBeenCalledWith(10, 10, 40, 40);
      expect(ctx.restore).toHaveBeenCalled();
    });

    it('handles reversed coordinates (drag left/up)', () => {
      const start: Point = { x: 50, y: 50 };
      const current: Point = { x: 10, y: 10 };

      drawSelectionBox(ctx, start, current);

      // Box should start at min coords
      expect(ctx.fillRect).toHaveBeenCalledWith(10, 10, 40, 40);
    });
  });

  describe('drawLassoPath', () => {
    it('does nothing with less than 2 points', () => {
      drawLassoPath(ctx, [{ x: 0, y: 0 }]);
      expect(ctx.beginPath).not.toHaveBeenCalled();
    });

    it('draws closed polygon for lasso', () => {
      const points: Point[] = [
        { x: 0, y: 0 },
        { x: 10, y: 0 },
        { x: 10, y: 10 },
        { x: 0, y: 10 },
      ];

      drawLassoPath(ctx, points);

      expect(ctx.save).toHaveBeenCalled();
      expect(ctx.beginPath).toHaveBeenCalled();
      expect(ctx.moveTo).toHaveBeenCalledWith(0, 0);
      expect(ctx.lineTo).toHaveBeenCalledTimes(3);
      expect(ctx.closePath).toHaveBeenCalled();
      expect(ctx.fill).toHaveBeenCalled();
      expect(ctx.stroke).toHaveBeenCalled();
      expect(ctx.restore).toHaveBeenCalled();
    });
  });
});

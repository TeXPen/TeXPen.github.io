import { describe, it, expect } from 'vitest';
import {
  distance,
  isPointInBounds,
  doSegmentsIntersect,
  isPointNearSegment,
  isPointNearStroke,
  isStrokeInRect,
  isPointInPolygon,
  isStrokeInPolygon,
  splitStrokes,
} from '../../utils/geometry';
import { Point } from '../../types/canvas';

describe('Geometry Utilities', () => {
  describe('distance', () => {
    it('returns 0 for same point', () => {
      expect(distance({ x: 5, y: 5 }, { x: 5, y: 5 })).toBe(0);
    });

    it('calculates horizontal distance', () => {
      expect(distance({ x: 0, y: 0 }, { x: 10, y: 0 })).toBe(10);
    });

    it('calculates vertical distance', () => {
      expect(distance({ x: 0, y: 0 }, { x: 0, y: 10 })).toBe(10);
    });

    it('calculates diagonal distance (3-4-5 triangle)', () => {
      expect(distance({ x: 0, y: 0 }, { x: 3, y: 4 })).toBe(5);
    });
  });

  describe('isPointInBounds', () => {
    const bounds = { minX: 10, minY: 10, maxX: 50, maxY: 50 };

    it('returns true for point inside bounds', () => {
      expect(isPointInBounds({ x: 30, y: 30 }, bounds)).toBe(true);
    });

    it('returns true for point on boundary', () => {
      expect(isPointInBounds({ x: 10, y: 10 }, bounds)).toBe(true);
      expect(isPointInBounds({ x: 50, y: 50 }, bounds)).toBe(true);
    });

    it('returns false for point outside bounds', () => {
      expect(isPointInBounds({ x: 5, y: 30 }, bounds)).toBe(false);
      expect(isPointInBounds({ x: 60, y: 30 }, bounds)).toBe(false);
    });

    it('respects padding', () => {
      expect(isPointInBounds({ x: 5, y: 30 }, bounds, 10)).toBe(true);
    });
  });

  describe('doSegmentsIntersect', () => {
    it('returns true for crossing segments', () => {
      const p1 = { x: 0, y: 0 }, p2 = { x: 10, y: 10 };
      const p3 = { x: 0, y: 10 }, p4 = { x: 10, y: 0 };
      expect(doSegmentsIntersect(p1, p2, p3, p4)).toBe(true);
    });

    it('returns false for parallel segments', () => {
      const p1 = { x: 0, y: 0 }, p2 = { x: 10, y: 0 };
      const p3 = { x: 0, y: 5 }, p4 = { x: 10, y: 5 };
      expect(doSegmentsIntersect(p1, p2, p3, p4)).toBe(false);
    });

    it('returns false for non-intersecting segments', () => {
      const p1 = { x: 0, y: 0 }, p2 = { x: 5, y: 5 };
      const p3 = { x: 10, y: 10 }, p4 = { x: 20, y: 20 };
      expect(doSegmentsIntersect(p1, p2, p3, p4)).toBe(false);
    });
  });

  describe('isPointNearSegment', () => {
    it('returns true for point on segment', () => {
      const p = { x: 5, y: 5 };
      const a = { x: 0, y: 0 }, b = { x: 10, y: 10 };
      expect(isPointNearSegment(p, a, b, 1)).toBe(true);
    });

    it('returns true for point near segment', () => {
      const p = { x: 5, y: 6 }; // 1 unit away from diagonal
      const a = { x: 0, y: 0 }, b = { x: 10, y: 10 };
      expect(isPointNearSegment(p, a, b, 2)).toBe(true);
    });

    it('returns false for point far from segment', () => {
      const p = { x: 50, y: 50 };
      const a = { x: 0, y: 0 }, b = { x: 10, y: 10 };
      expect(isPointNearSegment(p, a, b, 5)).toBe(false);
    });

    it('handles zero-length segment (point)', () => {
      const p = { x: 5, y: 5 };
      const a = { x: 5, y: 5 }, b = { x: 5, y: 5 };
      expect(isPointNearSegment(p, a, b, 1)).toBe(true);
    });
  });

  describe('isPointNearStroke', () => {
    const stroke = {
      points: [
        { x: 0, y: 0 },
        { x: 10, y: 0 },
        { x: 10, y: 10 },
      ],
    };

    it('returns true for point near stroke segment', () => {
      expect(isPointNearStroke({ x: 5, y: 0 }, stroke, 5)).toBe(true);
    });

    it('returns false for point far from stroke', () => {
      expect(isPointNearStroke({ x: 50, y: 50 }, stroke, 5)).toBe(false);
    });

    it('returns false for single-point stroke', () => {
      const singlePoint = { points: [{ x: 0, y: 0 }] };
      expect(isPointNearStroke({ x: 0, y: 0 }, singlePoint, 5)).toBe(false);
    });
  });

  describe('isStrokeInRect', () => {
    const stroke = {
      points: [
        { x: 20, y: 20 },
        { x: 30, y: 30 },
      ],
    };

    it('returns true when stroke overlaps rect', () => {
      const rect = { x: 10, y: 10, w: 50, h: 50 };
      expect(isStrokeInRect(stroke, rect)).toBe(true);
    });

    it('returns false when stroke is outside rect', () => {
      const rect = { x: 100, y: 100, w: 50, h: 50 };
      expect(isStrokeInRect(stroke, rect)).toBe(false);
    });

    it('returns true when stroke partially overlaps', () => {
      const rect = { x: 25, y: 25, w: 10, h: 10 };
      expect(isStrokeInRect(stroke, rect)).toBe(true);
    });
  });

  describe('isPointInPolygon', () => {
    const square: Point[] = [
      { x: 0, y: 0 },
      { x: 10, y: 0 },
      { x: 10, y: 10 },
      { x: 0, y: 10 },
    ];

    it('returns true for point inside polygon', () => {
      expect(isPointInPolygon({ x: 5, y: 5 }, square)).toBe(true);
    });

    it('returns false for point outside polygon', () => {
      expect(isPointInPolygon({ x: 15, y: 5 }, square)).toBe(false);
    });

    it('returns false for polygon with less than 3 points', () => {
      expect(isPointInPolygon({ x: 5, y: 5 }, [{ x: 0, y: 0 }, { x: 10, y: 10 }])).toBe(false);
    });

    it('handles concave polygon', () => {
      const concave: Point[] = [
        { x: 0, y: 0 },
        { x: 10, y: 0 },
        { x: 10, y: 10 },
        { x: 5, y: 5 }, // Concave indent
        { x: 0, y: 10 },
      ];
      expect(isPointInPolygon({ x: 7, y: 7 }, concave)).toBe(true);
      expect(isPointInPolygon({ x: 3, y: 7 }, concave)).toBe(false);
    });
  });

  describe('isStrokeInPolygon', () => {
    const square: Point[] = [
      { x: 0, y: 0 },
      { x: 20, y: 0 },
      { x: 20, y: 20 },
      { x: 0, y: 20 },
    ];

    it('returns true when stroke point is inside polygon', () => {
      const stroke = { points: [{ x: 5, y: 5 }, { x: 10, y: 10 }] };
      expect(isStrokeInPolygon(stroke, square)).toBe(true);
    });

    it('returns true when stroke crosses polygon edge', () => {
      const stroke = { points: [{ x: -5, y: 10 }, { x: 25, y: 10 }] };
      expect(isStrokeInPolygon(stroke, square)).toBe(true);
    });

    it('returns false when stroke is completely outside', () => {
      const stroke = { points: [{ x: 30, y: 30 }, { x: 40, y: 40 }] };
      expect(isStrokeInPolygon(stroke, square)).toBe(false);
    });

    it('returns false for polygon with less than 3 points', () => {
      const stroke = { points: [{ x: 5, y: 5 }] };
      expect(isStrokeInPolygon(stroke, [{ x: 0, y: 0 }, { x: 10, y: 10 }])).toBe(false);
    });
  });

  describe('splitStrokes', () => {
    it('returns empty array when erasing entire stroke', () => {
      const stroke = {
        points: [{ x: 5, y: 5 }, { x: 6, y: 6 }],
        tool: 'pen' as const,
        color: '#000',
        width: 3,
      };
      const result = splitStrokes([stroke], { x: 5.5, y: 5.5 }, 10);
      expect(result.length).toBe(0);
    });

    it('splits stroke into two when erasing middle', () => {
      const stroke = {
        points: [
          { x: 0, y: 0 },
          { x: 10, y: 0 },
          { x: 20, y: 0 },
          { x: 30, y: 0 },
          { x: 40, y: 0 },
        ],
        tool: 'pen' as const,
        color: '#000',
        width: 3,
      };
      const result = splitStrokes([stroke], { x: 20, y: 0 }, 3);
      expect(result.length).toBe(2);
    });

    it('preserves stroke properties after split', () => {
      const stroke = {
        points: [
          { x: 0, y: 0 },
          { x: 10, y: 0 },
          { x: 20, y: 0 },
          { x: 30, y: 0 },
        ],
        tool: 'pen' as const,
        color: '#ff0000',
        width: 5,
      };
      const result = splitStrokes([stroke], { x: 15, y: 0 }, 3);
      result.forEach((s) => {
        expect(s.color).toBe('#ff0000');
        expect(s.width).toBe(5);
        expect(s.tool).toBe('pen');
      });
    });

    it('returns original stroke when erase point is far away', () => {
      const stroke = {
        points: [{ x: 0, y: 0 }, { x: 10, y: 0 }],
        tool: 'pen' as const,
        color: '#000',
        width: 3,
      };
      const result = splitStrokes([stroke], { x: 100, y: 100 }, 5);
      expect(result.length).toBe(1);
      expect(result[0].points).toEqual(stroke.points);
    });
  });
});

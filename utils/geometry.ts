import { Point } from '../types/canvas';

export function distance(p1: Point, p2: Point): number {
  return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
}

/**
 * Check if a point is inside a bounding box (with optional padding)
 */
export function isPointInBounds(
  point: Point,
  bounds: { minX: number; minY: number; maxX: number; maxY: number },
  padding: number = 0
): boolean {
  return (
    point.x >= bounds.minX - padding &&
    point.x <= bounds.maxX + padding &&
    point.y >= bounds.minY - padding &&
    point.y <= bounds.maxY + padding
  );
}

// Check if two line segments (p1-p2) and (p3-p4) intersect
export function doSegmentsIntersect(p1: Point, p2: Point, p3: Point, p4: Point): boolean {
  const ccw = (a: Point, b: Point, c: Point) => {
    return (c.y - a.y) * (b.x - a.x) > (b.y - a.y) * (c.x - a.x);
  };
  return (
    ccw(p1, p3, p4) !== ccw(p2, p3, p4) && ccw(p1, p2, p3) !== ccw(p1, p2, p4)
  );
}

// Check if a point is close to a line segment
export function isPointNearSegment(p: Point, a: Point, b: Point, threshold: number): boolean {
  const l2 = Math.pow(a.x - b.x, 2) + Math.pow(a.y - b.y, 2);
  if (l2 === 0) return distance(p, a) < threshold;

  let t = ((p.x - a.x) * (b.x - a.x) + (p.y - a.y) * (b.y - a.y)) / l2;
  t = Math.max(0, Math.min(1, t));

  const proj = {
    x: a.x + t * (b.x - a.x),
    y: a.y + t * (b.y - a.y)
  };

  return distance(p, proj) < threshold;
}

export function isPointNearStroke(point: Point, stroke: { points: Point[] }, threshold: number): boolean {
  for (let i = 0; i < stroke.points.length - 1; i++) {
    if (isPointNearSegment(point, stroke.points[i], stroke.points[i + 1], threshold)) {
      return true;
    }
  }
  return false;
}

export function isStrokeInRect(stroke: { points: Point[] }, rect: { x: number, y: number, w: number, h: number }): boolean {
  // Simple bounding box check first
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  stroke.points.forEach(p => {
    minX = Math.min(minX, p.x);
    minY = Math.min(minY, p.y);
    maxX = Math.max(maxX, p.x);
    maxY = Math.max(maxY, p.y);
  });

  // Check if bounding boxes overlap
  if (minX > rect.x + rect.w || maxX < rect.x || minY > rect.y + rect.h || maxY < rect.y) {
    return false;
  }
  return true;
}

export function splitStrokes<T extends { points: Point[] }>(strokes: T[], erasePoint: Point, radius: number): T[] {
  const thresholdSq = radius * radius;
  const newStrokes: T[] = [];

  strokes.forEach(stroke => {
    let currentPoints: Point[] = [];
    let modified = false;

    stroke.points.forEach((p) => {
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
      // If the stroke was not touched at all, keep it as is.
      // Note: The logic in original CanvasBoard was implicit.
      // If !modified, then currentPoints contains all points.
      // But if original stroke had <= 1 point (shouldn't happen for valid strokes), we might drop it.
      // It is safer to re-push the existing reference if not modified?
      // But strictly following the logic:
      // If !modified, currentPoints == stroke.points.
      // newStrokes.push({ ...stroke, points: [...currentPoints] }) works.
    }
  });

  return newStrokes;
}

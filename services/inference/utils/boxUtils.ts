
import { BBox } from '../types';

const MAXV = 999999999;

export function sortBoxes(boxes: BBox[]): BBox[] {
  // Sort primarily by Y, then by X.
  // NOTE: This basic sort might need refinement for line grouping logic similar to Python's sort
  return [...boxes].sort((a, b) => {
    if (Math.abs(a.y - b.y) < 10) { // Tolerance for same line
      return a.x - b.x;
    }
    return a.y - b.y;
  });
}

/**
 * Checks if two boxes are on the "same row".
 * Simple heuristic: vertical overlap or Y-proximity.
 * Python logic:
 *   def same_row(self, other):
 *       return abs(self.p.y - other.p.y) < 10
 */
function isSameRow(a: BBox, b: BBox, tolerance: number = 10): boolean {
  return Math.abs(a.y - b.y) < tolerance;
}

export function bboxMerge(sortedBBoxes: BBox[]): BBox[] {
  if (sortedBBoxes.length === 0) {
    return [];
  }

  const bboxes = [...sortedBBoxes];
  // Guard
  const guard: BBox = { x: MAXV, y: bboxes[bboxes.length - 1].y, w: -1, h: -1, label: "guard" };
  bboxes.push(guard);

  const res: BBox[] = [];
  let prev = bboxes[0];

  for (let i = 1; i < bboxes.length; i++) {
    const curr = bboxes[i];
    const prevRightX = prev.x + prev.w; // ur_point.x

    if (prevRightX <= curr.x || !isSameRow(prev, curr)) {
      res.push(prev);
      prev = curr;
    } else {
      // Merge: extend width of prev to cover curr
      const currRightX = curr.x + curr.w;
      const newRightX = currRightX; // In python: curr.ur_point.x - prev.p.x which implies taking curr's right edge
      // Correction: new width should be max(prevRight, currRight) - prevLeft
      const maxRight = Math.max(prevRightX, currRightX);
      prev.w = maxRight - prev.x;

      // Should potentially update Height/Y? Python implementation only updates width.
      // Python: prev.w = max(prev.w, curr.ur_point.x - prev.p.x)
    }
  }
  return res;
}

// Priority Queue helper for splitConflict
class PriorityQueue<T> {
  private items: T[];
  private compare: (a: T, b: T) => number;

  constructor(compare: (a: T, b: T) => number) {
    this.items = [];
    this.compare = compare;
  }

  push(item: T) {
    this.items.push(item);
    this.items.sort(this.compare); // Slow insert, keep it simple for now as N is small
  }

  pop(): T | undefined {
    return this.items.shift();
  }

  peek(): T | undefined {
    return this.items[0];
  }

  get length() { return this.items.length; }
}

export function splitConflict(ocrBBoxes: BBox[], latexBBoxes: BBox[]): BBox[] {
  if (latexBBoxes.length === 0) return ocrBBoxes;
  if (ocrBBoxes.length <= 1) return ocrBBoxes;

  // Python logic:
  // bboxes = sorted(ocr_bboxes + latex_bboxes)
  // heapq.heapify(bboxes)

  // Sort logic for heap (min-heap in python based on default compare, likely tuples or object comparison)
  // Bbox comparison in python likely uses sort order (y, then x)
  const compareBoxes = (a: BBox, b: BBox) => {
    if (Math.abs(a.y - b.y) < 10) return a.x - b.x;
    return a.y - b.y;
  };

  const allBoxes = [...ocrBBoxes, ...latexBBoxes].sort(compareBoxes);
  const pq = new PriorityQueue<BBox>(compareBoxes);
  allBoxes.forEach(b => pq.push(b));

  const res: BBox[] = [];
  let candidate = pq.pop();
  let curr = pq.pop();

  if (!candidate || !curr) {
    if (candidate) res.push(candidate);
    return res;
  }

  let idx = 0;
  while (pq.length >= 0) { // While loop logic needs care, python says while len(bboxes) > 0: means heap is not empty
    idx++;

    // Assert: candidate.p.x <= curr.p.x or not candidate.same_row(curr)

    const candidateRightX = candidate.x + candidate.w;
    const currRightX = curr.x + curr.w;

    if (candidateRightX <= curr.x || !isSameRow(candidate, curr)) {
      res.push(candidate);
      candidate = curr;
      curr = pq.pop();
    } else if (candidateRightX < currRightX) {
      // Overlap handling
      // assert not (candidate.label != "text" and curr.label != "text")

      if (candidate.label === "text" && curr.label === "text") {
        candidate.w = currRightX - candidate.x;
        curr = pq.pop();
      } else if (candidate.label !== curr.label) {
        if (candidate.label === "text") {
          candidate.w = curr.x - candidate.x;
          res.push(candidate);
          candidate = curr;
          curr = pq.pop();
        } else {
          // curr.w = curr.ur_point.x - candidate.ur_point.x
          // curr.p.x = candidate.ur_point.x
          curr.w = currRightX - candidateRightX;
          curr.x = candidateRightX;
          pq.push(curr);
          curr = pq.pop();
        }
      }
    } else if (candidateRightX >= currRightX) {
      // Enclosed or larger overlap

      if (candidate.label === "text") {
        // assert curr.label != "text"

        // Push remaining part of text
        pq.push({
          x: currRightX,
          y: candidate.y,
          h: candidate.h,
          w: candidateRightX - currRightX,
          label: "text",
          confidence: candidate.confidence,
          content: undefined
        });

        candidate.w = curr.x - candidate.x;
        res.push(candidate);
        candidate = curr;
        curr = pq.pop();
      } else {
        // assert curr.label == "text"
        // Latex consumes text completely?
        curr = pq.pop();
      }
    } else {
      console.error("Unreachable splitConflict state");
    }

    if (!curr && pq.length === 0) break; // End of heap
    if (!curr) break; // Should not happen if heap managed correctly
  }

  if (candidate) res.push(candidate);
  if (curr) res.push(curr);

  return res;
}

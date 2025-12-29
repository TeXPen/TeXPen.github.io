
import { describe, it, expect } from 'vitest';
import { HistoryItem } from '../types';
import { Stroke } from '../types/canvas';

// Copying the logic from useHistory.ts (stripped of React) to test the reducer logic
const addToHistoryReducer = (prev: HistoryItem[], item: HistoryItem): HistoryItem[] => {
  // Smart Session Logic:
  if (prev.length > 0 && prev[0].sessionId === item.sessionId) {
    const existingItem = prev[0];

    // Prevent duplicate versions if the output hasn't changed
    if (existingItem.latex === item.latex) {
      return prev;
    }

    const previousVersions = existingItem.versions || [];
    const newVersions = [...previousVersions];

    // If versions was empty, it means this was the first item. Add the initial state.
    if (newVersions.length === 0) {
      // Reconstruct the initial version from the existing item
      newVersions.push({ ...existingItem, versions: undefined });
    }

    // Add the NEW state as a version
    newVersions.push({ ...item, versions: undefined });

    // Update the main item to reflect the new state, and attach the full history
    const updatedItem = {
      ...item,
      versions: newVersions
    };

    const newHistory = [...prev];
    newHistory[0] = updatedItem;
    return newHistory;
  }

  // New Session: Create new item, initialize its version history with itself
  return [{ ...item, versions: [{ ...item, versions: undefined }] }, ...prev].slice(0, 20);
};

describe('History Logic', () => {
  it('should preserve strokes in versions', () => {
    const sessionId = 'session-1';

    const stroke1: Stroke[] = [{
      points: [{ x: 0, y: 0 }], tool: 'pen', color: 'black', width: 1
    }];

    const item1: HistoryItem = {
      id: '1',
      sessionId,
      latex: 'A',
      timestamp: 1000,
      source: 'draw',
      strokes: stroke1
    };

    // 1. Add first item
    let history = addToHistoryReducer([], item1);

    expect(history.length).toBe(1);
    expect(history[0].strokes).toEqual(stroke1);
    expect(history[0].versions?.length).toBe(1);
    expect(history[0].versions?.[0].strokes).toEqual(stroke1);

    // 2. Add second item (new strokes, new latex)
    const stroke2: Stroke[] = [{
      points: [{ x: 10, y: 10 }], tool: 'pen', color: 'black', width: 1
    }];

    const item2: HistoryItem = {
      id: '2',
      sessionId,
      latex: 'B',
      timestamp: 2000,
      source: 'draw',
      strokes: stroke2
    };

    history = addToHistoryReducer(history, item2);

    expect(history.length).toBe(1); // Still 1 item (grouped)
    expect(history[0].latex).toBe('B');
    expect(history[0].strokes).toEqual(stroke2);

    expect(history[0].versions?.length).toBe(2);

    // Check Version 1
    expect(history[0].versions?.[0].latex).toBe('A');
    expect(history[0].versions?.[0].strokes).toEqual(stroke1); // <--- Key check

    // Check Version 2
    expect(history[0].versions?.[1].latex).toBe('B');
    expect(history[0].versions?.[1].strokes).toEqual(stroke2);
  });

  it('should drop update if latex is same', () => {
    const sessionId = 'session-1';
    const stroke1 = [{ points: [], tool: 'pen', color: 'black', width: 1 } as Stroke];
    const item1: HistoryItem = { id: '1', sessionId, latex: 'A', timestamp: 1, source: 'draw', strokes: stroke1 };

    const history = addToHistoryReducer([], item1);

    const stroke2 = [{ points: [{ x: 1, y: 1 }], tool: 'pen', color: 'black', width: 1 } as Stroke];
    const item2: HistoryItem = { id: '2', sessionId, latex: 'A', timestamp: 2, source: 'draw', strokes: stroke2 };

    const history2 = addToHistoryReducer(history, item2);
    expect(history2).toBe(history); // check reference equality (returned prev)
  });
});

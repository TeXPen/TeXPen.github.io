// @vitest-environment jsdom
import { renderHook, act } from '@testing-library/react';
import { useHistory } from '../hooks/useHistory';
import { HistoryItem } from '../types';
import { describe, it, expect, vi, beforeEach } from 'vitest';

describe('useHistory', () => {
  beforeEach(() => {
    // Clear localStorage before each test
    window.localStorage.clear();
  });

  it('should not add duplicate history items in the same session', () => {
    const { result } = renderHook(() => useHistory());

    const item1: HistoryItem = {
      id: '1',
      sessionId: 'session-1',
      latex: 'x^2',
      timestamp: Date.now(),
      source: 'draw'
    };

    act(() => {
      result.current.addToHistory(item1);
    });

    expect(result.current.history).toHaveLength(1);
    expect(result.current.history[0].latex).toBe('x^2');

    // Add identical item (same session, same latex)
    const item2: HistoryItem = {
      ...item1,
      id: '2', // ID might differ but logic checks latex
      timestamp: Date.now() + 100
    };

    act(() => {
      result.current.addToHistory(item2);
    });

    // Should still be 1 item, and NO versions (since 2nd was duplicate)
    // Wait, if 2nd is duplicate, we return `prev` unchanged. 
    // So versions should be undefined or empty if it was the first item?
    // Let's check the logic:
    // If it's a new session -> add to list.
    // If same session -> check latex.
    // First item logic: addToHistory(item1) -> new session path -> returns [{...item, versions: [...]}]

    // So item1 in history WILL have versions initialized.
    expect(result.current.history).toHaveLength(1);
    expect(result.current.history[0].versions).toBeDefined();
    expect(result.current.history[0].versions).toHaveLength(1); // It initializes with itself

    // Now add duplicate
    act(() => {
      result.current.addToHistory(item2);
    });

    // Should NOT have changed
    expect(result.current.history).toHaveLength(1);
    expect(result.current.history[0].versions).toHaveLength(1);

    // Now add DIFFERENT item (same session)
    const item3: HistoryItem = {
      ...item1,
      id: '3',
      latex: 'x^3', // Different content
      timestamp: Date.now() + 200
    };

    act(() => {
      result.current.addToHistory(item3);
    });

    // Should still be 1 main item (updated), but now with more versions
    expect(result.current.history).toHaveLength(1);
    expect(result.current.history[0].latex).toBe('x^3');
    expect(result.current.history[0].versions!.length).toBeGreaterThan(1);
  });
});

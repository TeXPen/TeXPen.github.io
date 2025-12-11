// @vitest-environment jsdom
import { renderHook, act } from '@testing-library/react';
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { useHistory } from '../../hooks/useHistory';
import { HistoryItem } from '../../types';

describe('useHistory', () => {
  beforeEach(() => {
    // Clear localStorage
    window.localStorage.clear();
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('initializes with empty history if localStorage is empty', () => {
    const { result } = renderHook(() => useHistory());
    expect(result.current.history).toEqual([]);
  });

  it('initializes with data from localStorage', () => {
    const mockData: HistoryItem[] = [{
      id: '1',
      latex: 'test',
      timestamp: 123,
      sessionId: 'session1'
    }];
    // Use real JSDOM localStorage
    window.localStorage.setItem('texpen_history', JSON.stringify(mockData));

    const { result } = renderHook(() => useHistory());
    expect(result.current.history).toEqual(mockData);
  });

  it('adds a new item to history', () => {
    const { result } = renderHook(() => useHistory());
    const newItem: HistoryItem = {
      id: '1',
      latex: 'test',
      timestamp: 123,
      sessionId: 'session1'
    };

    act(() => {
      result.current.addToHistory(newItem);
    });

    expect(result.current.history).toHaveLength(1);
    expect(result.current.history[0].id).toBe('1');
    // Initial version history should be created
    expect(result.current.history[0].versions).toHaveLength(1);
  });

  it('updates existing item if sessionId matches (Smart Session Logic)', () => {
    const { result } = renderHook(() => useHistory());
    const item1: HistoryItem = {
      id: '1',
      latex: 'test1',
      timestamp: 100,
      sessionId: 'session1'
    };

    act(() => {
      result.current.addToHistory(item1);
    });

    // Add second item with SAME sessionId
    const item2: HistoryItem = {
      id: '2',
      latex: 'test2',
      timestamp: 200,
      sessionId: 'session1'
    };

    act(() => {
      result.current.addToHistory(item2);
    });

    expect(result.current.history).toHaveLength(1); // Should still be 1 item
    expect(result.current.history[0].latex).toBe('test2'); // Should update to new content
    expect(result.current.history[0].versions).toHaveLength(2); // Should have 2 versions
    expect(result.current.history[0].versions?.[0].latex).toBe('test1');
    expect(result.current.history[0].versions?.[1].latex).toBe('test2');
  });

  it('does not duplicate version if latex content is identical', () => {
    const { result } = renderHook(() => useHistory());
    const item1: HistoryItem = {
      id: '1',
      latex: 'test',
      timestamp: 100,
      sessionId: 'session1'
    };

    act(() => {
      result.current.addToHistory(item1);
    });

    // Add same content again
    const item2: HistoryItem = {
      id: '2',
      latex: 'test', // Identical
      timestamp: 200,
      sessionId: 'session1'
    };

    act(() => {
      result.current.addToHistory(item2);
    });

    expect(result.current.history).toHaveLength(1);
    expect(result.current.history[0].versions).toHaveLength(1); // Should not increase
  });

  it('creates new item if sessionId differs', () => {
    const { result } = renderHook(() => useHistory());
    const item1: HistoryItem = {
      id: '1',
      latex: 'test1',
      timestamp: 100,
      sessionId: 'session1'
    };

    act(() => {
      result.current.addToHistory(item1);
    });

    const item2: HistoryItem = {
      id: '2',
      latex: 'test2',
      timestamp: 200,
      sessionId: 'session2' // Different session
    };

    act(() => {
      result.current.addToHistory(item2);
    });

    expect(result.current.history).toHaveLength(2);
    expect(result.current.history[0].sessionId).toBe('session2'); // Mewsest first
    expect(result.current.history[1].sessionId).toBe('session1');
  });

  it('clears history', () => {
    const { result } = renderHook(() => useHistory());
    const item1: HistoryItem = {
      id: '1',
      latex: 'test1',
      timestamp: 100,
      sessionId: 'session1'
    };

    act(() => {
      result.current.addToHistory(item1);
    });
    expect(result.current.history).toHaveLength(1);

    act(() => {
      result.current.clearHistory();
    });
    expect(result.current.history).toHaveLength(0);
    expect(window.localStorage.getItem('texpen_history')).toBe('[]');
  });

  it('deletes specific history item', () => {
    const { result } = renderHook(() => useHistory());
    const item1: HistoryItem = { id: '1', latex: 'a', timestamp: 1, sessionId: 's1' };
    const item2: HistoryItem = { id: '2', latex: 'b', timestamp: 2, sessionId: 's2' };

    act(() => {
      result.current.addToHistory(item1);
      result.current.addToHistory(item2);
    });

    expect(result.current.history).toHaveLength(2);
    expect(result.current.history[0].id).toBe('2');

    act(() => {
      result.current.deleteHistoryItem('1');
    });

    expect(result.current.history).toHaveLength(1);
    expect(result.current.history[0].id).toBe('2');
  });

  it('persists changes to localStorage', () => {
    // Spy on native Storage prototype
    const setItemSpy = vi.spyOn(Storage.prototype, 'setItem');
    const { result } = renderHook(() => useHistory());
    const newItem: HistoryItem = {
      id: '1',
      latex: 'test',
      timestamp: 123,
      sessionId: 'session1'
    };

    act(() => {
      result.current.addToHistory(newItem);
    });

    expect(setItemSpy).toHaveBeenCalledWith('texpen_history', expect.stringContaining('test'));
  });
});

import { useState, useEffect } from 'react';
import { HistoryItem } from '../types';

export const useHistory = () => {
  const [history, setHistory] = useState<HistoryItem[]>(() => {
    if (typeof window === 'undefined') return [];
    try {
      const item = window.localStorage.getItem('texpen_history');
      return item ? JSON.parse(item) : [];
    } catch (error) {
      console.warn('Failed to load history from localStorage:', error);
      return [];
    }
  });

  useEffect(() => {
    try {
      window.localStorage.setItem('texpen_history', JSON.stringify(history));
    } catch (error) {
      console.warn('Failed to save history to localStorage:', error);
    }
  }, [history]);

  const addToHistory = (item: HistoryItem) => {
    setHistory(prev => {
      // Smart Session Logic:
      // If the new item belongs to the same session as the most recent item,
      // we update the recent item instead of creating a new one.
      if (prev.length > 0 && prev[0].sessionId === item.sessionId) {
        const existingItem = prev[0];

        // Prevent duplicate versions if the output hasn't changed
        if (existingItem.latex === item.latex) {
          return prev;
        }

        const previousVersions = existingItem.versions || [];
        // Add the *current* state of the item (before this update) to versions? 
        // Or just add the *new* item to versions?
        // Let's say versions should contain ALL steps including the current one.

        // If this is the FIRST update to an existing item, we need to make sure the *original* state is saved too.
        // But `addToHistory` is called *after* inference.
        // Ideally:
        // - First stroke: Item created. versions = [Item].
        // - Second stroke: Item updated. versions = [Item_v1, Item_v2].

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
    });
  };

  const clearHistory = () => {
    setHistory([]);
  };

  const deleteHistoryItem = (id: string) => {
    setHistory(prev => prev.filter(item => item.id !== id));
  };

  return {
    history,
    addToHistory,
    deleteHistoryItem,
    clearHistory,
  };
};
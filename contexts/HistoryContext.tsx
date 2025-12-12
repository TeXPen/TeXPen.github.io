import { createContext, useContext } from 'react';
import { HistoryItem } from '../types';


interface HistoryContextType {
    history: HistoryItem[];
    addToHistory: (item: HistoryItem) => void;
    deleteHistoryItem: (id: string) => void;
    clearHistory: () => void;
    filterMode: 'all' | 'current';
    setFilterMode: (mode: 'all' | 'current') => void;
}

export const HistoryContext = createContext<HistoryContextType | undefined>(undefined);



export const useHistoryContext = () => {
    const context = useContext(HistoryContext);
    if (!context) {
        throw new Error('useHistoryContext must be used within a HistoryProvider');
    }
    return context;
};

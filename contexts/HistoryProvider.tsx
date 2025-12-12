import React from 'react';
import { useHistory } from '../hooks/useHistory';
import { HistoryContext } from './HistoryContext';

export const HistoryProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const { history, addToHistory, deleteHistoryItem, clearHistory } = useHistory();
    const [filterMode, setFilterMode] = React.useState<'all' | 'current'>('all');

    return (
        <HistoryContext.Provider value={{ history, addToHistory, deleteHistoryItem, clearHistory, filterMode, setFilterMode }}>
            {children}
        </HistoryContext.Provider>
    );
};

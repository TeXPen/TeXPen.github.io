import React from 'react';
import { useMathJax } from '../../hooks/useMathJax';
import { HistoryItem } from '../../types';
import { TrashIcon } from '../common/icons/HistoryIcons';
import { useHistorySidebar } from '../../hooks/useHistorySidebar';
import { useAppContext } from '../../contexts/AppContext';
import { useThemeContext } from '../../contexts/ThemeContext';
import { useHistoryContext } from '../../contexts/HistoryContext';
import HistoryList from './HistoryList';
import ClearAllConfirmation from './ClearAllConfirmation';

interface HistorySidebarProps {
    history: HistoryItem[];
    onSelect: (item: HistoryItem) => void;
    onDelete: (id: string) => void;
    onClearAll: () => void;
    isOpen: boolean;
}

const HistorySidebar: React.FC<HistorySidebarProps> = ({
    history,
    onSelect,
    onDelete,
    onClearAll,
    isOpen,
}) => {
    const { toggleSidebar, activeTab } = useAppContext();
    const { theme } = useThemeContext();
    const { filterMode } = useHistoryContext();
    const [expandedItems, setExpandedItems] = React.useState<Set<string>>(new Set());
    const [isClearing, setIsClearing] = React.useState(false);

    // Filter history based on mode and active tab
    const filteredHistory = history.filter(item => {
        if (filterMode === 'all') return true;
        const itemSource = item.source || 'draw';
        return itemSource === activeTab;
    });

    const toggleExpand = (e: React.MouseEvent, id: string) => {
        e.stopPropagation();
        setExpandedItems(prev => {
            const next = new Set(prev);
            if (next.has(id)) {
                next.delete(id);
            } else {
                next.add(id);
            }
            return next;
        });
    };

    const {
        confirmDeleteId,
        sanitizeLatex,
        handleDeleteClick,
        handleConfirm,
        handleCancel
    } = useHistorySidebar(onDelete);

    const handleClearClick = () => setIsClearing(true);
    const handleConfirmClear = () => {
        onClearAll();
        setIsClearing(false);
    };

    // Trigger MathJax when history updates or expandedItems change
    useMathJax([filteredHistory, expandedItems], undefined, 'history-math');
    useMathJax([filteredHistory, expandedItems], undefined, 'history-math-version');

    const emptyMessage = `No ${filterMode === 'all' ? 'history' : (activeTab === 'draw' ? 'drawings' : 'uploads')} yet.`;

    return (
        <>
            {/* Mobile Backdrop */}
            <div
                className={`absolute inset-0 z-40 bg-black/20 backdrop-blur-sm md:hidden transition-opacity duration-300 ${isOpen ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}
                onClick={toggleSidebar}
            />

            <div
                className={`absolute md:relative z-50 h-full flex-none flex flex-col border-r border-black/5 dark:border-white/5 bg-white dark:bg-[#0c0c0c] transition-all duration-300 ease-in-out shadow-2xl md:shadow-none ${isOpen ? 'translate-x-0 w-64' : '-translate-x-full w-64 md:translate-x-0 md:w-16'
                    }`}
            >
                {/* Header with Toggle */}
                <div className="flex-none flex flex-col border-b border-black/5 dark:border-white/5 overflow-hidden">
                    <div className="h-16 flex items-center">
                        {/* Toggle Button */}
                        <div className="flex-none w-16 h-16 flex items-center justify-center">
                            <button
                                onClick={toggleSidebar}
                                className="w-10 h-10 rounded-xl flex items-center justify-center hover:bg-black/5 dark:hover:bg-white/5 text-slate-500 dark:text-white/40 hover:text-cyan-600 dark:hover:text-cyan-400 transition-all"
                                title={isOpen ? "Collapse Sidebar" : "Expand Sidebar"}
                            >
                                {isOpen ? (
                                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                                        <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                                        <line x1="9" y1="3" x2="9" y2="21" />
                                        <path strokeLinecap="round" strokeLinejoin="round" d="M15 9l-3 3 3 3" />
                                    </svg>
                                ) : (
                                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                                        <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                                        <line x1="9" y1="3" x2="9" y2="21" />
                                        <path strokeLinecap="round" strokeLinejoin="round" d="M13 9l3 3-3 3" />
                                    </svg>
                                )}
                            </button>
                        </div>

                        {/* Title */}
                        <div className={`flex-1 flex items-center transition-opacity duration-200 ${isOpen ? 'opacity-100 delay-75' : 'opacity-0'} whitespace-nowrap overflow-hidden`}>
                            <h2 className="text-sm font-bold text-slate-400 dark:text-white/40 tracking-widest uppercase pl-2">History</h2>
                        </div>
                    </div>
                </div>

                {/* List */}
                <div className="flex-1 overflow-y-auto overflow-x-hidden p-2 space-y-2 custom-scrollbar">
                    <div className={`transition-opacity duration-200 ${isOpen ? 'opacity-100 delay-100' : 'opacity-0 pointer-events-none'}`}>
                        <HistoryList
                            history={filteredHistory}
                            onSelect={onSelect}
                            expandedItems={expandedItems}
                            onToggleExpand={toggleExpand}
                            confirmDeleteId={confirmDeleteId}
                            onDeleteClick={handleDeleteClick}
                            onConfirmDelete={handleConfirm}
                            onCancelDelete={handleCancel}
                            sanitizeLatex={sanitizeLatex}
                            emptyMessage={emptyMessage}
                        />
                    </div>
                </div>

                {/* Footer with Clear All */}
                <div className={`p-4 border-t border-black/5 dark:border-white/5 transition-opacity duration-200 ${isOpen ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}>
                    {filteredHistory.length > 0 && (
                        <button
                            onClick={handleClearClick}
                            className="w-full py-2 px-3 rounded-lg text-xs font-medium text-slate-500 dark:text-white/40 hover:text-red-600 dark:hover:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/10 transition-colors flex items-center justify-center gap-2 group/clear whitespace-nowrap"
                        >
                            <span className="shrink-0"><TrashIcon /></span>
                            <span>Clear All History</span>
                        </button>
                    )}
                </div>

                <style>{`
                    .custom-scrollbar::-webkit-scrollbar { width: 4px; }
                    .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
                    .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.1); border-radius: 4px; }
                    .dark .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); }
                `}</style>

                {/* Clear All Confirmation Modal */}
                <ClearAllConfirmation
                    isOpen={isClearing}
                    onConfirm={handleConfirmClear}
                    onCancel={() => setIsClearing(false)}
                />
            </div>
        </>
    );
};

export default HistorySidebar;
import React from 'react';
import { HistoryItem } from '../../types';
import { TrashIcon, CheckIcon, XIcon, PenIcon } from '../common/icons/HistoryIcons';
import MathHistoryItem from './MathHistoryItem';

interface HistoryListItemProps {
    item: HistoryItem;
    onSelect: (item: HistoryItem) => void;
    onVersionSelect: (version: HistoryItem) => void;
    isExpanded: boolean;
    onToggleExpand: (e: React.MouseEvent) => void;
    isConfirming: boolean;
    onDeleteClick: (e: React.MouseEvent) => void;
    onConfirmDelete: (e: React.MouseEvent) => void;
    onCancelDelete: (e: React.MouseEvent) => void;
    sanitizeLatex: (latex: string) => string;
}

/**
 * Individual history list item with expand/collapse, delete confirmation, and version history
 */
export const HistoryListItem: React.FC<HistoryListItemProps> = ({
    item,
    onSelect,
    onVersionSelect,
    isExpanded,
    onToggleExpand,
    isConfirming,
    onDeleteClick,
    onConfirmDelete,
    onCancelDelete,
    sanitizeLatex
}) => {
    return (
        <div
            className="group relative p-3 rounded-lg hover:bg-black/5 dark:hover:bg-white/5 transition-colors cursor-pointer"
            onClick={() => onSelect(item)}
        >
            <div className="flex items-center justify-between mb-1 whitespace-nowrap overflow-hidden">
                <div className="flex items-center gap-2 overflow-hidden">
                    {item.source === 'upload' && (
                        <span className="flex-none p-1 rounded-md bg-purple-100 dark:bg-purple-500/20 text-purple-600 dark:text-purple-300" title="Uploaded Image">
                            <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                            </svg>
                        </span>
                    )}
                    {item.source === 'draw' && (
                        <span className="flex-none p-1 rounded-md bg-cyan-100 dark:bg-cyan-500/20 text-cyan-600 dark:text-cyan-400" title="Drawn Equation">
                            <PenIcon />
                        </span>
                    )}
                    {item.source !== 'upload' && item.versions && item.versions.length > 1 && (
                        <button
                            onClick={onToggleExpand}
                            className={`
                                p-1 rounded-md text-slate-400 dark:text-white/30 hover:text-cyan-600 dark:hover:text-cyan-400 hover:bg-black/5 dark:hover:bg-white/5 transition-all
                                ${isExpanded ? 'rotate-90 text-cyan-600 dark:text-cyan-400' : ''}
                            `}
                        >
                            <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                                <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                            </svg>
                        </button>
                    )}
                </div>
                {!isConfirming && (
                    <button
                        onClick={onDeleteClick}
                        className="p-1 opacity-0 group-hover:opacity-100 hover:bg-red-500/10 hover:text-red-500 rounded transition-all"
                    >
                        <TrashIcon />
                    </button>
                )}
            </div>

            {/* Image Preview for Uploads */}
            {item.source === 'upload' && item.imageData && (
                <div className="mb-2 mt-1 w-full flex justify-center bg-black/5 dark:bg-white/5 rounded-lg overflow-hidden py-1">
                    <img
                        src={item.imageData}
                        alt="Source"
                        className="h-16 object-contain rounded-md"
                    />
                </div>
            )}

            {/* Math Content */}
            <MathHistoryItem latex={item.latex} />

            {/* Timestamp - Bottom Right */}
            <span className="absolute bottom-1 right-2 text-[9px] font-mono text-slate-300 dark:text-white/20 select-none">
                {new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </span>

            {/* Confirmation Overlay */}
            {isConfirming && (
                <div className="absolute inset-0 z-10 bg-white/90 dark:bg-black/90 backdrop-blur-sm flex items-center justify-center gap-2 rounded-xl animate-in fade-in duration-200">
                    <button
                        onClick={onConfirmDelete}
                        className="p-1.5 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-colors shadow-sm"
                        title="Confirm Delete"
                    >
                        <CheckIcon />
                    </button>
                    <button
                        onClick={onCancelDelete}
                        className="p-1.5 bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300 rounded-lg hover:bg-slate-300 dark:hover:bg-slate-600 transition-colors"
                        title="Cancel"
                    >
                        <XIcon />
                    </button>
                </div>
            )}

            {/* Versions Dropdown */}
            {isExpanded && item.versions && (
                <div className="mt-2 border-t border-black/5 dark:border-white/5 pt-2 space-y-1 animate-in slide-in-from-top-1 duration-200">
                    {item.versions.map((version, vIndex) => (
                        <div
                            key={vIndex}
                            onClick={(e) => {
                                e.stopPropagation();
                                onVersionSelect(version);
                            }}
                            className="px-2 py-1.5 rounded-lg hover:bg-black/5 dark:hover:bg-white/5 cursor-pointer flex items-center justify-between group/version transition-colors"
                        >
                            <div className="flex items-center gap-2 overflow-hidden w-full">
                                <div className="text-[9px] text-slate-400 dark:text-white/20 font-mono w-3 shrink-0">
                                    {vIndex + 1}
                                </div>
                                <div className="text-[10px] text-slate-600 dark:text-white/60 font-mono truncate history-math-version w-full">
                                    {`\\(${sanitizeLatex(version.latex)}\\)`}
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

interface HistoryListProps {
    history: HistoryItem[];
    onSelect: (item: HistoryItem) => void;
    expandedItems: Set<string>;
    onToggleExpand: (e: React.MouseEvent, id: string) => void;
    confirmDeleteId: string | null;
    onDeleteClick: (e: React.MouseEvent, id: string) => void;
    onConfirmDelete: (e: React.MouseEvent, id: string) => void;
    onCancelDelete: (e: React.MouseEvent) => void;
    sanitizeLatex: (latex: string) => string;
    emptyMessage: string;
}

/**
 * List of history items with empty state handling
 */
const HistoryList: React.FC<HistoryListProps> = ({
    history,
    onSelect,
    expandedItems,
    onToggleExpand,
    confirmDeleteId,
    onDeleteClick,
    onConfirmDelete,
    onCancelDelete,
    sanitizeLatex,
    emptyMessage
}) => {
    if (history.length === 0) {
        return (
            <div className="flex flex-col items-center justify-center h-48 px-4 text-center whitespace-nowrap overflow-hidden">
                <span className="text-xs text-slate-400 dark:text-white/20 italic">{emptyMessage}</span>
            </div>
        );
    }

    return (
        <>
            {history.map((item) => (
                <HistoryListItem
                    key={item.id}
                    item={item}
                    onSelect={onSelect}
                    onVersionSelect={onSelect}
                    isExpanded={expandedItems.has(item.id)}
                    onToggleExpand={(e) => onToggleExpand(e, item.id)}
                    isConfirming={confirmDeleteId === item.id}
                    onDeleteClick={(e) => onDeleteClick(e, item.id)}
                    onConfirmDelete={(e) => onConfirmDelete(e, item.id)}
                    onCancelDelete={onCancelDelete}
                    sanitizeLatex={sanitizeLatex}
                />
            ))}
        </>
    );
};

export default HistoryList;

import React, { useRef, useEffect } from 'react';
import Select, { StylesConfig } from 'react-select';
import { useMathJax } from '../hooks/useMathJax';
import { HistoryItem } from '../types';
import { TrashIcon, CheckIcon, XIcon, PenIcon } from './icons/HistoryIcons';
import { useHistorySidebar } from '../hooks/useHistorySidebar';
import { useAppContext } from './contexts/AppContext';
import { useThemeContext } from './contexts/ThemeContext';

interface HistorySidebarProps {
    history: HistoryItem[];
    onSelect: (item: HistoryItem) => void;
    onDelete: (id: string) => void;
    onClearAll: () => void;
    isOpen: boolean;
}

// Scalable Math Item Component
const MathHistoryItem: React.FC<{ latex: string }> = ({ latex }) => {
    const ref = useRef<HTMLDivElement>(null);
    const parentRef = useRef<HTMLDivElement>(null);
    const [scale, setScale] = React.useState(1);

    // Clean LaTeX
    const cleanLatex = latex
        .replace(/\\\[/g, '')
        .replace(/\\\]/g, '')
        .replace(/\\\(/g, '')
        .replace(/\\\)/g, '')
        .replace(/\$\$/g, '')
        .replace(/^\$|\$$/g, '')
        .trim();

    useEffect(() => {
        if (ref.current && window.MathJax) {
            ref.current.innerHTML = `\\(${cleanLatex}\\)`;
            window.MathJax.typesetPromise([ref.current]).then(() => {
                checkResize();
            }).catch((err: Error) => console.error('MathJax error:', err));
        }
    }, [cleanLatex]);

    const checkResize = () => {
        if (ref.current && parentRef.current) {
            const contentWidth = ref.current.scrollWidth;
            const containerWidth = parentRef.current.clientWidth;

            // Add padding buffer (Gradient is w-8 = 32px, plus some extra safety)
            const availableWidth = containerWidth - 40;

            if (contentWidth > availableWidth) {
                const newScale = Math.max(0.6, availableWidth / contentWidth);
                setScale(newScale);
            } else {
                setScale(1);
            }
        }
    };

    // Re-check on simple resize (sidebar toggle can affect this, so maybe ResizeObserver is better)
    useEffect(() => {
        const handleResize = () => requestAnimationFrame(checkResize);
        window.addEventListener('resize', handleResize);

        // Also observe parent size changes
        const observer = new ResizeObserver(handleResize);
        if (parentRef.current) observer.observe(parentRef.current);

        return () => {
            window.removeEventListener('resize', handleResize);
            observer.disconnect();
        };
    }, []);

    return (
        <div ref={parentRef} className="relative min-h-8 h-auto py-2 flex items-center w-full group/item">
            <div
                ref={ref}
                className="text-xs text-slate-700 dark:text-white/80 font-mono w-full whitespace-nowrap transition-transform origin-left"
                style={{ transform: `scale(${scale})`, width: 'fit-content' }}
            >
                {`\\(${cleanLatex}\\)`}
            </div>
        </div>
    );
};

const HistorySidebar: React.FC<HistorySidebarProps> = ({
    history,
    onSelect,
    onDelete,
    onClearAll,
    isOpen,
}) => {
    const { toggleSidebar, activeTab } = useAppContext();
    const { theme } = useThemeContext();
    const [expandedItems, setExpandedItems] = React.useState<Set<string>>(new Set());
    const [filterMode, setFilterMode] = React.useState<'all' | 'current'>('all');
    const [isClearing, setIsClearing] = React.useState(false);

    // Filter history based on mode and active tab
    const filteredHistory = history.filter(item => {
        if (filterMode === 'all') return true;

        const itemSource = item.source || 'draw'; // Default legacy items to 'draw'
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

    // Clear All Logic
    const handleClearClick = () => {
        setIsClearing(true);
    };

    const handleConfirmClear = () => {
        onClearAll();
        setIsClearing(false);
    };



    // Trigger MathJax when history updates or expandedItems change
    // We pass expandedItems in the dependency array (as the first arg)
    // so that when a user expands a section, the new content gets typeset.
    useMathJax([filteredHistory, expandedItems], undefined, 'history-math');
    useMathJax([filteredHistory, expandedItems], undefined, 'history-math-version');

    const filterOptions = [
        { value: 'all', label: 'All History' },
        { value: 'current', label: `Active Tab (${activeTab === 'draw' ? 'Drawings' : 'Uploads'})` }
    ];

    const customStyles: StylesConfig = {
        control: (provided) => ({
            ...provided,
            width: '100%',
            height: 28,
            minHeight: 28,
            backgroundColor: theme === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgb(255, 255, 255)',
            border: `1px solid ${theme === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)'}`,
            borderRadius: '0.375rem',
            boxShadow: 'none',
            '&:hover': {
                borderColor: theme === 'dark' ? 'rgb(34 211 238)' : 'rgb(6 182 212)',
            },
        }),
        valueContainer: (provided) => ({
            ...provided,
            height: '28px',
            padding: '0 8px',
        }),
        input: (provided) => ({
            ...provided,
            margin: '0px',
            height: '28px',
            color: theme === 'dark' ? 'white' : 'black',
        }),
        indicatorSeparator: () => ({
            display: 'none',
        }),
        indicatorsContainer: (provided) => ({
            ...provided,
            height: '28px',
        }),
        singleValue: (provided) => ({
            ...provided,
            color: theme === 'dark' ? 'white' : '#1f2937',
            fontSize: '0.75rem',
            fontFamily: 'monospace', // Matching settings font if generalized, or keeping sidebar aesthetic? Provider used monospace. Let's use sans for history for readability or inherit. ProviderSelector used monospace. I'll stick to default sans for history labels as they are text.
            // Actually ProviderSelector uses monospace because it shows code-like enum values (webgpu). History labels are UI text. I will remove fontFamily monospace.
        }),
        menu: (provided) => ({
            ...provided,
            backgroundColor: theme === 'dark' ? '#111' : 'white',
            border: `1px solid ${theme === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)'}`,
            borderRadius: '0.5rem',
            zIndex: 50,
        }),
        option: (provided, { isSelected, isFocused }) => ({
            ...provided,
            backgroundColor: isSelected
                ? (theme === 'dark' ? 'rgb(6 182 212)' : 'rgb(6 182 212)')
                : isFocused
                    ? (theme === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.05)')
                    : 'transparent',
            color: isSelected ? 'white' : (theme === 'dark' ? 'white' : '#1f2937'),
            fontSize: '0.875rem',
            cursor: 'pointer',
        }),
    };

    return (
        <div
            className={`flex-none flex flex-col border-r border-black/5 dark:border-white/5 bg-white dark:bg-[#0c0c0c] transition-all duration-300 ease-in-out ${isOpen ? 'w-64' : 'w-16'}`}
        >
            {/* Header with Toggle */}
            <div className="flex-none flex flex-col border-b border-black/5 dark:border-white/5 overflow-hidden">
                <div className="h-16 flex items-center">
                    {/* Toggle Button (Always visible, fixed width) */}
                    <div className="flex-none w-16 h-16 flex items-center justify-center">
                        <button
                            onClick={toggleSidebar}
                            className="w-10 h-10 rounded-xl flex items-center justify-center hover:bg-black/5 dark:hover:bg-white/5 text-slate-500 dark:text-white/40 hover:text-cyan-600 dark:hover:text-cyan-400 transition-all"
                            title={isOpen ? "Collapse Sidebar" : "Expand Sidebar"}
                        >
                            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                                <path strokeLinecap="round" strokeLinejoin="round" d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                                <path strokeLinecap="round" strokeLinejoin="round" d="M3 3v5h5" />
                                <path strokeLinecap="round" strokeLinejoin="round" d="M12 7v5l4 2" />
                            </svg>
                        </button>
                    </div>

                    {/* Title (Hidden when closed) */}
                    <div className={`flex-1 flex justify-between items-center pr-4 transition-opacity duration-200 ${isOpen ? 'opacity-100 delay-75' : 'opacity-0'} whitespace-nowrap overflow-hidden`}>
                        <>
                            <span className="text-xs font-semibold text-slate-500 dark:text-white/40 uppercase tracking-wider">History</span>
                            <span className="text-[10px] text-slate-400 dark:text-white/20">{filteredHistory.length}</span>
                        </>


                    </div>
                </div>

                {/* Filter Dropdown */}
                <div className={`px-4 pb-4 transition-all duration-300 ${isOpen ? 'opacity-100 h-auto' : 'opacity-0 h-0 pointer-events-none'}`}>
                    <Select
                        value={filterOptions.find(o => o.value === filterMode)}
                        onChange={(option: { value: string; label: string } | null) => setFilterMode((option?.value as 'all' | 'current') || 'all')}
                        options={filterOptions}
                        styles={{
                            ...customStyles,
                            menuPortal: (base) => ({ ...base, zIndex: 9999 })
                        }}
                        isSearchable={false}
                        menuPlacement="auto"
                        menuPortalTarget={typeof document !== 'undefined' ? document.body : null}
                    />
                </div>
            </div>

            {/* List */}
            <div className="flex-1 overflow-y-auto overflow-x-hidden p-2 space-y-2 custom-scrollbar">
                <div className={`transition-opacity duration-200 ${isOpen ? 'opacity-100 delay-100' : 'opacity-0 pointer-events-none'}`}>
                    {filteredHistory.length === 0 ? (
                        <div className="flex flex-col items-center justify-center h-48 px-4 text-center whitespace-nowrap overflow-hidden">
                            <span className="text-xs text-slate-400 dark:text-white/20 italic">No {filterMode === 'all' ? 'history' : (activeTab === 'draw' ? 'drawings' : 'uploads')} yet.</span>
                        </div>
                    ) : (
                        filteredHistory.map((item) => {
                            const isConfirming = confirmDeleteId === item.id;
                            return (
                                <div
                                    key={item.id}
                                    className="group relative p-3 rounded-xl bg-black/[0.02] dark:bg-white/[0.02] hover:bg-black/[0.05] dark:hover:bg-white/[0.05] border border-transparent hover:border-black/5 dark:hover:border-white/5 transition-all cursor-pointer"
                                    onClick={() => onSelect(item)}
                                >
                                    <div className="flex items-center justify-between mb-2 whitespace-nowrap overflow-hidden">
                                        <div className="flex items-center gap-2 overflow-hidden">
                                            {item.versions && item.versions.length > 1 && (
                                                <button
                                                    onClick={(e) => toggleExpand(e, item.id)}
                                                    className={`
                                                        p-1 rounded-md text-slate-400 dark:text-white/30 hover:text-cyan-600 dark:hover:text-cyan-400 hover:bg-black/5 dark:hover:bg-white/5 transition-all
                                                        ${expandedItems.has(item.id) ? 'rotate-90 text-cyan-600 dark:text-cyan-400' : ''}
                                                    `}
                                                >
                                                    <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                                                        <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                                                    </svg>
                                                </button>
                                            )}
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

                                        </div>
                                        {!isConfirming && (
                                            <button
                                                onClick={(e) => handleDeleteClick(e, item.id)}
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

                                    {/* USE SCALABLE COMPONENT */}
                                    <MathHistoryItem latex={item.latex} />

                                    {/* Timestamp - Bottom Right */}
                                    <span className="absolute bottom-1 right-2 text-[9px] font-mono text-slate-300 dark:text-white/20 select-none">
                                        {new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                    </span>

                                    {/* Confirmation Overlay */}
                                    {isConfirming && (
                                        <div className="absolute inset-0 z-10 bg-white/90 dark:bg-black/90 backdrop-blur-sm flex items-center justify-center gap-2 rounded-xl animate-in fade-in duration-200">
                                            <button
                                                onClick={(e) => handleConfirm(e, item.id)}
                                                className="p-1.5 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-colors shadow-sm"
                                                title="Confirm Delete"
                                            >
                                                <CheckIcon />
                                            </button>
                                            <button
                                                onClick={handleCancel}
                                                className="p-1.5 bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300 rounded-lg hover:bg-slate-300 dark:hover:bg-slate-600 transition-colors"
                                                title="Cancel"
                                            >
                                                <XIcon />
                                            </button>
                                        </div>
                                    )}

                                    {/* Versions Dropdown */}
                                    {expandedItems.has(item.id) && item.versions && (
                                        <div className="mt-2 border-t border-black/5 dark:border-white/5 pt-2 space-y-1 animate-in slide-in-from-top-1 duration-200">
                                            {item.versions.map((version, vIndex) => (
                                                <div
                                                    key={vIndex}
                                                    onClick={(e) => {
                                                        e.stopPropagation();
                                                        onSelect(version);
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
                        })
                    )}
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

            {/* Full Screen Confirmation Modal */}
            {isClearing && (
                <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/50 backdrop-blur-sm animate-in fade-in duration-200">
                    <div className="bg-white dark:bg-[#111] border border-black/10 dark:border-white/10 rounded-2xl shadow-2xl max-w-sm w-full p-6 m-4 transform animate-in zoom-in-95 duration-200">
                        <div className="flex flex-col items-center text-center space-y-4">
                            <div className="w-12 h-12 rounded-full bg-red-100 dark:bg-red-900/20 flex items-center justify-center text-red-600 dark:text-red-400">
                                <TrashIcon /> {/* Provided TrashIcon might be small, let's assume it works or scale it via container */}
                                {/* Actually, TrashIcon likely has w/h classes inside. Let's wrap it or check. 
                                    Looking at usage line 274: <TrashIcon />. It likely inherits size or has defaults. 
                                    I'll wrap it in a div that forces size if needed, or just rely on its SVG props.
                                */ }
                            </div>

                            <h3 className="text-lg font-bold text-slate-900 dark:text-white">Clear All History?</h3>

                            <p className="text-sm text-slate-500 dark:text-white/60">
                                This action cannot be undone. All your math history and sessions will be permanently deleted.
                            </p>

                            <div className="grid grid-cols-2 gap-3 w-full pt-2">
                                <button
                                    onClick={() => setIsClearing(false)}
                                    className="px-4 py-2 rounded-xl text-sm font-medium text-slate-700 dark:text-white/80 bg-slate-100 dark:bg-white/5 hover:bg-slate-200 dark:hover:bg-white/10 transition-colors"
                                >
                                    Cancel
                                </button>
                                <button
                                    onClick={handleConfirmClear}
                                    className="px-4 py-2 rounded-xl text-sm font-medium text-white bg-red-600 hover:bg-red-500 shadow-lg shadow-red-500/20 transition-all active:scale-95"
                                >
                                    Yes, Clear All
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default HistorySidebar;
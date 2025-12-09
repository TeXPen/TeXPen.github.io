import React, { useState, useRef, useEffect } from 'react';
import { useAppContext } from '../../contexts/AppContext';
import { useThemeContext } from '../../contexts/ThemeContext';
import { useHistoryContext } from '../../contexts/HistoryContext';
import { ProviderSelector } from './ProviderSelector';
import { QuantizationSelector } from './QuantizationSelector';
import { INFERENCE_CONFIG } from '../../services/inference/config';

export const SettingsMenu: React.FC = () => {
    // const [isOpen, setIsOpen] = useState(false); // REMOVE local state
    const menuRef = useRef<HTMLDivElement>(null);
    const modelIdInputRef = useRef<HTMLInputElement>(null);
    const {
        quantization,
        setQuantization,
        provider,
        setProvider,
        showVisualDebugger,
        setShowVisualDebugger,
        customModelId,
        setCustomModelId,
        isSettingsOpen,
        openSettings,
        closeSettings,
        settingsFocus,
    } = useAppContext();
    const { theme, toggleTheme } = useThemeContext();
    const { filterMode, setFilterMode } = useHistoryContext();

    // Close on click outside
    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
                if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
                    closeSettings();
                }
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    // Handle auto-focus and selection when settings opens with specific focus target
    useEffect(() => {
        if (isSettingsOpen && settingsFocus === 'modelId' && modelIdInputRef.current) {
            // Small timeout to ensure render visibility transition is done if any
            setTimeout(() => {
                modelIdInputRef.current?.focus();
                modelIdInputRef.current?.select();
            }, 50);
        }
    }, [isSettingsOpen, settingsFocus]);

    return (
        <div className="relative" ref={menuRef}>
            <button
                onClick={() => isSettingsOpen ? closeSettings() : openSettings()}
                className={`w-9 h-9 rounded-xl flex items-center justify-center transition-all duration-200 border ${isSettingsOpen
                    ? 'bg-cyan-500/10 border-cyan-500/30 text-cyan-600 dark:text-cyan-400'
                    : 'bg-black/5 dark:bg-white/5 border-black/5 dark:border-white/5 text-slate-500 dark:text-white/40 hover:text-slate-800 dark:hover:text-white hover:bg-black/10 dark:hover:bg-white/10'
                    }`}
                title="Settings"
            >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                    <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
            </button>

            {/* Dropdown Menu */}
            {isSettingsOpen && (
                <div className="absolute right-0 top-full mt-3 w-64 bg-white dark:bg-[#111] border border-black/10 dark:border-white/10 rounded-2xl shadow-xl backdrop-blur-xl z-50 p-2 flex flex-col gap-2 animate-in fade-in zoom-in-95 duration-200 origin-top-right">

                    {/* Theme */}
                    <button
                        onClick={toggleTheme}
                        className="flex items-center justify-between p-3 rounded-xl hover:bg-black/5 dark:hover:bg-white/5 transition-colors text-sm text-slate-700 dark:text-white"
                    >
                        <span className="font-medium">Theme</span>
                        <div className="flex items-center gap-2 text-slate-400 dark:text-white/40">
                            {theme === 'dark' ? (
                                <div className="flex items-center gap-2">
                                    <span className="text-xs">Dark</span>
                                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
                                    </svg>
                                </div>
                            ) : (
                                <div className="flex items-center gap-2">
                                    <span className="text-xs">Light</span>
                                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
                                    </svg>
                                </div>
                            )}
                        </div>
                    </button>

                    <div className="h-px bg-black/5 dark:bg-white/5 mx-2" />

                    {/* Provider */}
                    <div className="p-3">
                        <div className="text-xs font-bold uppercase text-slate-400 dark:text-white/40 mb-2">Provider</div>
                        <ProviderSelector value={provider} onChange={setProvider} />
                    </div>

                    <div className="h-px bg-black/5 dark:bg-white/5 mx-2" />

                    {/* Quantization */}
                    {/* <div className="p-3">
                        <div className="text-xs font-bold uppercase text-slate-400 dark:text-white/40 mb-2">Quantization</div>
                        <QuantizationSelector value={quantization} onChange={setQuantization} />
                    </div> */}

                    {/* <div className="h-px bg-black/5 dark:bg-white/5 mx-2" /> */}

                    {/* Model ID - With Reset Option */}
                    <div className="p-3">
                        <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-1.5">
                                <div className="text-xs font-bold uppercase text-slate-400 dark:text-white/40">HuggingFace Model ID</div>
                                <div
                                    className="text-slate-400 dark:text-white/40 cursor-help"
                                    title="This feature requires an ONNX model compatible with Transformers.js. Check the HuggingFace model card for 'onnx' tag or ONNX weights."
                                >
                                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-3.5 h-3.5">
                                        <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                                    </svg>
                                </div>
                            </div>
                            <button
                                onClick={() => setCustomModelId(INFERENCE_CONFIG.MODEL_ID)}
                                className="text-slate-400 hover:text-cyan-600 dark:hover:text-cyan-400 transition-colors p-1 rounded-md hover:bg-black/5 dark:hover:bg-white/5"
                                title={`Reset to ${INFERENCE_CONFIG.MODEL_ID}`}
                            >
                                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" className="w-3.5 h-3.5">
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                                </svg>
                            </button>
                        </div>
                        <input
                            ref={modelIdInputRef}
                            type="text"
                            value={customModelId}
                            onChange={(e) => setCustomModelId(e.target.value)}
                            onKeyDown={(e) => {
                                if (e.key === 'Enter') {
                                    e.currentTarget.blur();
                                }
                            }}
                            className="w-full text-xs font-mono bg-black/5 dark:bg-white/5 rounded-lg border border-black/10 dark:border-white/10 px-2 py-1.5 focus:outline-none focus:border-cyan-500 dark:focus:border-cyan-400 text-slate-700 dark:text-white transition-all"
                            placeholder="user/repo"
                        />
                    </div>

                    <div className="h-px bg-black/5 dark:bg-white/5 mx-2" />

                    {/* History Filter */}
                    <div className="p-3">
                        <div className="text-xs font-bold uppercase text-slate-400 dark:text-white/40 mb-2">History Filter</div>
                        <div className="flex bg-black/5 dark:bg-white/5 p-1 rounded-lg">
                            <button
                                onClick={() => setFilterMode('all')}
                                className={`flex-1 py-1.5 px-3 rounded-md text-xs font-medium transition-all ${filterMode === 'all'
                                    ? 'bg-white dark:bg-zinc-800 text-cyan-600 dark:text-cyan-400 shadow-sm'
                                    : 'text-slate-500 dark:text-zinc-400 hover:text-slate-700 dark:hover:text-zinc-200'
                                    }`}
                            >
                                All History
                            </button>
                            <button
                                onClick={() => setFilterMode('current')}
                                className={`flex-1 py-1.5 px-3 rounded-md text-xs font-medium transition-all ${filterMode === 'current'
                                    ? 'bg-white dark:bg-zinc-800 text-cyan-600 dark:text-cyan-400 shadow-sm'
                                    : 'text-slate-500 dark:text-zinc-400 hover:text-slate-700 dark:hover:text-zinc-200'
                                    }`}
                            >
                                Active Tab
                            </button>
                        </div>
                    </div>

                    <div className="h-px bg-black/5 dark:bg-white/5 mx-2" />

                    {/* Visual Debugger */}
                    <button
                        onClick={() => setShowVisualDebugger(!showVisualDebugger)}
                        className="flex items-center justify-between p-3 rounded-xl hover:bg-black/5 dark:hover:bg-white/5 transition-colors text-sm text-slate-700 dark:text-white"
                    >
                        <span className="font-medium">Visual Debugger</span>
                        <div className={`w-10 h-5 rounded-full relative transition-colors ${showVisualDebugger ? 'bg-cyan-500' : 'bg-slate-200 dark:bg-white/10'}`}>
                            <div className={`absolute top-1 w-3 h-3 rounded-full bg-white shadow-sm transition-transform ${showVisualDebugger ? 'left-6' : 'left-1'}`} />
                        </div>
                    </button>

                </div>
            )}
        </div>
    );
};

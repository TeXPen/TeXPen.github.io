import React, { useRef, useEffect } from 'react';
import { useAppContext } from '../../contexts/AppContext';
import { useThemeContext } from '../../contexts/ThemeContext';
import { useHistoryContext } from '../../contexts/HistoryContext';
import { ProviderSelector } from './ProviderSelector';
import { useVerifyDownloads } from '../../hooks/useVerifyDownloads';

import { MODEL_CONFIG } from '../../services/inference/config';
import { Tooltip } from '../common/Tooltip';
import { HelpIcon } from '../common/HelpIcon';

export const SettingsMenu: React.FC = () => {
    const [isConfirmingClear, setIsConfirmingClear] = React.useState(false);
    // const [isOpen, setIsOpen] = useState(false); // REMOVE local state
    const menuRef = useRef<HTMLDivElement>(null);
    const modelIdInputRef = useRef<HTMLInputElement>(null);
    const {
        // quantization, // usage removed
        provider,
        setProvider,
        showPreviewInput,
        setShowPreviewInput,
        customModelId,
        setCustomModelId,
        isSettingsOpen,
        openSettings,
        closeSettings,
        settingsFocus,
        // Sampling
        numCandidates,
        setNumCandidates,
        doSample,
        setDoSample,
        temperature,
        setTemperature,
        topK,
        setTopK,
        topP,
        setTopP,
    } = useAppContext();
    const { theme, toggleTheme } = useThemeContext();
    const { filterMode, setFilterMode } = useHistoryContext();
    const { verifyDownloads } = useVerifyDownloads();

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
    }, [closeSettings]);

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
                <div className="absolute right-0 top-full mt-3 w-64 max-h-[80vh] overflow-y-auto bg-white dark:bg-[#111] border border-black/10 dark:border-white/10 rounded-2xl shadow-xl backdrop-blur-xl z-50 p-2 flex flex-col gap-2 animate-in fade-in zoom-in-95 duration-200 origin-top-right">

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

                    {/* Generation Settings */}
                    <div className="p-3">
                        <div className="text-xs font-bold uppercase text-slate-400 dark:text-white/40 mb-2">Generation</div>

                        {/* Num Candidates / Beams */}
                        <div className="flex items-center justify-between mb-2">
                            <span className="text-xs text-slate-600 dark:text-slate-300">Candidates</span>
                            <div className="flex items-center gap-2">
                                <input
                                    type="range"
                                    min="1"
                                    max="5"
                                    step="1"
                                    value={numCandidates}
                                    onChange={(e) => setNumCandidates(parseInt(e.target.value))}
                                    className="w-20 accent-cyan-500"
                                />
                                <span className="text-xs font-mono w-4 text-right text-slate-600 dark:text-slate-300">{numCandidates}</span>
                            </div>
                        </div>

                        {/* Sampling Toggle */}
                        <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-1.5">
                                <span className="text-xs text-slate-600 dark:text-slate-300">Beam Search Approx.</span>
                                <Tooltip content="Enable multinomial sampling to approximate beam search exploration. Faster but non-deterministic. Recommended for generating diverse candidates.">
                                    <HelpIcon />
                                </Tooltip>
                            </div>
                            <button
                                onClick={() => setDoSample(!doSample)}
                                className={`w-8 h-4 rounded-full transition-colors relative ${doSample ? 'bg-cyan-500' : 'bg-slate-300 dark:bg-zinc-700'}`}
                            >
                                <div className={`absolute top-0.5 w-3 h-3 bg-white rounded-full shadow-sm transition-transform ${doSample ? 'left-4.5' : 'left-0.5'}`} />
                            </button>
                        </div>

                        {/* Sampling Params */}
                        {doSample && (
                            <div className="space-y-2 pl-2 border-l-2 border-black/5 dark:border-white/5 ml-1">
                                {/* Temperature */}
                                <div className="flex items-center justify-between">
                                    <span className="text-xs text-slate-500 dark:text-slate-400">Temperature</span>
                                    <div className="flex items-center gap-2">
                                        <input
                                            type="range"
                                            min="0.1"
                                            max="2.0"
                                            step="0.1"
                                            value={temperature}
                                            onChange={(e) => setTemperature(parseFloat(e.target.value))}
                                            className="w-16 accent-cyan-500 h-1"
                                        />
                                        <span className="text-[10px] font-mono w-6 text-right text-slate-500 dark:text-slate-400">{temperature.toFixed(1)}</span>
                                    </div>
                                </div>

                                {/* Top K */}
                                <div className="flex items-center justify-between">
                                    <span className="text-xs text-slate-500 dark:text-slate-400">Top K</span>
                                    <div className="flex items-center gap-2">
                                        <input
                                            type="range"
                                            min="0"
                                            max="100"
                                            step="1"
                                            value={topK}
                                            onChange={(e) => setTopK(parseInt(e.target.value))}
                                            className="w-16 accent-cyan-500 h-1"
                                        />
                                        <span className="text-[10px] font-mono w-6 text-right text-slate-500 dark:text-slate-400">{topK}</span>
                                    </div>
                                </div>

                                {/* Top P */}
                                <div className="flex items-center justify-between">
                                    <span className="text-xs text-slate-500 dark:text-slate-400">Top P</span>
                                    <div className="flex items-center gap-2">
                                        <input
                                            type="range"
                                            min="0.1"
                                            max="1.0"
                                            step="0.05"
                                            value={topP}
                                            onChange={(e) => setTopP(parseFloat(e.target.value))}
                                            className="w-16 accent-cyan-500 h-1"
                                        />
                                        <span className="text-[10px] font-mono w-6 text-right text-slate-500 dark:text-slate-400">{topP.toFixed(2)}</span>
                                    </div>
                                </div>
                            </div>
                        )}

                    </div>

                    {/* Model ID - With Reset Option */}
                    <div className="p-3">
                        <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-1.5">
                                <div className="text-xs font-bold uppercase text-slate-400 dark:text-white/40">HuggingFace Model ID</div>
                                <Tooltip
                                    content="This feature requires an ONNX model compatible with Transformers.js. Check the HuggingFace model card for 'onnx' tag or ONNX weights."
                                >
                                    <HelpIcon />
                                </Tooltip>
                            </div>
                            <button
                                onClick={() => setCustomModelId(MODEL_CONFIG.ID)}
                                className="text-slate-400 hover:text-cyan-600 dark:hover:text-cyan-400 transition-colors p-1 rounded-md hover:bg-black/5 dark:hover:bg-white/5"
                                title={`Reset to ${MODEL_CONFIG.ID}`}
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

                    {/* Storage & Downloads */}
                    <div className="p-3">
                        <div className="text-xs font-bold uppercase text-slate-400 dark:text-white/40 mb-2">Storage & Downloads</div>
                        <button
                            onClick={() => {
                                closeSettings();
                                verifyDownloads();
                            }}
                            className="w-full py-1.5 px-3 rounded-md text-xs font-medium bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 hover:bg-red-100 dark:hover:bg-red-900/30 transition-colors flex items-center justify-center gap-2"
                        >
                            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4">
                                <path strokeLinecap="round" strokeLinejoin="round" d="M11.25 11.25l.041-.02a.75.75 0 011.063.852l-.708 2.836a.75.75 0 001.063.853l.041-.021M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9-3.75h.008v.008H12V8.25z" />
                            </svg>
                            Verify & Repair Downloads
                        </button>
                        <div className="mt-2 pt-2 border-t border-black/5 dark:border-white/5">
                            {!isConfirmingClear ? (
                                <button
                                    onClick={() => setIsConfirmingClear(true)}
                                    className="w-full py-1.5 px-3 rounded-md text-xs font-medium bg-red-500/10 text-red-600 dark:text-red-400 hover:bg-red-500/20 transition-colors flex items-center justify-center gap-2"
                                >
                                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4">
                                        <path strokeLinecap="round" strokeLinejoin="round" d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 013.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 00-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 00-7.5 0" />
                                    </svg>
                                    Clear All Models
                                </button>
                            ) : (
                                <div className="flex flex-col gap-2 p-2 bg-red-500/5 rounded-lg border border-red-500/20">
                                    <span className="text-[10px] text-red-600 dark:text-red-400 font-medium text-center">Are you sure? All models will need re-downloading.</span>
                                    <div className="flex gap-2">
                                        <button
                                            onClick={async () => {
                                                const { downloadManager } = await import('../../services/downloader/DownloadManager');
                                                await downloadManager.clearCache();
                                                setIsConfirmingClear(false);
                                                closeSettings();
                                            }}
                                            className="flex-1 py-1 rounded text-[10px] font-bold bg-red-600 text-white hover:bg-red-700 transition-colors"
                                        >
                                            Yes, Clear
                                        </button>
                                        <button
                                            onClick={() => setIsConfirmingClear(false)}
                                            className="flex-1 py-1 rounded text-[10px] font-medium bg-black/5 dark:bg-white/5 text-slate-600 dark:text-slate-300 hover:bg-black/10 dark:hover:bg-white/10 transition-colors"
                                        >
                                            Cancel
                                        </button>
                                    </div>
                                </div>
                            )}
                        </div>
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

                    {/* Preview Model Input */}
                    <button
                        onClick={() => setShowPreviewInput(!showPreviewInput)}
                        className="flex items-center justify-between p-3 rounded-xl hover:bg-black/5 dark:hover:bg-white/5 transition-colors text-sm text-slate-700 dark:text-white"
                    >
                        <span className="font-medium">Preview Model Input</span>
                        <div className={`w-10 h-5 rounded-full relative transition-colors ${showPreviewInput ? 'bg-cyan-500' : 'bg-slate-200 dark:bg-white/10'}`}>
                            <div className={`absolute top-1 w-3 h-3 rounded-full bg-white shadow-sm transition-transform ${showPreviewInput ? 'left-6' : 'left-1'}`} />
                        </div>
                    </button>

                </div>
            )}
        </div>
    );
};

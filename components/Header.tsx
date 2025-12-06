import React from 'react';
import { useAppContext } from './contexts/AppContext';
import { useThemeContext } from './contexts/ThemeContext';
import { QuantizationSelector } from './QuantizationSelector';
import { ProviderSelector } from './ProviderSelector';

const Header: React.FC = () => {
    const {
        isSidebarOpen,
        toggleSidebar,
        numCandidates,
        setNumCandidates,
        quantization,
        setQuantization,
        provider,
        setProvider,
        showVisualDebugger,
        setShowVisualDebugger,
        activeTab,
        setActiveTab,
    } = useAppContext();
    const { theme, toggleTheme } = useThemeContext();

    return (
        <div className="h-14 flex-none flex items-center justify-between px-4 border-b border-black/5 dark:border-white/5 bg-white/40 dark:bg-black/20 select-none z-30 backdrop-blur-md">
            {/* Left: Sidebar Toggle & Logo */}
            <div className="flex items-center gap-4">
                <button
                    onClick={toggleSidebar}
                    className="p-2 rounded-lg text-slate-500 dark:text-white/40 hover:bg-black/5 dark:hover:bg-white/5 hover:text-slate-800 dark:hover:text-white transition-all"
                    title={isSidebarOpen ? "Close Sidebar" : "Open Sidebar"}
                >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        {isSidebarOpen ? (
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                        ) : (
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                        )}
                    </svg>
                </button>

                <div className="flex items-center gap-3 group">
                    <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-cyan-500/20 to-blue-600/20 dark:from-cyan-500/10 dark:to-blue-600/10 border border-black/10 dark:border-white/10 flex items-center justify-center group-hover:border-cyan-500/40 transition-colors">
                        <svg className="w-5 h-5 text-cyan-600 dark:text-cyan-400" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                        </svg>
                    </div>
                    <h1 className="text-lg font-bold tracking-tight text-slate-800 dark:text-white/90">InkTeX</h1>
                </div>
            </div>

            {/* Right: Controls */}
            <div className="flex items-center gap-4">

                {/* Tab Switcher - Major Mode Toggle */}
                <div className="hidden md:flex items-center p-1.5 bg-black/5 dark:bg-white/5 rounded-2xl border border-black/10 dark:border-white/10 shadow-inner">
                    <button
                        onClick={() => setActiveTab('draw')}
                        className={`flex items-center gap-2 px-4 py-2 text-sm font-bold rounded-xl transition-all duration-200 ${activeTab === 'draw'
                            ? 'bg-gradient-to-r from-cyan-500 to-blue-500 text-white shadow-lg shadow-cyan-500/25'
                            : 'text-slate-500 dark:text-white/40 hover:text-slate-700 dark:hover:text-white/80 hover:bg-black/5 dark:hover:bg-white/5'
                            }`}
                    >
                        <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M12 19l7-7 3 3-7 7-3-3z" />
                            <path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z" />
                            <path d="M2 2l7.586 7.586" />
                            <circle cx="11" cy="11" r="2" />
                        </svg>
                        Draw
                    </button>
                    <button
                        onClick={() => setActiveTab('upload')}
                        className={`flex items-center gap-2 px-4 py-2 text-sm font-bold rounded-xl transition-all duration-200 ${activeTab === 'upload'
                            ? 'bg-gradient-to-r from-cyan-500 to-blue-500 text-white shadow-lg shadow-cyan-500/25'
                            : 'text-slate-500 dark:text-white/40 hover:text-slate-700 dark:hover:text-white/80 hover:bg-black/5 dark:hover:bg-white/5'
                            }`}
                    >
                        <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                            <polyline points="17 8 12 3 7 8" />
                            <line x1="12" y1="3" x2="12" y2="15" />
                        </svg>
                        Upload
                    </button>
                </div>

                {/* Separator */}
                <div className="hidden md:block w-px h-6 bg-black/5 dark:bg-white/5"></div>

                {/* Candidate Count Group */}
                <div className="hidden md:flex items-center p-1 bg-black/5 dark:bg-white/5 rounded-xl border border-black/5 dark:border-white/5">
                    <div className="flex items-center gap-2 px-2">
                        <span className="text-[10px] font-bold uppercase text-slate-400 dark:text-white/40">Candidates</span>
                        <input
                            type="number"
                            min="1"
                            max="5"
                            value={numCandidates}
                            onChange={(e) => {
                                const val = parseInt(e.target.value);
                                if (!isNaN(val)) {
                                    setNumCandidates(Math.min(5, Math.max(1, val)));
                                }
                            }}
                            className="w-10 h-6 text-center text-xs font-mono bg-white dark:bg-white/10 rounded-md border border-black/10 dark:border-white/10 focus:outline-none focus:border-cyan-500 dark:focus:border-cyan-400 text-slate-700 dark:text-white"
                        />

                        <div className="relative group/info">
                            <svg className="w-3.5 h-3.5 text-slate-400 dark:text-white/30 cursor-help hover:text-cyan-600 dark:hover:text-cyan-400 transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>

                            {/* Tooltip */}
                            <div className="absolute top-full right-0 mt-2 w-48 p-2 bg-white dark:bg-[#111] border border-black/10 dark:border-white/10 rounded-lg shadow-xl backdrop-blur-xl z-50 opacity-0 invisible group-hover/info:opacity-100 group-hover/info:visible transition-all duration-200 text-left pointer-events-none">
                                <p className="text-[10px] text-slate-500 dark:text-white/60 leading-tight">
                                    <span className="font-bold text-cyan-600 dark:text-cyan-400">1 Candidate:</span> Fast (Greedy)<br />
                                    <span className="font-bold text-purple-600 dark:text-purple-400">2-5 Candidates:</span> Slower (Beam Search)
                                </p>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Separator */}
                <div className="hidden md:block w-px h-6 bg-black/5 dark:bg-white/5"></div>

                {/* Provider Group */}
                <div className="hidden md:flex items-center p-1 bg-black/5 dark:bg-white/5 rounded-xl border border-black/5 dark:border-white/5">
                    <div className="flex items-center gap-2 px-2">
                        <span className="text-[10px] font-bold uppercase text-slate-400 dark:text-white/40">Provider</span>
                        <ProviderSelector
                            value={provider}
                            onChange={setProvider}
                        />
                    </div>
                </div>

                {/* Separator */}
                <div className="hidden md:block w-px h-6 bg-black/5 dark:bg-white/5"></div>


                {/* Quantization Group */}
                <div className="hidden md:flex items-center p-1 bg-black/5 dark:bg-white/5 rounded-xl border border-black/5 dark:border-white/5">
                    <div className="flex items-center gap-2 px-2">
                        <span className="text-[10px] font-bold uppercase text-slate-400 dark:text-white/40">Quantization</span>
                        <QuantizationSelector
                            value={quantization}
                            onChange={setQuantization}
                        />
                    </div>
                </div>

                {/* Separator */}
                <div className="hidden md:block w-px h-6 bg-black/5 dark:bg-white/5"></div>

                {/* Theme Toggle */}
                <button
                    onClick={toggleTheme}
                    className="w-9 h-9 rounded-xl bg-black/5 dark:bg-white/5 border border-black/5 dark:border-white/5 flex items-center justify-center text-slate-500 dark:text-white/40 hover:text-amber-500 dark:hover:text-yellow-300 hover:bg-black/10 dark:hover:bg-white/10 transition-all shadow-sm"
                    title={theme === 'dark' ? "Switch to Light Mode" : "Switch to Dark Mode"}
                >
                    {theme === 'dark' ? (
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
                        </svg>
                    ) : (
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
                        </svg>
                    )}
                </button>

                {/* Debug Toggle */}
                <button
                    onClick={() => setShowVisualDebugger(!showVisualDebugger)}
                    className={`w-9 h-9 rounded-xl border flex items-center justify-center transition-all shadow-sm ${showVisualDebugger
                        ? 'bg-cyan-500/20 border-cyan-500/40 text-cyan-600 dark:text-cyan-400'
                        : 'bg-black/5 dark:bg-white/5 border-black/5 dark:border-white/5 text-slate-500 dark:text-white/40 hover:text-cyan-500 dark:hover:text-cyan-400 hover:bg-black/10 dark:hover:bg-white/10'
                        }`}
                    title={showVisualDebugger ? "Hide Visual Debugger" : "Show Visual Debugger"}
                >
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
                    </svg>
                </button>

            </div>
        </div>
    );
};

export default Header;

import React from 'react';
import { useAppContext } from './contexts/AppContext';
import { TeXPenLogo } from './TeXPenLogo';
import { SettingsMenu } from './SettingsMenu';

const Header: React.FC = () => {
    const {
        numCandidates,
        setNumCandidates,
        activeTab,
        setActiveTab,
        toggleSidebar,
        isSidebarOpen,
    } = useAppContext();

    return (
        <div className="relative h-16 flex-none flex items-center justify-between px-6 border-b border-black/5 dark:border-white/5 bg-white/40 dark:bg-black/20 select-none z-30 backdrop-blur-md">

            {/* Left: Logo */}
            <div className="flex items-center gap-3 group">
                {/* Minimalist Nib Icon */}
                <div className="relative flex items-center justify-center">
                    <div className="absolute inset-0 bg-cyan-500/20 blur-xl rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                    <svg className="w-8 h-8 text-cyan-500 dark:text-cyan-400 overflow-visible" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                        {/* Rotating Pen Group */}
                        <g className="origin-center transform transition-transform duration-500 group-hover:-rotate-45">
                            <path d="M12 19l7-7 3 3-7 7-3-3z" />
                            <path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z" />
                            <path d="M2 2l7.586 7.586" />
                            <circle cx="11" cy="11" r="2" />
                        </g>
                        {/* Stationary Ink Trace (Arc of the tip movement) */}
                        {/* Tip moves from (2,2) -> (-2.14, 12) when rotated -45deg around (12,12) */}
                        <path
                            d="M 2 2 A 14.14 14.14 0 0 0 -2.14 12"
                            className="opacity-0 group-hover:opacity-100 transition-opacity duration-500"
                            strokeWidth="2"
                        />
                    </svg>
                </div>
                <TeXPenLogo className="h-8 w-auto" />
            </div>

            {/* Center: Mode Switcher */}
            <div className="absolute left-1/2 -translate-x-1/2 flex items-center bg-black/5 dark:bg-white/5 p-1 rounded-full border border-black/5 dark:border-white/5">
                <button
                    onClick={() => setActiveTab('draw')}
                    className={`px-4 py-1.5 rounded-full text-xs font-semibold transition-all duration-300 ${activeTab === 'draw'
                        ? 'bg-white dark:bg-[#222] text-cyan-600 dark:text-cyan-400 shadow-sm'
                        : 'text-slate-500 dark:text-white/40 hover:text-slate-700 dark:hover:text-white'
                        }`}
                >
                    Draw
                </button>
                <button
                    onClick={() => setActiveTab('upload')}
                    className={`px-4 py-1.5 rounded-full text-xs font-semibold transition-all duration-300 ${activeTab === 'upload'
                        ? 'bg-white dark:bg-[#222] text-cyan-600 dark:text-cyan-400 shadow-sm'
                        : 'text-slate-500 dark:text-white/40 hover:text-slate-700 dark:hover:text-white'
                        }`}
                >
                    Upload
                </button>
            </div>

            {/* Right: Controls */}
            <div className="flex items-center gap-3">

                {/* Candidate Count Group */}
                <div className="flex items-center p-1 bg-black/5 dark:bg-white/5 rounded-xl border border-black/5 dark:border-white/5">
                    <div className="flex items-center gap-2 px-2">
                        <span className="hidden sm:inline text-[10px] font-bold uppercase text-slate-400 dark:text-white/40">Candidates</span>
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
                                <p className="text-[10px] text-slate-500 dark:text-white/60 leading-tight w-40">
                                    The model generates multiple guesses for your input to improve accuracy. Choosing more candidates may be slightly slower.
                                </p>
                            </div>
                        </div>
                    </div>
                </div>



                {/* Settings Menu */}
                <SettingsMenu />

            </div>
        </div>
    );
};

export default Header;

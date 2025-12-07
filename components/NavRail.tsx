import React from 'react';
import { useThemeContext } from './contexts/ThemeContext';
import { useAppContext } from './contexts/AppContext';
import { TeXPenLogo } from './TeXPenLogo';

interface NavRailProps {
    activeMode: 'draw' | 'upload';
    onModeChange: (mode: 'draw' | 'upload') => void;
}

const NavRail: React.FC<NavRailProps> = ({ activeMode, onModeChange }) => {
    const { theme, toggleTheme } = useThemeContext();
    const { isSidebarOpen, toggleSidebar } = useAppContext();

    return (
        <div className="flex flex-col w-20 h-full bg-white dark:bg-[#0c0c0c] border-r border-black/5 dark:border-white/5 z-20 flex-none transition-colors duration-500">
            {/* Logo Area */}
            <div className="flex flex-col items-center justify-center py-6 gap-2">
                <div className="group relative flex items-center justify-center w-12 h-12">
                    <div className="absolute inset-0 bg-cyan-500/20 blur-xl rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                    {/* Minimalist Nib Icon with Flow Curve */}
                    <svg className="w-10 h-10 text-cyan-500 dark:text-cyan-400 transform group-hover:-rotate-12 transition-transform duration-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                        {/* Nib Body */}
                        <path d="M12 19l7-7 3 3-7 7-3-3z" />
                        <path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z" />
                        {/* Ink Channel */}
                        <path d="M2 2l7.586 7.586" />
                        {/* Breather hole */}
                        <circle cx="11" cy="11" r="2" />
                        {/* Drawing Curve */}
                        <path d="M12 19 C 12 19, 8 23, 4 21" className="opacity-0 group-hover:opacity-100 transition-opacity duration-500" strokeWidth="2" />
                    </svg>
                </div>

                {/* Brand Name - Custom TeXPen Logo */}
                <div className="w-16 h-8 flex items-center justify-center">
                    <TeXPenLogo
                        className="w-full h-full text-slate-900 dark:text-white relative z-10 transition-all duration-300"
                    />
                </div>
            </div>

            {/* Main Navigation */}
            <div className="flex-1 flex flex-col items-center gap-4 py-8">
                {/* Draw Mode */}
                <button
                    onClick={() => onModeChange('draw')}
                    className={`group relative flex items-center justify-center w-12 h-12 rounded-2xl transition-all duration-300 ${activeMode === 'draw'
                        ? 'bg-cyan-500/10 text-cyan-500 shadow-[0_0_20px_rgba(6,182,212,0.15)]'
                        : 'text-slate-400 dark:text-white/30 hover:bg-black/5 dark:hover:bg-white/5 hover:text-slate-600 dark:hover:text-white'
                        }`}
                    title="Draw"
                >
                    <svg className="w-6 h-6 transition-transform duration-300 group-hover:scale-110" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={activeMode === 'draw' ? 2.5 : 2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                    </svg>

                    {activeMode === 'draw' && (
                        <div className="absolute -left-4 top-1/2 -translate-y-1/2 w-1 h-8 bg-cyan-500 rounded-r-full shadow-[0_0_10px_rgba(6,182,212,0.5)]" />
                    )}
                </button>

                {/* Upload Mode */}
                <button
                    onClick={() => onModeChange('upload')}
                    className={`group relative flex items-center justify-center w-12 h-12 rounded-2xl transition-all duration-300 ${activeMode === 'upload'
                        ? 'bg-cyan-500/10 text-cyan-500 shadow-[0_0_20px_rgba(6,182,212,0.15)]'
                        : 'text-slate-400 dark:text-white/30 hover:bg-black/5 dark:hover:bg-white/5 hover:text-slate-600 dark:hover:text-white'
                        }`}
                    title="Upload Image"
                >
                    <svg className="w-6 h-6 transition-transform duration-300 group-hover:scale-110" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={activeMode === 'upload' ? 2.5 : 2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>

                    {activeMode === 'upload' && (
                        <div className="absolute -left-4 top-1/2 -translate-y-1/2 w-1 h-8 bg-cyan-500 rounded-r-full shadow-[0_0_10px_rgba(6,182,212,0.5)]" />
                    )}
                </button>
            </div>

            {/* Bottom Actions */}
            <div className="flex flex-col items-center gap-3 py-6 pb-8">
                {/* History Toggle */}
                <button
                    onClick={toggleSidebar}
                    className={`w-10 h-10 rounded-xl flex items-center justify-center transition-all duration-200 ${isSidebarOpen
                        ? 'text-cyan-600 dark:text-cyan-400 bg-cyan-50/50 dark:bg-cyan-900/10'
                        : 'text-slate-400 dark:text-white/30 hover:text-slate-600 dark:hover:text-white hover:bg-black/5 dark:hover:bg-white/5'
                        }`}
                    title="History"
                >
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                        <path strokeLinecap="round" strokeLinejoin="round" d="M3 3v5h5" />
                        <path strokeLinecap="round" strokeLinejoin="round" d="M12 7v5l4 2" />
                    </svg>
                </button>

                {/* Theme Toggle */}
                <button
                    onClick={toggleTheme}
                    className="w-10 h-10 rounded-xl text-slate-400 dark:text-white/30 hover:text-amber-500 dark:hover:text-yellow-300 hover:bg-black/5 dark:hover:bg-white/5 flex items-center justify-center transition-all duration-200"
                    title="Toggle Theme"
                >
                    {theme === 'dark' ? (
                        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
                        </svg>
                    ) : (
                        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
                        </svg>
                    )}
                </button>
            </div>
        </div>
    );
};

export default NavRail;

import React from 'react';
import { useAppContext } from '../../contexts/AppContext';
import { TeXPenLogo } from '../common/TeXPenLogo';
import { SettingsMenu } from '../settings/SettingsMenu';
import { Tooltip } from '../common/Tooltip';
import { HelpIcon } from '../common/HelpIcon';
import { PenIcon } from '../common/icons/PenIcon';
import { HamburgerIcon } from '../common/icons/HamburgerIcon';
import { DrawIcon } from '../common/icons/DrawIcon';
import { UploadIcon } from '../common/icons/UploadIcon';
import { IconButton } from '../common/IconButton';

const Header: React.FC = () => {
    const {
        numCandidates,
        setNumCandidates,
        activeTab,
        setActiveTab,
        toggleSidebar,
    } = useAppContext();

    return (
        <div className="relative h-14 md:h-16 flex-none flex items-center justify-between px-3 md:px-6 border-b border-black/5 dark:border-white/5 bg-white/40 dark:bg-black/20 select-none z-30 backdrop-blur-md">

            {/* Left: Logo & Sidebar Toggle */}
            <div className="flex items-center gap-2 md:gap-3 group">
                {/* Mobile Sidebar Toggle */}
                <div className="-ml-2 md:hidden">
                    <IconButton
                        onClick={toggleSidebar}
                        variant="ghost"
                        icon={<HamburgerIcon />}
                    />
                </div>

                {/* Minimalist Nib Icon */}
                <div className="relative flex items-center justify-center">
                    <div className="absolute inset-0 bg-cyan-500/20 blur-xl rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                    <PenIcon />
                </div>
                <TeXPenLogo className="h-8 w-auto" />
            </div>

            {/* Center: Mode Switcher (Desktop Only) */}
            <div className="hidden md:flex md:absolute md:left-1/2 md:-translate-x-1/2 items-center bg-black/5 dark:bg-white/5 p-1 rounded-full border border-black/5 dark:border-white/5">
                <IconButton
                    label="Draw"
                    onClick={() => setActiveTab('draw')}
                    isActive={activeTab === 'draw'}
                    icon={<DrawIcon />}
                    variant="pill"
                />
                <IconButton
                    label="Upload"
                    onClick={() => setActiveTab('upload')}
                    isActive={activeTab === 'upload'}
                    icon={<UploadIcon />}
                    variant="pill"
                />
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
                            className="w-12 h-7 text-center text-xs font-mono bg-black/[0.05] dark:bg-white/[0.05] rounded-lg border border-black/10 dark:border-white/10 focus:outline-none focus:border-cyan-500 dark:focus:border-cyan-400 text-slate-700 dark:text-white transition-all hover:bg-black/[0.08] dark:hover:bg-white/[0.08] hover:border-black/20 dark:hover:border-white/20"
                        />

                        <Tooltip
                            content="The model generates multiple guesses for your input to improve accuracy. Choosing more candidates may be slightly slower."
                        >
                            <HelpIcon />
                        </Tooltip>
                    </div>
                </div>



                {/* Settings Menu */}
                <SettingsMenu />

            </div>
        </div>
    );
};

export default Header;

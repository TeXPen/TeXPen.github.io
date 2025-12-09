import React from 'react';

interface TooltipProps {
    content: React.ReactNode;
    children: React.ReactNode;
    width?: string;
}

export const Tooltip: React.FC<TooltipProps> = ({ content, children, width = 'w-48' }) => {
    return (
        <div className="relative group/tooltip">
            {children}

            <div className={`absolute bottom-full mb-2 ${width} p-2 bg-white dark:bg-[#111] border border-black/10 dark:border-white/10 rounded-lg shadow-xl z-[60] opacity-0 invisible group-hover/tooltip:opacity-100 group-hover/tooltip:visible transition-all duration-200 text-left pointer-events-none -right-2 md:right-0 md:left-auto`}>
                <div className="text-[10px] text-slate-500 dark:text-white/60 leading-tight">
                    {content}
                </div>
            </div>
        </div>
    );
};

import React from 'react';
import { useMathJax } from '../hooks/useMathJax';

interface OutputDisplayProps {
    latex: string;
    isInferencing?: boolean;
}

const OutputDisplay: React.FC<OutputDisplayProps> = ({ latex, isInferencing = false }) => {
    // Trigger MathJax on latex change for specific container
    useMathJax(latex, 'latex-output');

    const handleCopy = () => {
        if (latex) navigator.clipboard.writeText(latex);
    };

    const sanitizeLatex = (text: string) => {
        if (!text) return '';

        // Remove all instances of delimiters globally.
        // We use the 'g' flag to catch them anywhere in the string.
        const clean = text
            .replace(/\\\[/g, '')  // Remove \[
            .replace(/\\\]/g, '')  // Remove \]
            .replace(/\\\(/g, '')  // Remove \(
            .replace(/\\\)/g, '')  // Remove \)
            .replace(/\$\$/g, '')  // Remove $$
            .replace(/^\$|\$$/g, ''); // Remove single $ at start or end

        return clean.trim();
    };

    return (
        <div className="h-[30%] md:h-[35%] relative flex flex-col items-center justify-center bg-gradient-to-b from-white/[0.2] dark:from-white/[0.02] to-transparent z-10">
            <div id="latex-output" className="w-full text-center text-2xl md:text-5xl text-slate-800 dark:text-white px-8 py-4 overflow-x-auto overflow-y-auto scrollbar-thin">
                {isInferencing ? (
                    <div className="flex items-center justify-center gap-3 text-blue-500 dark:text-blue-400">
                        {/* Animated spinner */}
                        <div className="relative w-6 h-6">
                            <div className="absolute inset-0 border-2 border-blue-500/30 dark:border-blue-400/30 rounded-full"></div>
                            <div className="absolute inset-0 border-2 border-blue-500 dark:border-blue-400 border-t-transparent rounded-full animate-spin"></div>
                        </div>
                        <span className="font-medium text-lg animate-pulse">Generating LaTeX...</span>
                    </div>
                ) : latex ? (
                    `\\[${sanitizeLatex(latex)}\\]`
                ) : (
                    <span className="text-slate-300 dark:text-white/10 font-light italic text-xl">Equation preview...</span>
                )}
            </div>

            {/* ... existing Action Bar code ... */}
            <div className="absolute top-4 right-4 flex gap-2">
                <button
                    onClick={handleCopy}
                    disabled={!latex}
                    className="p-2 text-slate-400 dark:text-white/30 hover:text-slate-800 dark:hover:text-white hover:bg-black/5 dark:hover:bg-white/10 rounded-lg transition-colors disabled:opacity-0"
                    title="Copy LaTeX"
                >
                    <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3" />
                    </svg>
                </button>
            </div>
        </div>
    );
};

export default OutputDisplay;
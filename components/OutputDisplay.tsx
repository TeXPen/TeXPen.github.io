import React from 'react';
import { useMathJax } from '../hooks/useMathJax';

interface OutputDisplayProps {
  latex: string;
}

const OutputDisplay: React.FC<OutputDisplayProps> = ({ latex }) => {
  // Trigger MathJax on latex change for specific container
  useMathJax(latex, 'latex-output');

  const handleCopy = () => {
    if (latex) navigator.clipboard.writeText(latex);
  };

  const sanitizeLatex = (text: string) => {
    // 1. Trim whitespace first so the regex anchors (^ and $) match correctly
    let clean = text.trim();

    // 2. Replace using [\s\S]* to match content even if it has newlines
    //    We check the most specific delimiters (\[...\] and $$...$$) first.
    clean = clean
      .replace(/^\\\[([\s\S]*)\\\]$/, '$1') // Matches \[ ... \]
      .replace(/^\$\$([\s\S]*)\$\$$/, '$1') // Matches $$ ... $$
      .replace(/^\\\(([\s\S]*)\\\)$/, '$1') // Matches \( ... \)
      .replace(/^\$([\s\S]*)\$$/, '$1');    // Matches $ ... $

    return clean.trim();
  };

  return (
    <div className="h-[30%] md:h-[35%] relative flex flex-col items-center justify-center bg-gradient-to-b from-white/[0.2] dark:from-white/[0.02] to-transparent z-10">
        <div id="latex-output" className="w-full text-center text-2xl md:text-5xl text-slate-800 dark:text-white px-8 py-4 overflow-x-auto overflow-y-hidden scrollbar-hide">
            {latex ? `\\[${sanitizeLatex(latex)}\\]` : <span className="text-slate-300 dark:text-white/10 font-light italic text-xl">Equation preview...</span>}
        </div>
        
        {/* Action Bar overlay */}
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
        <style>{`
        .scrollbar-hide::-webkit-scrollbar { display: none; }
        .scrollbar-hide { -ms-overflow-style: none; scrollbar-width: none; }
      `}</style>
    </div>
  );
};

export default OutputDisplay;
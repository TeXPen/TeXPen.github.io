import React, { useState, useRef, useCallback } from 'react';

interface ImageUploadAreaProps {
    onImageSelect: (file: File) => void;
    onConvert: () => void;
    isInferencing: boolean;
    previewUrl: string | null;
}

const ImageUploadArea: React.FC<ImageUploadAreaProps> = ({
    onImageSelect,
    onConvert,
    isInferencing,
    previewUrl
}) => {
    const [isDragging, setIsDragging] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            onImageSelect(file);
        }
    };

    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            onImageSelect(file);
        }
    };

    React.useEffect(() => {
        const handlePaste = (e: ClipboardEvent) => {
            if (e.clipboardData && e.clipboardData.files.length > 0) {
                const file = e.clipboardData.files[0];
                if (file.type.startsWith('image/')) {
                    e.preventDefault();
                    onImageSelect(file);
                }
            }
        };

        window.addEventListener('paste', handlePaste);
        return () => window.removeEventListener('paste', handlePaste);
    }, [onImageSelect]);

    return (
        <div className="flex-1 flex flex-col items-center justify-center p-2 bg-transparent w-full h-full">
            {!previewUrl ? (
                <div
                    className={`
                        w-full h-full max-w-4xl border-4 border-dashed rounded-xl flex flex-col items-center justify-center cursor-pointer transition-all duration-300
                        ${isDragging
                            ? 'border-cyan-500 bg-cyan-50/10 dark:bg-cyan-900/10 scale-[0.99]'
                            : 'border-black/10 dark:border-white/10 hover:border-cyan-400 dark:hover:border-cyan-600 hover:bg-black/5 dark:hover:bg-white/5'
                        }
                    `}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    onClick={() => fileInputRef.current?.click()}
                >
                    <input
                        type="file"
                        ref={fileInputRef}
                        className="hidden"
                        accept="image/*"
                        onChange={handleFileSelect}
                    />

                    <div className="w-20 h-20 mb-6 rounded-2xl bg-black/5 dark:bg-white/5 flex items-center justify-center">
                        <svg className="w-10 h-10 text-slate-400 dark:text-white/40" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                        </svg>
                    </div>

                    <h3 className="text-xl font-bold text-slate-700 dark:text-white mb-2">Upload Image</h3>
                    <p className="text-slate-500 dark:text-white/50 text-center max-w-xs">
                        {isDragging ? "Drop it here!" : "Drag & drop an image here, or click to browse"}
                    </p>
                    <p className="mt-4 text-xs text-slate-400 dark:text-white/30 font-mono">
                        Supports PNG, JPG, JPEG
                    </p>
                </div>
            ) : (
                <div className="w-full max-w-xl flex flex-col items-center gap-6 animate-in fade-in zoom-in duration-300">
                    <div className="relative w-full aspect-square md:aspect-video rounded-3xl overflow-hidden shadow-2xl border border-black/10 dark:border-white/10 bg-white dark:bg-black/50">
                        <img
                            src={previewUrl}
                            alt="Preview"
                            className="w-full h-full object-contain"
                        />
                        <button
                            onClick={() => fileInputRef.current?.click()}
                            className="absolute top-4 right-4 p-2 rounded-xl bg-black/50 hover:bg-black/70 text-white transition-colors backdrop-blur-md"
                            title="Change Image"
                        >
                            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                            </svg>
                        </button>
                    </div>

                    <div className="flex gap-4 w-full">
                        <button
                            onClick={onConvert}
                            disabled={isInferencing}
                            className={`
                                flex-1 py-4 rounded-xl font-bold text-lg shadow-xl shadow-cyan-500/20 transition-all active:scale-95
                                ${isInferencing
                                    ? 'bg-slate-100 dark:bg-white/10 text-slate-400 cursor-not-allowed'
                                    : 'bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white'
                                }
                            `}
                        >
                            {isInferencing ? (
                                <span className="flex items-center justify-center gap-3">
                                    <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"></circle>
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                    </svg>
                                    Processing...
                                </span>
                            ) : "Convert to LaTeX"}
                        </button>

                        <input
                            type="file"
                            ref={fileInputRef}
                            className="hidden"
                            accept="image/*"
                            onChange={handleFileSelect}
                        />
                    </div>
                </div>
            )}
        </div>
    );
};

export default ImageUploadArea;

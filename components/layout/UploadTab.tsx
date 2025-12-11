import React from 'react';
import { useAppContext } from '../../contexts/AppContext';
import ImageUploadArea from '../upload/ImageUploadArea';
import OutputDisplay from '../display/OutputDisplay';
import Candidates from '../display/Candidates';

interface UploadTabProps {
    onImageSelect: (file: File) => void;
    onConvert: () => Promise<void>;
    onUploadAnother: () => void;
    renderLoadingOverlay: () => React.ReactNode;
}

const UploadTab: React.FC<UploadTabProps> = ({
    onImageSelect,
    onConvert,
    onUploadAnother,
    renderLoadingOverlay
}) => {
    const {
        status,
        latex,
        userConfirmed,
        isInferencing,
        activeInferenceTab,
        uploadPreview,
        showUploadResult
    } = useAppContext();

    const isUploadInferencing = isInferencing && activeInferenceTab === 'upload';

    return (
        <div className="flex-1 relative overflow-hidden flex flex-col">
            <div className="absolute inset-0 z-10 bg-transparent p-4 flex flex-col overflow-hidden">
                <div className={`flex-1 bg-white/50 dark:bg-black/20 rounded-2xl overflow-hidden backdrop-blur-sm w-full h-full flex flex-col ${showUploadResult ? 'relative' : 'items-center justify-center'}`}>

                    {!showUploadResult ? (
                        /* Input / Preview State */
                        <ImageUploadArea
                            onImageSelect={onImageSelect}
                            onConvert={onConvert}
                            isInferencing={isUploadInferencing}
                            previewUrl={uploadPreview}
                        />
                    ) : (
                        /* Result Split View */
                        <div className="flex-1 flex w-full h-full animate-in fade-in duration-500 flex-col divide-y divide-black/5 dark:divide-white/5">

                            {/* Top Panel: Result & Controls */}
                            <div className="h-1/2 flex flex-col relative bg-white/40 dark:bg-transparent min-h-0">
                                <OutputDisplay
                                    latex={latex}
                                    isInferencing={isUploadInferencing}
                                    className="flex-1 w-full"
                                />
                                <Candidates />
                            </div>

                            {/* Bottom Panel: Source Image */}
                            <div
                                onClick={onUploadAnother}
                                className="flex-1 relative bg-black/5 dark:bg-white/5 min-h-0 cursor-pointer group hover:bg-black/10 dark:hover:bg-white/10 transition-colors"
                            >
                                {/* Image Area - Constrained to top portion */}
                                <div className="absolute inset-x-0 top-0 bottom-24 flex items-center justify-center p-4">
                                    {uploadPreview && (
                                        <img
                                            src={uploadPreview}
                                            alt="Original"
                                            className="max-w-full max-h-full object-contain shadow-lg rounded-lg transition-transform duration-300 group-hover:scale-[1.02]"
                                        />
                                    )}
                                    <div className="absolute top-4 left-4 inline-flex items-center px-2 py-1 bg-black/20 backdrop-blur-sm rounded text-xs text-white/50">
                                        Original Image
                                    </div>
                                </div>

                                {/* Floating Upload Button - In reserved bottom space */}
                                <button
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        onUploadAnother();
                                    }}
                                    className="absolute bottom-6 left-1/2 -translate-x-1/2 px-6 py-2 md:px-8 md:py-3 text-sm md:text-base bg-cyan-500/10 dark:bg-cyan-500/10 hover:bg-cyan-500/20 dark:hover:bg-cyan-500/20 text-cyan-600 dark:text-cyan-400 font-bold rounded-full border border-cyan-500/50 dark:border-cyan-400/50 hover:border-cyan-500 dark:hover:border-cyan-400 hover:shadow-lg hover:shadow-cyan-500/20 active:scale-95 transition-all flex items-center gap-2 backdrop-blur-md z-10"
                                >
                                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                                    </svg>
                                    Upload Another Image
                                </button>
                            </div>

                        </div>
                    )}
                </div>
                {status === 'loading' && userConfirmed && renderLoadingOverlay()}
            </div>
        </div>
    );
};

export default UploadTab;

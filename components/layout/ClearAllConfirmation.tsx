import React from 'react';
import { TrashIcon } from '../common/icons/HistoryIcons';

interface ClearAllConfirmationProps {
    isOpen: boolean;
    onConfirm: () => void;
    onCancel: () => void;
}

/**
 * Modal dialog for confirming history clear action
 */
const ClearAllConfirmation: React.FC<ClearAllConfirmationProps> = ({
    isOpen,
    onConfirm,
    onCancel
}) => {
    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/50 backdrop-blur-sm animate-in fade-in duration-200">
            <div className="bg-white dark:bg-[#111] border border-black/10 dark:border-white/10 rounded-2xl shadow-2xl max-w-sm w-full p-6 m-4 transform animate-in zoom-in-95 duration-200">
                <div className="flex flex-col items-center text-center space-y-4">
                    <div className="w-12 h-12 rounded-full bg-red-100 dark:bg-red-900/20 flex items-center justify-center text-red-600 dark:text-red-400">
                        <TrashIcon />
                    </div>

                    <h3 className="text-lg font-bold text-slate-900 dark:text-white">Clear All History?</h3>

                    <p className="text-sm text-slate-500 dark:text-white/60">
                        This action cannot be undone. All your math history and sessions will be permanently deleted.
                    </p>

                    <div className="grid grid-cols-2 gap-3 w-full pt-2">
                        <button
                            onClick={onCancel}
                            className="px-4 py-2 rounded-xl text-sm font-medium text-slate-700 dark:text-white/80 bg-slate-100 dark:bg-white/5 hover:bg-slate-200 dark:hover:bg-white/10 transition-colors"
                        >
                            Cancel
                        </button>
                        <button
                            onClick={onConfirm}
                            className="px-4 py-2 rounded-xl text-sm font-medium text-white bg-red-600 hover:bg-red-500 shadow-lg shadow-red-500/20 transition-all active:scale-95"
                        >
                            Yes, Clear All
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ClearAllConfirmation;

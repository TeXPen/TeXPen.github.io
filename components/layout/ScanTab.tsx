import React, { useState } from 'react';
import { PaddleOCREngine, ScanResult } from '../../services/ocr_backend/PaddleOCREngine';

const paddleEngine = new PaddleOCREngine();

interface ScanTabProps {
    renderLoadingOverlay: () => React.ReactNode;
}

const ScanTab: React.FC<ScanTabProps> = ({ renderLoadingOverlay }) => {
    const [status, setStatus] = useState<string>('Ready');
    const [result, setResult] = useState<ScanResult | null>(null);
    const [imagePreview, setImagePreview] = useState<string | null>(null);

    const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        setImagePreview(URL.createObjectURL(file));
        setStatus('Initializing Engine...');

        try {
            await paddleEngine.init(); // Ideally this should be single init
            setStatus('Processing...');

            const res = await paddleEngine.process(file);
            console.log('ScanResult:', res);
            setResult(res);
            setStatus('Done');
        } catch (err) {
            console.error(err);
            setStatus('Error: ' + String(err));
        }
    };

    return (
        <div className="flex flex-col items-center justify-center w-full h-full p-6 text-slate-600 dark:text-slate-300">
            <h2 className="text-2xl font-bold mb-4">Structure Scan (Beta)</h2>
            <div className="mb-6">
                <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileChange}
                    className="block w-full text-sm text-slate-500
                        file:mr-4 file:py-2 file:px-4
                        file:rounded-full file:border-0
                        file:text-sm file:font-semibold
                        file:bg-violet-50 file:text-violet-700
                        hover:file:bg-violet-100"
                />
            </div>

            {imagePreview && (
                <div className="mb-4 relative border border-slate-200 dark:border-slate-700 rounded-lg overflow-hidden max-h-[300px]">
                    <img src={imagePreview} alt="Preview" className="object-contain w-full h-full" />
                </div>
            )}

            <div className="text-sm font-mono mb-4 bg-slate-100 dark:bg-slate-900 p-2 rounded">
                Status: {status}
            </div>

            {renderLoadingOverlay()}

            {result && (
                <div className="w-full max-w-4xl grid grid-cols-2 gap-4 h-full overflow-auto">
                    <div className="bg-white dark:bg-neutral-900 p-4 rounded-xl shadow-sm border border-slate-200 dark:border-slate-800 overflow-auto">
                        <h3 className="font-bold mb-2">Layout</h3>
                        <pre className="text-xs">{JSON.stringify(result.layout, null, 2)}</pre>
                    </div>
                    <div className="bg-white dark:bg-neutral-900 p-4 rounded-xl shadow-sm border border-slate-200 dark:border-slate-800 overflow-auto">
                        <h3 className="font-bold mb-2">Extracted Text</h3>
                        <pre className="whitespace-pre-wrap text-sm">{result.rawText}</pre>
                    </div>
                </div>
            )}
        </div>
    );
};

export default ScanTab;

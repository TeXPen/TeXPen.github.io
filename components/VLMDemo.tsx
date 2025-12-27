
import React, { useState, useRef, useEffect } from 'react';
import { VLMInferenceEngine } from '../services/inference/VLMInferenceEngine';
import { VLMInferenceResult } from '../services/inference/types';

const vlmEngine = new VLMInferenceEngine();

export const VLMDemo: React.FC = () => {
    const [image, setImage] = useState<Blob | null>(null);
    const [imagePreview, setImagePreview] = useState<string | null>(null);
    const [prompt, setPrompt] = useState("Describe this image.");
    const [result, setResult] = useState<VLMInferenceResult | null>(null);
    const [loading, setLoading] = useState(false);
    const [status, setStatus] = useState("");
    const [progress, setProgress] = useState(0);

    const fileInputRef = useRef<HTMLInputElement>(null);

    useEffect(() => {
        // Init engine on mount (or first interaction)
        return () => {
            // cleanup if needed
        };
    }, []);

    const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            const file = e.target.files[0];
            setImage(file);
            setImagePreview(URL.createObjectURL(file));
            setResult(null); // clear previous
        }
    };

    const handleRun = async () => {
        if (!image) return;
        setLoading(true);
        setStatus("Initializing...");
        try {
            await vlmEngine.init((s, p) => {
                setStatus(s);
                if (p !== undefined) setProgress(p);
            });

            setStatus("Running Inference...");
            const res = await vlmEngine.inferVLM(image, prompt);
            setResult(res);
            setStatus("Done");
        } catch (e) {
            console.error(e);
            setStatus("Error: " + (e as Error).message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="container mx-auto p-4 max-w-4xl">
            <h1 className="text-3xl font-bold mb-6">PaddleOCR-VL Local Demo</h1>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="border rounded-lg p-4 bg-gray-50 dark:bg-gray-800">
                    <h2 className="text-xl font-semibold mb-4">Input</h2>

                    <div
                        className="border-2 border-dashed border-gray-300 rounded-lg h-64 flex items-center justify-center cursor-pointer mb-4 hover:bg-gray-100 dark:hover:bg-gray-700 transition"
                        onClick={() => fileInputRef.current?.click()}
                    >
                        {imagePreview ? (
                            <img src={imagePreview} alt="Preview" className="max-h-full max-w-full object-contain" />
                        ) : (
                            <span className="text-gray-500">Click to upload image</span>
                        )}
                        <input
                            type="file"
                            ref={fileInputRef}
                            className="hidden"
                            accept="image/*"
                            onChange={handleImageUpload}
                        />
                    </div>

                    <div className="mb-4">
                        <label className="block text-sm font-medium mb-1">Prompt</label>
                        <textarea
                            className="w-full p-2 border rounded dark:bg-gray-700 font-mono"
                            rows={3}
                            value={prompt}
                            onChange={(e) => setPrompt(e.target.value)}
                        />
                    </div>

                    <button
                        className={`w-full py-2 px-4 rounded font-bold text-white transition ${loading || !image ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'}`}
                        onClick={handleRun}
                        disabled={loading || !image}
                    >
                        {loading ? 'Processing...' : 'Run Inference'}
                    </button>

                    {loading && (
                        <div className="mt-4">
                            <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">{status}</div>
                            <div className="w-full bg-gray-200 rounded-full h-2.5 dark:bg-gray-700">
                                <div className="bg-blue-600 h-2.5 rounded-full transition-all duration-300" style={{ width: `${progress}%` }}></div>
                            </div>
                        </div>
                    )}
                </div>

                <div className="border rounded-lg p-4 bg-gray-50 dark:bg-gray-800 flex flex-col">
                    <h2 className="text-xl font-semibold mb-4">Output</h2>
                    <div className="flex-grow bg-white dark:bg-gray-900 rounded border p-4 font-mono whitespace-pre-wrap overflow-auto h-96">
                        {result ? result.markdown : (
                            <span className="text-gray-400 italic">Results will appear here...</span>
                        )}
                    </div>
                    {result?.timings && (
                        <div className="mt-4 text-xs text-gray-500">
                            <p>Preprocess: {result.timings.preprocess?.toFixed(2)}ms</p>
                            <p>Vision Encoder: {result.timings.vision_encoder?.toFixed(2)}ms</p>
                            <p>Generation: {result.timings.generation?.toFixed(2)}ms</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

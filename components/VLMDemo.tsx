import React, { useState, useRef, useEffect } from 'react';
import { VLMInferenceEngine } from '../services/inference/VLMInferenceEngine';
import { VLMInferenceResult } from '../services/inference/types';

const vlmEngine = new VLMInferenceEngine();

const blobToBase64 = (blob: Blob): Promise<string> => {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result as string);
        reader.onerror = reject;
        reader.readAsDataURL(blob);
    });
};

const forceReload = () => {
    console.log("Forcing reload...");
    try {
        window.location.reload();
    } catch (e) {
        console.error("Standard reload failed, forcing location assignment", e);
        window.location.href = window.location.href;
    }
};

export const VLMDemo: React.FC = () => {
    const [image, setImage] = useState<Blob | null>(null);
    const [imagePreview, setImagePreview] = useState<string | null>(null);
    const [prompt, setPrompt] = useState("Describe this image.");
    const [result, setResult] = useState<VLMInferenceResult | null>(null);
    const [loading, setLoading] = useState(false);
    const [status, setStatus] = useState("");
    const [progress, setProgress] = useState(0);

    const fileInputRef = useRef<HTMLInputElement>(null);

    // Restoration Effect
    useEffect(() => {
        const restoreState = async () => {
            const flag = sessionStorage.getItem("vlm_restore_needed");
            if (flag === "true") {
                console.log("[VLMDemo] restoring state after reload...");
                setStatus("Restoring previous session...");

                const savedPrompt = sessionStorage.getItem("vlm_prompt");
                const savedImageB64 = sessionStorage.getItem("vlm_image");
                const savedStrategy = sessionStorage.getItem("vlm_next_strategy");

                if (savedPrompt) setPrompt(savedPrompt);

                if (savedStrategy) {
                    const idx = parseInt(savedStrategy, 10);
                    if (!isNaN(idx)) {
                        console.log(`[VLMDemo] Forcing strategy index: ${idx}`);
                        vlmEngine.setStrategyIndex(idx);
                    }
                }

                if (savedImageB64) {
                    try {
                        const res = await fetch(savedImageB64);
                        const blob = await res.blob();
                        setImage(blob);
                        setImagePreview(savedImageB64);

                        // Auto-run?
                        if (sessionStorage.getItem("vlm_autorun") === "true") {
                            setTimeout(() => handleRun(blob, savedPrompt || "Describe this image."), 500);
                        }
                    } catch (e) {
                        console.error("Failed to restore image", e);
                    }
                }

                // Cleanup
                sessionStorage.removeItem("vlm_restore_needed");
                sessionStorage.removeItem("vlm_prompt");
                sessionStorage.removeItem("vlm_image");
                sessionStorage.removeItem("vlm_next_strategy");
                sessionStorage.removeItem("vlm_autorun");
            }
        };

        restoreState();

        const unhandledHandler = (event: PromiseRejectionEvent) => {
            const reason = event.reason;
            console.error("[VLMDemo] Uncaught Promise Rejection:", reason);
            const msg = reason instanceof Error ? reason.message : String(reason);
            if (msg && (
                msg.includes("valid external Instance") ||
                msg.includes("Aborted()") ||
                msg.includes("out of memory") ||
                msg.includes("createBuffer")
            )) {
                event.preventDefault();
                sessionStorage.setItem("vlm_restore_needed", "true");
                sessionStorage.setItem("vlm_autorun", "true");

                const currentIdx = vlmEngine.getStrategyIndex();
                sessionStorage.setItem("vlm_next_strategy", (currentIdx + 1).toString());

                // Use robust reload
                forceReload();
            }
        };

        window.addEventListener("unhandledrejection", unhandledHandler);

        return () => {
            window.removeEventListener("unhandledrejection", unhandledHandler);
            vlmEngine.dispose();
        };
    }, []);

    // Keep a ref to image for the event listener (state is stale in useEffect)
    const imageRef = useRef<Blob | null>(null);
    const promptRef = useRef<string>("");

    useEffect(() => { imageRef.current = image; }, [image]);
    useEffect(() => { promptRef.current = prompt; }, [prompt]);

    const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            const file = e.target.files[0];
            setImage(file);
            setImagePreview(URL.createObjectURL(file));
            setResult(null); // clear previous
        }
    };

    const handleReloadRecovery = async (currentImage: Blob, currentPrompt: string) => {
        setStatus("Critical Error. Preparing to reload...");

        // Prepare state to save
        const saveState = async () => {
            try {
                const b64 = await blobToBase64(currentImage);
                sessionStorage.setItem("vlm_image", b64);
            } catch (e) {
                console.error("Failed to save image during crash recovery", e);
            }
            sessionStorage.setItem("vlm_restore_needed", "true");
            sessionStorage.setItem("vlm_prompt", currentPrompt);
            sessionStorage.setItem("vlm_autorun", "true");

            const currentIdx = vlmEngine.getStrategyIndex();
            // Downgrade for next run
            sessionStorage.setItem("vlm_next_strategy", (currentIdx + 1).toString());
        };

        // Race: Try to save, but reload anyway after 500ms
        const timeout = new Promise((resolve) => setTimeout(resolve, 500));

        try {
            await Promise.race([saveState(), timeout]);
            console.log("State save attempt finished (or timed out). Reloading.");
            forceReload();
        } catch (e) {
            console.error("Recovery logic failed", e);
            forceReload(); // Last resort
        }
    };

    const handleRun = async (imgOverride?: Blob, promptOverride?: string) => {
        const imgToUse = imgOverride || image;
        const promptToUse = promptOverride || prompt;

        if (!imgToUse) return;

        setLoading(true);
        setStatus("Initializing...");
        let fatal = false;
        try {
            await vlmEngine.init((s, p) => {
                setStatus(s);
                if (p !== undefined) setProgress(p);
            });

            setStatus("Running Inference...");
            const res = await vlmEngine.inferVLM(imgToUse, promptToUse);
            setResult(res);
            setStatus("Done");
        } catch (e) {
            console.error(e);
            let errMsg = "";
            let isFatal = false;

            if (e instanceof Error) {
                errMsg = e.message;
            } else {
                errMsg = String(e);
            }

            // Catch specific strings OR the custom error
            if (errMsg === "FATAL_RELOAD_NEEDED" ||
                errMsg.includes("Aborted") ||
                errMsg.includes("valid external Instance")
            ) {
                fatal = true;
                await handleReloadRecovery(imgToUse, promptToUse);
            } else {
                setStatus("Error: " + errMsg);
            }
        } finally {
            if (!fatal) {
                setLoading(false);
            }
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
                        onClick={() => handleRun()}
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

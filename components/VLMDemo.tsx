import React, { useState, useRef, useEffect } from 'react';
import { VLMWorkerClient as VLMInferenceEngine } from '../services/inference/VLMWorkerClient';
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

const determineNextStrategy = (currentIdx: number): number => {
    const phase = sessionStorage.getItem("vlm_phase");
    console.log(`[VLMDemo] Recovery Analysis: Crash at phase '${phase}', Current Strategy ${currentIdx}`);

    if (phase === 'LOADING_LLM') {
        // LLM failed on GPU. Must move LLM to CPU.
        // Strategies 0-3 have LLM on GPU. Strategy 4 is CPU_ONLY.
        console.log("-> Jump to CPU_ONLY (Index 4)");
        return 4;
    }

    if (phase === 'LOADING_VISION') {
        // Vision failed, but LLM likely loaded ok (since it loads before Vision now).
        // Try to keep LLM on GPU, but move Vision to CPU.
        // Strategy 2 (LLM_GPU) does exactly this.
        if (currentIdx < 2) {
            console.log("-> Jump to LLM_GPU (Index 2)");
            return 2;
        }
    }

    // Default: Linear backoff
    return currentIdx + 1;
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

    const initializedRef = useRef(false);

    // Consolidated Initialization & Restoration Effect
    useEffect(() => {
        if (initializedRef.current) return;
        initializedRef.current = true;

        const performInitialLoad = async () => {
            const flag = sessionStorage.getItem("vlm_restore_needed");
            const isRestoring = flag === "true";

            setLoading(true);

            if (isRestoring) {
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
                        await vlmEngine.setStrategyIndex(idx);
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
                            // Run after engine init
                            setTimeout(async () => {
                                setStatus("Initializing for Auto-run...");
                                try {
                                    await vlmEngine.init((s, p) => {
                                        setStatus(s);
                                        if (p !== undefined) setProgress(p);
                                    }, (phase) => {
                                        sessionStorage.setItem("vlm_phase", phase);
                                    });
                                    handleRun(blob, savedPrompt || "Describe this image.");
                                } catch (e) {
                                    console.error("Auto-run init failed", e);
                                    setLoading(false);
                                }
                            }, 100);
                        }
                    } catch (e) {
                        console.error("Failed to restore image", e);
                    }
                }

                // Cleanup session storage early to prevent re-restoration on unrelated reloads
                sessionStorage.removeItem("vlm_restore_needed");
                sessionStorage.removeItem("vlm_prompt");
                sessionStorage.removeItem("vlm_image");
                sessionStorage.removeItem("vlm_next_strategy");
                sessionStorage.removeItem("vlm_autorun");

                if (sessionStorage.getItem("vlm_autorun") !== "true") {
                    setLoading(false);
                    setStatus("Ready");
                }
            } else {
                // Standard Preload
                try {
                    setStatus("Preloading Models...");
                    await vlmEngine.init((s, p) => {
                        setStatus(s);
                        if (p !== undefined) setProgress(p);
                    }, (phase) => {
                        sessionStorage.setItem("vlm_phase", phase);
                    });
                    setStatus("Ready");
                    setLoading(false);
                } catch (e) {
                    console.error("Preload failed", e);
                    setLoading(false);
                }
            }
        };

        performInitialLoad();

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
                const nextIdx = determineNextStrategy(currentIdx);

                if (nextIdx > 4 || (nextIdx === 4 && currentIdx === 4)) {
                    console.error("All strategies failed. Stopping auto-reload.");
                    setStatus("Critical Error: All recovery strategies failed.");
                    setLoading(false);
                    return;
                }

                sessionStorage.setItem("vlm_next_strategy", nextIdx.toString());
                forceReload();
            }
        };

        window.addEventListener("unhandledrejection", unhandledHandler);

        return () => {
            window.removeEventListener("unhandledrejection", unhandledHandler);
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
            const nextIdx = determineNextStrategy(currentIdx);

            if (nextIdx > 4 || (nextIdx === 4 && currentIdx === 4)) {
                console.error("All strategies failed. Stopping auto-reload.");
                setStatus("Critical Error: All recovery strategies failed.");
                // We don't reload here, just letting the error stay visible
                return;
            }

            // Downgrade for next run
            sessionStorage.setItem("vlm_next_strategy", nextIdx.toString());
        };

        // Race: Try to save, but reload anyway after 500ms
        const timeout = new Promise((resolve) => setTimeout(resolve, 500));

        try {
            const currentIdx = vlmEngine.getStrategyIndex();
            // Wait, double check logic: determineNextStrategy uses currentIdx. 
            // If determining next strategy was handled in saveState, we probably don't need check here? 
            // But we do need to check if we SHOULD restart.

            // Re-calc nextIdx just to check termination condition
            const nextIdx = determineNextStrategy(currentIdx);
            if (nextIdx > 4 || (nextIdx === 4 && currentIdx === 4)) return;

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
            }, (phase) => {
                sessionStorage.setItem("vlm_phase", phase);
            });

            setStatus("Running Inference...");
            const res = await vlmEngine.runInference(imgToUse, promptToUse, (token, fullText) => {
                setResult(prev => ({
                    markdown: fullText,
                    timings: prev?.timings || {}
                }));
            });
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

    const handleAbort = async () => {
        try {
            setStatus("Aborting...");
            await vlmEngine.abort();
        } catch (e) {
            console.error("Abort failed", e);
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

                    <div className="flex gap-2">
                        <button
                            className={`flex-1 py-2 px-4 rounded font-bold text-white transition ${loading || !image ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'}`}
                            onClick={() => handleRun()}
                            disabled={loading || !image}
                        >
                            {loading ? 'Processing...' : 'Run Inference'}
                        </button>
                        {loading && (
                            <button
                                className="py-2 px-4 rounded font-bold text-white bg-red-600 hover:bg-red-700 transition"
                                onClick={handleAbort}
                                title="Stop generation"
                            >
                                Stop
                            </button>
                        )}
                        <button
                            className="py-2 px-4 rounded font-bold text-gray-700 bg-gray-200 hover:bg-gray-300 transition dark:bg-gray-700 dark:text-gray-200 dark:hover:bg-gray-600"
                            onClick={async () => {
                                await vlmEngine.dispose();
                                setStatus("Models Unloaded");
                                setProgress(0);
                            }}
                            disabled={loading}
                            title="Unload models from memory"
                        >
                            Unload
                        </button>
                    </div>

                    {loading && (
                        <div className="mt-4">
                            <div className="flex justify-between items-center mb-1">
                                <div className="text-sm text-gray-600 dark:text-gray-400">{status}</div>
                                <div className="text-xs font-bold text-blue-600 uppercase">
                                    Strategy: {vlmEngine.getStrategyIndex() + 1}/5
                                </div>
                            </div>
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

import React, { useState } from 'react';
import Select, { StylesConfig } from 'react-select';
import { useThemeContext } from '../../contexts/ThemeContext';
import { useAppContext } from '../../contexts/AppContext';
import { Quantization, PerformanceProfile, QUANTIZATION_OPTIONS, PROFILE_OPTIONS } from '../../services/inference/types';
import { Tooltip } from '../common/Tooltip';
import { HelpIcon } from '../common/HelpIcon';

export function QuantizationSelector() {
    const { theme } = useThemeContext();
    const {
        performanceProfile,
        setPerformanceProfile,
        encoderQuantization,
        setEncoderQuantization,
        decoderQuantization,
        setDecoderQuantization,

        setQuantization,
        provider,
    } = useAppContext();

    const [isAdvanced, setIsAdvanced] = useState(false);

    const getSelectStyles = (isSubOption = false): StylesConfig => ({
        control: (provided) => ({
            ...provided,
            width: '100%',
            backgroundColor: theme === 'dark' ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.05)',
            border: `1px solid ${theme === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)'}`,
            borderRadius: '0.5rem',
            boxShadow: 'none',
            cursor: 'pointer',
            padding: '2px 8px',
            minHeight: '38px',
            fontSize: isSubOption ? '0.8rem' : '0.9rem',
            '&:hover': {
                backgroundColor: theme === 'dark' ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.08)',
                borderColor: theme === 'dark' ? 'rgba(255, 255, 255, 0.2)' : 'rgba(0, 0, 0, 0.2)',
            },
        }),
        menu: (provided) => ({
            ...provided,
            backgroundColor: theme === 'dark' ? '#1a1a1a' : 'white',
            border: `1px solid ${theme === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)'}`,
            borderRadius: '0.5rem',
            zIndex: 100,
        }),
        option: (provided, { isSelected, isFocused }) => ({
            ...provided,
            backgroundColor: isSelected
                ? 'rgb(6 182 212)'
                : isFocused
                    ? (theme === 'dark' ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.05)')
                    : 'transparent',
            color: isSelected
                ? 'white'
                : (theme === 'dark' ? 'rgba(255, 255, 255, 0.8)' : '#1f2937'),
            cursor: 'pointer',
            fontSize: '0.85rem',
        }),
        singleValue: (provided) => ({
            ...provided,
            color: theme === 'dark' ? 'rgba(255, 255, 255, 0.9)' : 'rgba(0, 0, 0, 0.9)',
        }),
        input: (provided) => ({
            ...provided,
            color: theme === 'dark' ? 'white' : 'black',
        }),
    });

    const handleProfileChange = (val: PerformanceProfile) => {
        setPerformanceProfile(val);
        // The AppProvider effect will update the individual quantizations
    };

    const handleManualChange = (type: 'encoder' | 'decoder', val: Quantization) => {
        if (type === 'encoder') setEncoderQuantization(val);
        if (type === 'decoder') setDecoderQuantization(val);

        // Changing simple quantization for legacy compatibility if needed, though mostly unused now
        if (type === 'encoder') setQuantization(val);

        setPerformanceProfile('custom');
    };

    const selectedProfile = PROFILE_OPTIONS.find(p => p.value === performanceProfile);
    const selectedEncoder = QUANTIZATION_OPTIONS.find(q => q.value === encoderQuantization);
    const selectedDecoder = QUANTIZATION_OPTIONS.find(q => q.value === decoderQuantization);

    return (
        <div className="flex flex-col gap-3">
            {/* Main Profile Selector */}
            <div>
                <div className="flex items-center gap-1.5 mb-2">
                    <div className="text-xs font-bold uppercase text-slate-400 dark:text-white/40">Performance Profile</div>
                    <Tooltip content="Select a preset for the best balance of quality and speed.">
                        <HelpIcon />
                    </Tooltip>
                </div>
                <Select
                    value={selectedProfile}
                    onChange={(opt) => opt && handleProfileChange((opt as { value: PerformanceProfile }).value)}
                    options={PROFILE_OPTIONS.filter(p => {
                        if (provider === 'wasm') {
                            // CPU (WASM): Hide 'fast' (FP16 encoder) and also likely 'high_quality' (FP32) if we want to force int8 default behavior preference
                            // User said: "fp16 is very very slow on CPU, so it should not appear" -> This targets 'fast' profile (FP16 encoder) 
                            // and 'high_quality' (FP32) is also slow but maybe acceptable?
                            // User said "default to balanced".
                            // Let's hide 'fast' (FP16/FP32). 
                            // Should we hide 'high_quality' (FP32/FP32)? It's slow but compatible. 
                            // Let's keep 'high_quality' as an option for "Full Quality" but discourage it?
                            // Actually, user said "For CPU, default to "Balanced"... with options for "Lowest Memory"... and "Full Quality"".
                            // So we KEEP 'high_quality' and 'balanced' and 'low_memory'. We HIDE 'fast' (FP16).
                            return p.value !== 'fast';
                        } else {
                            // GPU (WebGPU): Hide 'balanced' (Int8) and 'low_memory' (Int4)
                            return p.value !== 'balanced' && p.value !== 'low_memory';
                        }
                    })}
                    styles={getSelectStyles()}
                    isSearchable={false}
                    formatOptionLabel={(option: unknown) => {
                        const opt = option as { label: string; description?: string };
                        return (
                            <div className="flex flex-col gap-0.5">
                                <span className="font-medium">{opt.label}</span>
                                {opt.description && (
                                    <span className="text-[10px] opacity-60">{opt.description}</span>
                                )}
                            </div>
                        );
                    }}
                />
            </div>

            {/* Advanced Toggle */}
            <div>
                <button
                    onClick={() => setIsAdvanced(!isAdvanced)}
                    className="flex items-center gap-2 text-xs text-cyan-600 dark:text-cyan-400 font-medium hover:underline focus:outline-none"
                >
                    <svg
                        xmlns="http://www.w3.org/2000/svg"
                        fill="none"
                        viewBox="0 0 24 24"
                        strokeWidth={2}
                        stroke="currentColor"
                        className={`w-3 h-3 transition-transform ${isAdvanced ? 'rotate-90' : ''}`}
                    >
                        <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
                    </svg>
                    Advanced Settings
                </button>
            </div>

            {/* Advanced Controls */}
            {isAdvanced && (
                <div className="flex flex-col gap-3 pl-3 border-l-2 border-slate-100 dark:border-white/5 animate-in slide-in-from-top-2 duration-200">

                    {/* Information Box */}
                    <div className="p-2 rounded bg-blue-500/10 border border-blue-500/20 text-[10px] text-blue-600 dark:text-blue-300">
                        <p className="font-semibold mb-0.5">Note:</p>
                        The FP16 decoder has been removed as it is slower than FP32 with marginal memory savings on GPU, and significantly slower on CPU due to lack of native support.
                    </div>

                    {/* Encoder */}
                    <div className="flex flex-col gap-1">
                        <span className="text-[10px] font-bold uppercase text-slate-400 dark:text-white/40">Encoder</span>
                        <Select
                            value={selectedEncoder}
                            onChange={(opt) => opt && handleManualChange('encoder', (opt as { value: Quantization }).value)}
                            options={QUANTIZATION_OPTIONS.filter(q => {
                                if (provider === 'wasm') {
                                    // Hide FP16 for CPU
                                    return q.value !== 'fp16';
                                } else {
                                    // Hide Int8/Int4 for GPU (unless we really want to allow it in Advanced... user said "hide int8/int4 in GPU tab")
                                    // "GPU tab" might refer to the main selector, but since this is "Advanced", maybe we allow it?
                                    // The request "hide int8/int4 in GPU tab" likely refers to the "Provider" context.
                                    // If I hide them here, the user can't select them even manually.
                                    // Let's be strict as requested.
                                    return q.value !== 'int8' && q.value !== 'int4';
                                }
                            })}
                            styles={getSelectStyles(true)}
                            isSearchable={false}
                        />
                    </div>

                    {/* Decoder */}
                    <div className="flex flex-col gap-1">
                        <div className="flex items-center gap-1.5">
                            <span className="text-[10px] font-bold uppercase text-slate-400 dark:text-white/40">Decoder</span>
                            {/* Filter out FP16 from view if needed, or keeping it but it acts as FP32 internally */}
                        </div>
                        <Select
                            value={selectedDecoder}
                            // Prevent selecting FP16 for decoder based on general removal AND provider specific logic
                            options={QUANTIZATION_OPTIONS.filter(o => {
                                if (o.value === 'fp16') return false; // Global removal of FP16 decoder
                                if (provider === 'webgpu') {
                                    return o.value !== 'int8' && o.value !== 'int4';
                                }
                                return true;
                            })}
                            onChange={(opt) => opt && handleManualChange('decoder', (opt as { value: Quantization }).value)}
                            styles={getSelectStyles(true)}
                            isSearchable={false}
                        />
                    </div>
                </div>
            )}
        </div>
    );
}

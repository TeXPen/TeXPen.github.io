import React, { useEffect, useState } from 'react';
import Select, { StylesConfig } from 'react-select';
import { useThemeContext } from './contexts/ThemeContext';
import { isWebGPUAvailable } from '../utils/env';

type Provider = 'webgpu' | 'wasm' | 'webgl';

interface ProviderSelectorProps {
    value: Provider;
    onChange: (value: Provider) => void;
}

const providerOptions: { value: Provider, label: string }[] = [
    { value: 'webgpu', label: 'WebGPU (Fastest)' },
    { value: 'webgl', label: 'WebGL (Legacy GPU)' },
    { value: 'wasm', label: 'WASM (Most Compatible, CPU)' },
];

export function ProviderSelector({ value, onChange }: ProviderSelectorProps) {
    const { theme } = useThemeContext();
    const [availableProviders, setAvailableProviders] = useState(providerOptions);

    useEffect(() => {
        isWebGPUAvailable().then(available => {
            if (!available) {
                setAvailableProviders(providerOptions.filter(p => p.value !== 'webgpu'));
            }
        });
    }, []);


    const selectedOption = availableProviders.find(option => option.value === value);

    const customStyles: StylesConfig = {
        control: (provided) => ({
            ...provided,
            width: 200,
            height: 24,
            minHeight: 24,
            backgroundColor: theme === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgb(255, 255, 255)',
            border: `1px solid ${theme === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)'}`,
            borderRadius: '0.375rem',
            boxShadow: 'none',
            '&:hover': {
                borderColor: theme === 'dark' ? 'rgb(34 211 238)' : 'rgb(6 182 212)',
            },
        }),
        valueContainer: (provided) => ({
            ...provided,
            height: '24px',
            padding: '0 8px',
        }),
        input: (provided) => ({
            ...provided,
            margin: '0px',
            height: '24px',
            color: theme === 'dark' ? 'white' : 'black',
        }),
        indicatorSeparator: () => ({
            display: 'none',
        }),
        indicatorsContainer: (provided) => ({
            ...provided,
            height: '24px',
        }),
        singleValue: (provided) => ({
            ...provided,
            color: theme === 'dark' ? 'white' : '#1f2937',
            fontSize: '0.75rem',
            fontFamily: 'monospace',
        }),
        menu: (provided) => ({
            ...provided,
            backgroundColor: theme === 'dark' ? '#111' : 'white',
            border: `1px solid ${theme === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)'}`,
            borderRadius: '0.5rem',
        }),
        option: (provided, { isSelected, isFocused }) => ({
            ...provided,
            backgroundColor: isSelected ? (theme === 'dark' ? 'rgb(6 182 212)' : 'rgb(6 182 212)') : isFocused ? (theme === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.05)') : 'transparent',
            color: isSelected ? 'white' : (theme === 'dark' ? 'white' : '#1f2937'),
            fontSize: '0.875rem',
        }),
    };

    return (
        <Select
            value={selectedOption}
            onChange={(option: { value: Provider; label: string } | null) => onChange(option?.value || 'wasm')}
            options={availableProviders}
            isSearchable={false}
            styles={customStyles}
            placeholder="Select provider..."
        />
    );
}

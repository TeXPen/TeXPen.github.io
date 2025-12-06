import React from 'react';
import Select, { StylesConfig } from 'react-select';
import { useThemeContext } from './contexts/ThemeContext';

interface QuantizationSelectorProps {
    value: string;
    onChange: (value: string) => void;
}

const options = [
    { value: 'fp32', label: 'FP32 (Large, Slow)' },
    { value: 'fp16', label: 'FP16 (Disabled - Bad Performance)', isDisabled: true },
    { value: 'q8', label: 'INT8 (Disabled - Bad Performance)', isDisabled: true },
];

export function QuantizationSelector({ value, onChange }: QuantizationSelectorProps) {
    const { theme } = useThemeContext();
    const selectedOption = options.find(option => option.value === value);

    const customStyles: StylesConfig = {
        control: (provided) => ({
            ...provided,
            width: 160,
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
        option: (provided, { isSelected, isFocused, isDisabled }) => ({
            ...provided,
            backgroundColor: isSelected
                ? (theme === 'dark' ? 'rgb(6 182 212)' : 'rgb(6 182 212)')
                : isFocused
                    ? (theme === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.05)')
                    : 'transparent',
            color: isDisabled
                ? (theme === 'dark' ? 'rgba(255, 255, 255, 0.3)' : 'rgba(0, 0, 0, 0.3)')
                : isSelected
                    ? 'white'
                    : (theme === 'dark' ? 'white' : '#1f2937'),
            fontSize: '0.875rem',
            cursor: isDisabled ? 'not-allowed' : 'default',
            opacity: isDisabled ? 0.5 : 1,
        }),
    };

    return (
        <Select
            value={selectedOption}
            onChange={(option: { value: string, label: string } | null) => onChange(option?.value || '')}
            options={options}
            isSearchable={false}
            styles={customStyles}
            placeholder="Select quantization..."
        />
    );
}

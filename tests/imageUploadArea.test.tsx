// @vitest-environment jsdom
import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import ImageUploadArea from '../components/ImageUploadArea';
import { AppContext } from '../components/contexts/AppContext';

// Partial mock of AppContext
const mockInferFromUrl = vi.fn();
const mockAddToHistory = vi.fn();

const defaultContext: any = {
    theme: 'light',
    toggleTheme: vi.fn(),
    brushColor: '#000000',
    setBrushColor: vi.fn(),
    brushSize: 2,
    setBrushSize: vi.fn(),
    isErasing: false,
    setIsErasing: vi.fn(),
    clearCanvas: vi.fn(),
    undo: vi.fn(),
    redo: vi.fn(),
    runInference: vi.fn(),
    inferFromUrl: mockInferFromUrl,
    history: [],
    addToHistory: mockAddToHistory,
    clearHistory: vi.fn(),
    activeTab: 'upload',
    setActiveTab: vi.fn(),
};

const Wrapper = () => {
    const [previewUrl, setPreviewUrl] = React.useState<string | null>(null);
    const [isInferencing, setIsInferencing] = React.useState(false);

    const handleImageSelect = (file: File) => {
        setPreviewUrl(URL.createObjectURL(file));
    };

    const handleConvert = async () => {
        setIsInferencing(true);
        await mockInferFromUrl();
        setIsInferencing(false);
    };

    return (
        <AppContext.Provider value={defaultContext}>
            <ImageUploadArea
                onImageSelect={handleImageSelect}
                onConvert={handleConvert}
                isInferencing={isInferencing}
                previewUrl={previewUrl}
            />
        </AppContext.Provider>
    );
};

describe('ImageUploadArea', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        global.URL.createObjectURL = vi.fn(() => 'blob:mock-url');
    });

    it('renders upload area initially', () => {
        render(<Wrapper />);
        expect(screen.getByText(/Drag & drop an image here, or click to browse/i)).toBeInTheDocument();
    });

    it('handles file selection', async () => {
        const { container } = render(<Wrapper />);
        const file = new File(['dummy content'], 'test.png', { type: 'image/png' });
        const input = container.querySelector('input[type="file"]');

        Object.defineProperty(input, 'files', {
            value: [file]
        });
        fireEvent.change(input!);

        await waitFor(() => {
            expect(screen.getByAltText('Preview')).toBeInTheDocument();
        });
    });

    it('triggers conversion on button click', async () => {
        const { container } = render(<Wrapper />);
        const file = new File(['dummy content'], 'test.png', { type: 'image/png' });
        const input = container.querySelector('input[type="file"]');

        Object.defineProperty(input, 'files', {
            value: [file]
        });
        fireEvent.change(input!);

        await waitFor(() => {
            expect(screen.getByText('Convert to LaTeX')).toBeInTheDocument();
        });

        mockInferFromUrl.mockResolvedValue({ latex: 'test latex', candidates: [] });

        fireEvent.click(screen.getByText('Convert to LaTeX'));

        expect(mockInferFromUrl).toHaveBeenCalledTimes(1);
    });
});

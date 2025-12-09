// @vitest-environment jsdom
import React, { useEffect } from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import CanvasArea from '../components/canvas/CanvasArea';
import { describe, it, expect, vi } from 'vitest';

// Mock CanvasBoard to ensure refs are passed back to CanvasArea
vi.mock('../components/canvas/CanvasBoard', () => ({
    default: ({ refCallback, contentRefCallback }: any) => {
        useEffect(() => {
            // Create mock canvas elements
            const mockCanvas = document.createElement('canvas');
            const mockContentCanvas = document.createElement('canvas');

            // Invoke callbacks to simulate component mounting
            refCallback(mockCanvas);
            contentRefCallback(mockContentCanvas);
        }, []);
        return <div>MockCanvasBoard</div>;
    }
}));

describe('CanvasArea', () => {
    it('calls onClear when Clear button is clicked', () => {
        const mockOnClear = vi.fn();
        const mockOnStrokeEnd = vi.fn();

        render(
            <CanvasArea
                theme="light"
                onClear={mockOnClear}
                onStrokeEnd={mockOnStrokeEnd}
            />
        );

        const clearBtn = screen.getByTitle('Clear Canvas');
        fireEvent.click(clearBtn);

        expect(mockOnClear).toHaveBeenCalled();
    });
});

// @vitest-environment jsdom
import React from 'react';
import { render, fireEvent, screen } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import CanvasToolbar from '../../../components/canvas/CanvasToolbar';
import { ToolType } from '../../../types/canvas';

describe('CanvasToolbar', () => {
    const defaultProps = {
        activeTool: 'pen' as ToolType,
        onToolChange: vi.fn(),
        onUndo: vi.fn(),
        onRedo: vi.fn(),
        canUndo: true,
        canRedo: true
    };

    it('renders all main tools', () => {
        render(<CanvasToolbar {...defaultProps} />);

        expect(screen.getByTitle('Undo')).toBeDefined();
        expect(screen.getByTitle('Redo')).toBeDefined();
        // Check for Pen, Select, Eraser - titles might be in SVGs or buttons
        // The component puts titles on buttons
        expect(screen.getByTitle('Pen')).toBeDefined();
        expect(screen.getByTitle('Select')).toBeDefined();
        expect(screen.getByTitle('Eraser')).toBeDefined();
    });

    it('calls onToolChange when clicking pen', () => {
        // Start with something else
        render(<CanvasToolbar {...defaultProps} activeTool="select-rect" />);

        const penBtn = screen.getByTitle('Pen'); // Adjust if title is on parent
        fireEvent.click(penBtn);

        expect(defaultProps.onToolChange).toHaveBeenCalledWith('pen');
    });

    it('toggles eraser menu when clicking eraser', () => {
        render(<CanvasToolbar {...defaultProps} activeTool="eraser-line" />);

        const eraserBtn = screen.getByTitle('Eraser');

        // 1. Click to open menu (since it's already active)
        fireEvent.click(eraserBtn);

        expect(screen.getByTitle('Radial Eraser')).toBeDefined();
        expect(screen.getByTitle('Stroke Eraser')).toBeDefined();

        // 2. Click to close
        fireEvent.click(eraserBtn);
        expect(screen.queryByTitle('Radial Eraser')).toBeNull();
    });

    it('selects eraser tool and opens menu if not active', () => {
        render(<CanvasToolbar {...defaultProps} activeTool="pen" />);

        const eraserBtn = screen.getByTitle('Eraser');
        fireEvent.click(eraserBtn);

        // Should switch to default eraser (line) AND open menu
        expect(defaultProps.onToolChange).toHaveBeenCalledWith('eraser-line');
        expect(screen.getByTitle('Radial Eraser')).toBeDefined();
    });

    it('toggles select menu when clicking select', () => {
        render(<CanvasToolbar {...defaultProps} activeTool="select-rect" />);

        const selectBtn = screen.getByTitle('Select');

        // 1. Click to open menu
        fireEvent.click(selectBtn);

        expect(screen.getByTitle('Rectangle Selection')).toBeDefined();
        expect(screen.getByTitle('Lasso Selection')).toBeDefined();

        // 2. Click to close
        fireEvent.click(selectBtn);
        expect(screen.queryByTitle('Rectangle Selection')).toBeNull();
    });

    it('selects default select tool and opens menu if not active', () => {
        render(<CanvasToolbar {...defaultProps} activeTool="pen" />);

        const selectBtn = screen.getByTitle('Select');
        fireEvent.click(selectBtn);

        // Should switch to default select (rect) AND open menu
        expect(defaultProps.onToolChange).toHaveBeenCalledWith('select-rect');
        expect(screen.getByTitle('Rectangle Selection')).toBeDefined();
    });

    it('calls onUndo and onRedo', () => {
        render(<CanvasToolbar {...defaultProps} />);

        const undoBtn = screen.getByTitle('Undo');
        const redoBtn = screen.getByTitle('Redo');

        fireEvent.click(undoBtn);
        expect(defaultProps.onUndo).toHaveBeenCalled();

        fireEvent.click(redoBtn);
        expect(defaultProps.onRedo).toHaveBeenCalled();
    });

    it('disables undo/redo buttons when canUndo/canRedo is false', () => {
        render(<CanvasToolbar {...defaultProps} canUndo={false} canRedo={false} />);

        const undoBtn = screen.getByTitle('Undo') as HTMLButtonElement;
        const redoBtn = screen.getByTitle('Redo') as HTMLButtonElement;

        expect(undoBtn.disabled).toBe(true);
        expect(redoBtn.disabled).toBe(true);
    });
});

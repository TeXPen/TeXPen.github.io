// @vitest-environment jsdom
import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import Header from '../components/Header';
import { AppContext } from '../components/contexts/AppContext';
import { ThemeContext } from '../components/contexts/ThemeContext';

// Mocks
vi.mock('../components/QuantizationSelector', () => ({
    QuantizationSelector: () => <div data-testid="quantization-selector" />
}));
vi.mock('../components/ProviderSelector', () => ({
    ProviderSelector: () => <div data-testid="provider-selector" />
}));

const mockSetActiveTab = vi.fn();
const mockToggleSidebar = vi.fn();
const mockSetNumCandidates = vi.fn();
const mockToggleTheme = vi.fn();

const defaultAppContext: any = {
    isSidebarOpen: true,
    toggleSidebar: mockToggleSidebar,
    numCandidates: 3,
    setNumCandidates: mockSetNumCandidates,
    quantization: 'q8',
    setQuantization: vi.fn(),
    provider: 'webgpu',
    setProvider: vi.fn(),
    showVisualDebugger: false,
    setShowVisualDebugger: vi.fn(),
    activeTab: 'draw',
    setActiveTab: mockSetActiveTab,
};

const defaultThemeContext: any = {
    theme: 'light',
    toggleTheme: mockToggleTheme,
};

const renderHeader = (appOverrides = {}, themeOverrides = {}) => {
    return render(
        <ThemeContext.Provider value={{ ...defaultThemeContext, ...themeOverrides }}>
            <AppContext.Provider value={{ ...defaultAppContext, ...appOverrides }}>
                <Header />
            </AppContext.Provider>
        </ThemeContext.Provider>
    );
};

describe('Header', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('renders correct active tab styling', () => {
        renderHeader({ activeTab: 'draw' });
        const drawBtn = screen.getByText('Draw');
        const uploadBtn = screen.getByText('Upload');

        expect(drawBtn.className).toContain('text-slate-800'); // active style
        expect(uploadBtn.className).toContain('text-slate-400'); // inactive style
    });

    it('switches tabs when clicked', () => {
        renderHeader({ activeTab: 'draw' });
        const uploadBtn = screen.getByText('Upload');

        fireEvent.click(uploadBtn);
        expect(mockSetActiveTab).toHaveBeenCalledWith('upload');
    });

    it('toggles sidebar', () => {
        renderHeader();
        const toggleBtn = screen.getByTitle('Close Sidebar');
        fireEvent.click(toggleBtn);
        expect(mockToggleSidebar).toHaveBeenCalled();
    });

    it('toggles theme', () => {
        renderHeader();
        const themeBtn = screen.getByTitle('Switch to Dark Mode');
        fireEvent.click(themeBtn);
        expect(mockToggleTheme).toHaveBeenCalled();
    });
});

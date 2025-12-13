/* eslint-disable @typescript-eslint/no-explicit-any */
import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import Header from '../../../components/layout/Header';
import { AppContext } from '../../../contexts/AppContext';
import { ThemeContext } from '../../../contexts/ThemeContext';
import { HistoryContext } from '../../../contexts/HistoryContext';
import { MODEL_CONFIG } from '../../../services/inference/config';

// Mocks
vi.mock('../../../components/settings/QuantizationSelector', () => ({
    QuantizationSelector: () => <div data-testid="quantization-selector" />
}));
vi.mock('../../../components/settings/ProviderSelector', () => ({
    ProviderSelector: () => <div data-testid="provider-selector" />
}));

const mockSetActiveTab = vi.fn();
const mockToggleSidebar = vi.fn();
const mockSetNumCandidates = vi.fn();
const mockToggleTheme = vi.fn();

const defaultAppContext = {
    isSidebarOpen: true,
    toggleSidebar: mockToggleSidebar,
    numCandidates: 3,
    setNumCandidates: mockSetNumCandidates,
    provider: MODEL_CONFIG.PROVIDERS.WEBGPU,
    setProvider: vi.fn(),
    showVisualDebugger: false,
    setShowVisualDebugger: vi.fn(),
    activeTab: 'draw',
    setActiveTab: mockSetActiveTab,
    customModelId: 'test/model',
    setCustomModelId: vi.fn(),
    isInitialized: true,
    // Settings state
    isSettingsOpen: false,
    openSettings: vi.fn(),
    closeSettings: vi.fn(),
    settingsFocus: null,
} as any;

const defaultThemeContext = {
    theme: 'light',
    toggleTheme: mockToggleTheme,
} as any;

const defaultHistoryContext = {
    history: [],
    addToHistory: vi.fn(),
    deleteHistoryItem: vi.fn(),
    clearHistory: vi.fn(),
    filterMode: 'all',
    setFilterMode: vi.fn(),
} as any;

const renderHeader = (appOverrides: any = {}, themeOverrides: any = {}) => {
    return render(
        <ThemeContext.Provider value={{ ...defaultThemeContext, ...themeOverrides }}>
            <AppContext.Provider value={{ ...defaultAppContext, ...appOverrides }}>
                <HistoryContext.Provider value={defaultHistoryContext}>
                    <Header />
                </HistoryContext.Provider>
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
        const drawBtn = screen.getByRole('button', { name: 'Draw' });
        const uploadBtn = screen.getByRole('button', { name: 'Upload' });

        expect(drawBtn.className).toContain('text-cyan-600'); // active style updated
        expect(uploadBtn.className).toContain('text-slate-500'); // inactive style updated
    });

    it('switches tabs when clicked', () => {
        renderHeader({ activeTab: 'draw' });
        const uploadBtn = screen.getByText('Upload');

        fireEvent.click(uploadBtn);
        expect(mockSetActiveTab).toHaveBeenCalledWith('upload');
    });

    it('opens settings when clicking the gear icon', () => {
        const mockOpenSettings = vi.fn();
        renderHeader({ openSettings: mockOpenSettings, isSettingsOpen: false });

        const settingsBtn = screen.getByTitle('Settings');
        fireEvent.click(settingsBtn);

        expect(mockOpenSettings).toHaveBeenCalled();
    });

    it('toggles theme via settings menu', () => {
        // Render with settings ALREADY open to test internal buttons
        renderHeader({ isSettingsOpen: true });

        // Find theme button (text "Theme") - visible only when open
        const themeBtn = screen.getByText('Theme');
        fireEvent.click(themeBtn);

        expect(mockToggleTheme).toHaveBeenCalled();
    });
});

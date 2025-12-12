import React from 'react';
import { useTheme } from '../hooks/useTheme';
import { ThemeContext } from './ThemeContext';

export const ThemeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const { theme, toggleTheme } = useTheme();

    return (
        <ThemeContext.Provider value={{ theme, toggleTheme }}>
            {children}
        </ThemeContext.Provider>
    );
};

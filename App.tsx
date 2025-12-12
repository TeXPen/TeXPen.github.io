import React from 'react';
import { AppProvider } from './contexts/AppProvider';
import { ThemeProvider } from './contexts/ThemeProvider';
import { HistoryProvider } from './contexts/HistoryProvider';
import Main from './components/layout/Main';

const App: React.FC = () => {
    return (
        <ThemeProvider>
            <HistoryProvider>
                <AppProvider>
                    <Main />
                </AppProvider>
            </HistoryProvider>
        </ThemeProvider>
    );
};

export default App;
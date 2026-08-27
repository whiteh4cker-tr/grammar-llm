import { useCallback, useMemo, useState, type ReactNode } from 'react';
import Box from '@mui/material/Box';
import CssBaseline from '@mui/material/CssBaseline';
import { ThemeProvider } from '@mui/material/styles';
import { DownloadProvider } from './DownloadProvider';
import { ColorModeContext, useColorMode } from './colorMode';
import { backgroundGradient, getStoredMode, getTheme, storeMode, type ThemeMode } from './theme';

/** Page gradient, kept out of `body` so it cannot fight `<CssBaseline />`. */
function BrandBackdrop() {
  const { mode } = useColorMode();
  return (
    <Box
      aria-hidden
      sx={{
        position: 'fixed',
        inset: 0,
        zIndex: -1,
        background: backgroundGradient(mode),
      }}
    />
  );
}

export function AppProviders({ children }: { children: ReactNode }) {
  const [mode, setModeState] = useState<ThemeMode>(getStoredMode);

  const setMode = useCallback((next: ThemeMode) => {
    storeMode(next);
    setModeState(next);
  }, []);

  const value = useMemo(() => ({ mode, setMode }), [mode, setMode]);

  return (
    <ColorModeContext.Provider value={value}>
      <DownloadProvider>
        <ThemeProvider theme={getTheme(mode)}>
          <CssBaseline />
          <BrandBackdrop />
          {children}
        </ThemeProvider>
      </DownloadProvider>
    </ColorModeContext.Provider>
  );
}

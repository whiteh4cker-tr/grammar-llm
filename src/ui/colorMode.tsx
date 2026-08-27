import { createContext, useContext } from 'react';
import type { ThemeMode } from './theme';

export interface ColorModeValue {
  mode: ThemeMode;
  setMode: (mode: ThemeMode) => void;
}

/**
 * Kept separate from `<AppProviders />` so each file exports one kind of thing
 * (React Fast Refresh requires component-only modules).
 */
export const ColorModeContext = createContext<ColorModeValue | null>(null);

/** Read/toggle the app color scheme. Provided by `<AppProviders />`. */
export function useColorMode(): ColorModeValue {
  const value = useContext(ColorModeContext);
  if (!value) throw new Error('useColorMode must be used inside <AppProviders>');
  return value;
}

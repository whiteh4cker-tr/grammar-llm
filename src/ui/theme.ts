import { createTheme, type PaletteOptions, type Theme } from '@mui/material/styles';

export type ThemeMode = 'light' | 'dark';

/** Brand colors carried over from the pre-MUI stylesheet. */
export const BRAND = {
  purple: '#667eea',
  purpleLight: '#8b9cf3',
  purpleDark: '#5a67d8',
  violet: '#764ba2',
  violetLight: '#9b6cc4',
  violetDark: '#5f3c84',
  green: '#48bb78',
  greenLight: '#68d391',
  greenDark: '#38a169',
  red: '#e53e3e',
  redLight: '#fc8181',
  redDark: '#c53030',
} as const;

/**
 * Full-page gradient behind the app. Rendered by `<BrandBackdrop />` rather
 * than `palette.background.default`, because a palette entry must be a flat
 * color.
 */
export function backgroundGradient(mode: ThemeMode): string {
  return mode === 'dark'
    ? 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)'
    : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
}

const lightPalette: PaletteOptions = {
  primary: { main: BRAND.purple, light: BRAND.purpleLight, dark: BRAND.purpleDark, contrastText: '#ffffff' },
  secondary: { main: BRAND.violet, light: BRAND.violetLight, dark: BRAND.violetDark, contrastText: '#ffffff' },
  success: { main: BRAND.green, light: BRAND.greenLight, dark: BRAND.greenDark, contrastText: '#ffffff' },
  error: { main: BRAND.red, light: BRAND.redLight, dark: BRAND.redDark, contrastText: '#ffffff' },
  background: { default: '#eef1f8', paper: '#ffffff' },
  text: { primary: '#2d3748', secondary: '#4a5568' },
};

const darkPalette: PaletteOptions = {
  // Same brand hues as light mode (as before the port); a few shades are
  // nudged so white/dark label text keeps its contrast.
  primary: { main: BRAND.purple, light: BRAND.purpleLight, dark: BRAND.purpleDark, contrastText: '#ffffff' },
  secondary: { main: '#9b79c4', light: BRAND.violetLight, dark: BRAND.violetDark, contrastText: '#ffffff' },
  success: { main: '#2f9e63', light: BRAND.greenLight, dark: BRAND.greenDark, contrastText: '#ffffff' },
  error: { main: '#e05252', light: BRAND.redLight, dark: BRAND.redDark, contrastText: '#ffffff' },
  background: { default: '#16213e', paper: '#2d3748' },
  text: { primary: '#e2e8f0', secondary: '#a0aec0' },
  divider: 'rgba(226, 232, 240, 0.16)',
};

/** Ink of the writing-quality score readout, which sits on `background.paper`. */
export interface ScoreInk {
  /** The number itself. */
  value: string;
  /** The dimmer `/ 100` suffix. */
  suffix: string;
  /** Outlined chip border. */
  border: string;
}

/**
 * Score readout colours per colour scheme, as *painted* (solid hex, never
 * `opacity`: a translucent colour must be measured as the blend it produces,
 * and an `opacity` on text is how contrast quietly drops).
 */
export const SCORE_INK: Record<ThemeMode, ScoreInk> = {
  // Measured against `background.paper` (white / #2d3748):
  //   light  6.73:1, 4.54:1, outline 3.39:1
  //   dark   8.54:1, 6.46:1, outline 3.73:1
  light: { value: '#276749', suffix: '#2f855a', border: '#689580' },
  dark: { value: '#86efac', suffix: '#68d391', border: '#5e9c7f' },
};

/**
 * Green used as *text* (not as a fill) on a paper surface.
 *
 * `palette.success.main` is tuned so that a white label stays readable *on*
 * green; that same green is far too dark to read as text on a dark panel and far
 * too light to read on white, so anything that writes green words uses this.
 */
export const SUCCESS_TEXT: Record<ThemeMode, string> = {
  light: '#1e6e3e', // 6.26:1 on white
  dark: '#68d391', // 6.46:1 on #2d3748, 9.40:1 on the dialog surface
};

/** Background of the settings dialog surface per colour scheme. */
export const DIALOG_SURFACE: Record<ThemeMode, string> = {
  light: '#ffffff',
  // Deeper than `background.paper` (#2d3748), see the MuiDialog override below.
  dark: '#121a28',
};

const shared = {
  typography: {
    fontFamily: "'Roboto', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
    // Material's default is `uppercase`. This app renders real words in buttons
    // (the word-fix popup suggests e.g. "their"), where casing is wrong.
    button: { textTransform: 'none' },
    h4: { fontWeight: 700 },
    h5: { fontWeight: 600 },
    h6: { fontWeight: 600 },
    subtitle2: { fontWeight: 600 },
  },
  shape: { borderRadius: 8 },
  components: {
    MuiButton: { styleOverrides: { root: { fontWeight: 600 } } },
    MuiDialog: {
      styleOverrides: {
        // Dark mode: the dialog holds far more text than the cards behind it, so
        // it gets a deeper surface. On `background.paper` (#2d3748) its secondary
        // text sits at 5.3:1 and the green captions at 3.5:1; on this surface
        // they reach 7.7:1 and 9.4:1 while the title goes to 14.2:1.
        paper: ({ theme }: { theme: Theme }) => ({
          backgroundColor: DIALOG_SURFACE[theme.palette.mode],
        }),
      },
    },
  },
};

export const lightTheme: Theme = createTheme({ ...shared, palette: { ...lightPalette, mode: 'light' } });
export const darkTheme: Theme = createTheme({ ...shared, palette: { ...darkPalette, mode: 'dark' } });

export function getTheme(mode: ThemeMode): Theme {
  return mode === 'dark' ? darkTheme : lightTheme;
}

/**
 * Persist the color scheme.
 *
 * Deliberately keeps the pre-MUI `'theme'` localStorage key so existing
 * installs keep their preference. Two plain (non-CSS-variable) themes are
 * swapped instead of using `colorSchemes`: in MUI v9 a theme with
 * `colorSchemes` is routed to the CSS-theme-variable provider, where palette
 * values become `var(--mui-…)` strings — that breaks `alpha()` and the exact
 * legacy hex values the editor overlay and suggestion highlights need.
 */
const STORAGE_KEY = 'theme';

export function getStoredMode(): ThemeMode {
  try {
    return localStorage.getItem(STORAGE_KEY) === 'light' ? 'light' : 'dark';
  } catch {
    return 'dark';
  }
}

export function storeMode(mode: ThemeMode): void {
  try {
    localStorage.setItem(STORAGE_KEY, mode);
  } catch {
    // ignore storage errors
  }
}

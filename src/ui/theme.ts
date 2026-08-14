export type Theme = 'light' | 'dark';

export function getStoredTheme(): Theme {
  try {
    return localStorage.getItem('theme') === 'light' ? 'light' : 'dark';
  } catch {
    return 'dark';
  }
}

export function applyTheme(theme: Theme): void {
  document.body.classList.toggle('dark-mode', theme === 'dark');
  try {
    localStorage.setItem('theme', theme);
  } catch {
    // ignore storage errors
  }
}

/** Apply the persisted theme to <body>; returns the theme applied. */
export function applyStoredTheme(): Theme {
  const theme = getStoredTheme();
  applyTheme(theme);
  return theme;
}

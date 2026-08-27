// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useTheme } from '@mui/material/styles';
import { AppProviders } from './AppProviders';
import { useColorMode } from './colorMode';

/** Minimal in-memory Storage: Node's experimental global shadows jsdom's. */
function memoryStorage(): Storage {
  const store = new Map<string, string>();
  return {
    getItem: (key: string) => (store.has(key) ? store.get(key) ?? null : null),
    setItem: (key: string, value: string) => { store.set(key, value); },
    removeItem: (key: string) => { store.delete(key); },
    clear: () => store.clear(),
    key: (index: number) => Array.from(store.keys())[index] ?? null,
    length: 0,
  };
}

function Probe() {
  const { mode, setMode } = useColorMode();
  const theme = useTheme();
  return (
    <div>
      <span data-testid="mode">{mode}</span>
      <span data-testid="paper">{theme.palette.background.paper}</span>
      <button onClick={() => setMode('light')}>to-light</button>
    </div>
  );
}

let storage: Storage;

beforeEach(() => {
  storage = memoryStorage();
  vi.stubGlobal('localStorage', storage);
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

function renderProbe() {
  return render(
    <AppProviders>
      <Probe />
    </AppProviders>,
  );
}

describe('color mode plumbing', () => {
  it('starts in dark and themes the whole tree', () => {
    renderProbe();
    expect(screen.getByTestId('mode').textContent).toBe('dark');
    expect(screen.getByTestId('paper').textContent).toBe('#2d3748');
  });

  it('swaps the active theme and persists under the legacy "theme" key', () => {
    renderProbe();
    fireEvent.click(screen.getByRole('button', { name: 'to-light' }));
    expect(screen.getByTestId('mode').textContent).toBe('light');
    expect(screen.getByTestId('paper').textContent).toBe('#ffffff');
    expect(storage.getItem('theme')).toBe('light');
  });

  it('restores a stored preference on mount', () => {
    storage.setItem('theme', 'light');
    renderProbe();
    expect(screen.getByTestId('mode').textContent).toBe('light');
    expect(screen.getByTestId('paper').textContent).toBe('#ffffff');
  });
});

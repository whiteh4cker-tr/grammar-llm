// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { AppProviders } from '../AppProviders';
import { Settings } from './Settings';
import { DIALOG_SURFACE } from '../theme';
import { hexToRgb } from '../contrast';

const rgbOf = (hex: string) => `rgb(${hexToRgb(hex).join(', ')})`;

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

// `api.ts` snapshots `window.api` at import time, so the stub must be in place
// before this file's imports run — hence `vi.hoisted`.
const { apiStub } = vi.hoisted(() => {
  const apiStub = {
    getSettings: vi.fn(async () => ({ contextSize: 4096 })),
    setContextSize: vi.fn(async () => ({ contextSize: 8192 })),
    listModels: vi.fn(async () => [] as string[]),
  };
  (window as unknown as { api: typeof apiStub }).api = apiStub;
  return { apiStub };
});

let storage: Storage;

beforeEach(() => {
  storage = memoryStorage();
  vi.stubGlobal('localStorage', storage);
  apiStub.setContextSize.mockClear();
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

function renderSettings(onWordLevelChange = vi.fn(), onClose = vi.fn()) {
  render(
    <AppProviders>
      <Settings
        status={{ state: 'ready', modelName: 'm.gguf' }}
        wordLevelEnabled
        onWordLevelChange={onWordLevelChange}
        onClose={onClose}
      />
    </AppProviders>,
  );
  return { onWordLevelChange, onClose };
}

describe('Settings dialog', () => {
  it('shows every section', async () => {
    renderSettings();
    expect(screen.getByText('Settings')).toBeTruthy();
    expect(screen.getByText('Theme')).toBeTruthy();
    expect(screen.getByText('Word-level corrections')).toBeTruthy();
    await waitFor(() => expect(screen.getByDisplayValue('4096')).toBeTruthy());
  });

  // The dialog surface is where most of the app's small text lives; the theme
  // override must actually reach the rendered paper, not just the token.
  it('paints the dialog surface with the accessible dark background', async () => {
    render(
      <AppProviders>
        <Settings
          status={{ state: 'ready', modelName: 'm.gguf' }}
          wordLevelEnabled
          onWordLevelChange={vi.fn()}
          onClose={vi.fn()}
        />
      </AppProviders>,
    );
    await waitFor(() => expect(screen.getByText('Settings')).toBeTruthy());
    // The dialog renders through a portal, so it lives outside the render container.
    const paper = document.querySelector('.MuiDialog-paper') as HTMLElement;
    expect(getComputedStyle(paper).backgroundColor).toBe(rgbOf(DIALOG_SURFACE.dark));
  });

  it('toggles word-level corrections and the color scheme', async () => {
    const { onWordLevelChange } = renderSettings();
    await waitFor(() => expect(screen.getByDisplayValue('4096')).toBeTruthy());

    fireEvent.click(screen.getByRole('switch', { name: 'Word-level corrections' }));
    expect(onWordLevelChange).toHaveBeenCalledWith(false);

    fireEvent.click(screen.getByRole('button', { name: /Light/ }));
    expect(storage.getItem('theme')).toBe('light');
  });

  it('rejects an out-of-range context size before calling the backend', async () => {
    renderSettings();
    const field = await screen.findByDisplayValue('4096');
    fireEvent.change(field, { target: { value: '100' } });
    fireEvent.click(screen.getByRole('button', { name: 'Apply' }));
    expect(screen.getByText('Context size must be an integer between 256 and 131072.')).toBeTruthy();
    expect(apiStub.setContextSize).not.toHaveBeenCalled();
  });

  it('saves a valid context size and confirms it', async () => {
    renderSettings();
    const field = await screen.findByDisplayValue('4096');
    fireEvent.change(field, { target: { value: '8192' } });
    fireEvent.click(screen.getByRole('button', { name: 'Apply' }));
    await waitFor(() => expect(apiStub.setContextSize).toHaveBeenCalledWith({ contextSize: 8192 }));
    await waitFor(() => expect(screen.getByText(/Saved — the new context size applies/)).toBeTruthy());
  });

  it('dismisses the saved confirmation on its own', async () => {
    vi.useFakeTimers();
    renderSettings();
    // Let the initial getSettings() poll land.
    await act(async () => {
      await Promise.resolve();
    });

    fireEvent.change(screen.getByDisplayValue('4096'), { target: { value: '8192' } });
    fireEvent.click(screen.getByRole('button', { name: 'Apply' }));
    await act(async () => {
      await Promise.resolve();
    });
    expect(screen.getByText(/Saved — the new context size applies/)).toBeTruthy();

    // The old note never went away; this one does (4s hide + exit transition).
    await act(async () => {
      await vi.advanceTimersByTimeAsync(4500);
    });
    expect(screen.queryByText(/Saved —/)).toBeNull();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(5000);
    });
    expect(screen.queryByText(/Saved —/)).toBeNull();
    vi.useRealTimers();
  });

  it('closes from the header button', async () => {
    const { onClose } = renderSettings();
    await waitFor(() => expect(screen.getByDisplayValue('4096')).toBeTruthy());
    fireEvent.click(screen.getByRole('button', { name: 'Close settings' }));
    expect(onClose).toHaveBeenCalled();
  });
});

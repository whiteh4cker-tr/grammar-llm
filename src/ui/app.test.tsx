// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { DownloadProgress, ModelStatus } from '../electron/ipc-types';

// jsdom has no ResizeObserver; the editor uses it to re-measure the mirror.
globalThis.ResizeObserver = class {
  observe() {}
  unobserve() {}
  disconnect() {}
} as unknown as typeof ResizeObserver;

const MB = 1024 * 1024;

const { status, installed, listeners } = vi.hoisted(() => {
  const status: { current: ModelStatus } = { current: { state: 'ready', modelName: 'a.gguf' } };
  const installed: { current: string[] } = { current: ['a.gguf'] };
  const listeners = new Set<(p: DownloadProgress) => void>();
  const apiStub = {
    modelStatus: vi.fn(async () => status.current),
    getWordLevelCorrection: vi.fn(async () => ({ enabled: true })),
    setWordLevelCorrection: vi.fn(async () => ({ enabled: true })),
    listModels: vi.fn(async () => installed.current),
    getSettings: vi.fn(async () => ({ contextSize: 4096 })),
    setContextSize: vi.fn(async () => ({ contextSize: 4096 })),
    onDownloadProgress: vi.fn((cb: (p: DownloadProgress) => void) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    }),
    cancelDownload: vi.fn(async () => {}),
    downloadModel: vi.fn((): Promise<void> => new Promise<void>(() => {})),
    correct: vi.fn(async () => ({ suggestions: [], correctedText: '' })),
  };
  (window as unknown as { api: typeof apiStub }).api = apiStub;
  return { status, installed, listeners };
});

const { AppProviders } = await import('./AppProviders');
const { default: App } = await import('./App');

const editorOpen = () => screen.queryByRole('button', { name: /check grammar/i }) !== null;
const gateOpen = () => screen.queryByText(/No model detected/) !== null;

const emitProgress = (p: DownloadProgress) => listeners.forEach((cb) => cb(p));

/** Let the immediate `modelStatus()` poll land. */
async function flush() {
  await act(async () => {
    await Promise.resolve();
  });
}

/** Advance past one status poll. */
async function nextPoll() {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(2000);
  });
}

function renderApp() {
  return render(
    <AppProviders>
      <App />
    </AppProviders>,
  );
}

const openSettings = () => fireEvent.click(screen.getByRole('button', { name: 'Settings' }));

beforeEach(() => {
  status.current = { state: 'ready', modelName: 'a.gguf' };
  installed.current = ['a.gguf'];
  listeners.clear();
  vi.useFakeTimers();
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

describe('App gating', () => {
  it('does not tear the editor down when a second model starts downloading', async () => {
    renderApp();
    await flush();
    expect(editorOpen()).toBe(true);

    // What the main process reports mid-download: state flips to `downloading`
    // and modelName points at the file being fetched.
    status.current = { state: 'downloading', modelName: 'b.gguf' };
    await nextPoll();

    expect(editorOpen()).toBe(true);
    expect(gateOpen()).toBe(false);
  });

  it('shows the gate when nothing is installed', async () => {
    status.current = { state: 'missing' };
    renderApp();
    await flush();
    expect(gateOpen()).toBe(true);
    expect(editorOpen()).toBe(false);
  });

  it('shows the gate when the loaded model disappears', async () => {
    renderApp();
    await flush();
    expect(editorOpen()).toBe(true);

    status.current = { state: 'missing' };
    await nextPoll();

    expect(gateOpen()).toBe(true);
    expect(editorOpen()).toBe(false);
  });

  it('reports progress on the first-run gate, where nothing is loaded yet', async () => {
    status.current = { state: 'missing' };
    installed.current = [];
    renderApp();
    await flush();
    expect(gateOpen()).toBe(true);

    fireEvent.click(screen.getByRole('button', { name: 'Download GRMR-V3-G4B-Q4_K_M' }));
    act(() => emitProgress({ percent: 30, transferred: 30 * MB, total: 100 * MB }));

    expect(screen.getByText(/Downloading GRMR-V3-G4B-Q4_K_M\.gguf — 30% \(30 MB of 100 MB\)/)).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Cancel' })).toBeTruthy();
  });
});

describe('download visibility in Settings', () => {
  it('shows the in-flight download in the dialog and survives closing it', async () => {
    renderApp();
    await flush();
    openSettings();

    fireEvent.click(screen.getByRole('button', { name: 'Download GRMR-V3-G4B-Q4_K_M' }));
    act(() => emitProgress({ percent: 25, transferred: 25 * MB, total: 100 * MB }));

    const whileOpen = screen.getAllByText(/Downloading GRMR-V3-G4B-Q4_K_M\.gguf — 25%/);
    expect(whileOpen.length).toBeGreaterThanOrEqual(2); // dialog banner + gate section

    // Close the dialog: the download keeps running and stays visible.
    fireEvent.click(screen.getByRole('button', { name: 'Close settings' }));
    expect(editorOpen()).toBe(true);

    openSettings();
    expect(
      screen.getAllByText(/Downloading GRMR-V3-G4B-Q4_K_M\.gguf — 25%/).length,
    ).toBeGreaterThanOrEqual(1);
  });
});

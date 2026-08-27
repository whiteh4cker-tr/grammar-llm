// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { DownloadProgress } from '../electron/ipc-types';

// `api.ts` snapshots `window.api` at import time, so the stub has to be in
// place before this file's imports run — hence `vi.hoisted`.
const { apiStub, listeners } = vi.hoisted(() => {
  const listeners = new Set<(p: DownloadProgress) => void>();
  const apiStub = {
    onDownloadProgress: vi.fn((cb: (p: DownloadProgress) => void) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    }),
    downloadModel: vi.fn((): Promise<void> => Promise.resolve()),
    cancelDownload: vi.fn(async () => {}),
  };
  (window as unknown as { api: typeof apiStub }).api = apiStub;
  return { apiStub, listeners };
});

const { DownloadProvider } = await import('./DownloadProvider');
const { formatDownload, useDownloadTracker } = await import('./downloadState');

function emit(progress: DownloadProgress) {
  listeners.forEach((cb) => cb(progress));
}

let deferred: { promise: Promise<void>; resolve: () => void };

function Probe() {
  const { active, track, cancel } = useDownloadTracker();
  return (
    <div>
      <span data-testid="state">{active ? formatDownload(active) : 'idle'}</span>
      <button onClick={() => void track('GRMR-Q8.gguf', () => apiStub.downloadModel())}>start</button>
      <button onClick={() => void cancel()}>cancel</button>
    </div>
  );
}

/** `showConsumer={false}` models the Settings dialog closing (or the app
 *  switching screens) while the provider — and the download — keep living. */
function Harness({ showConsumer = true }: { showConsumer?: boolean }) {
  return <DownloadProvider>{showConsumer ? <Probe /> : <span data-testid="bare" />}</DownloadProvider>;
}

const state = () => screen.getByTestId('state').textContent;

beforeEach(() => {
  listeners.clear();
  deferred = { promise: Promise.resolve(), resolve: () => {} };
  deferred.promise = new Promise<void>((resolve) => {
    deferred.resolve = resolve;
  });
  apiStub.downloadModel.mockImplementation(() => deferred.promise);
  apiStub.cancelDownload.mockClear();
});

afterEach(() => {
  cleanup();
});

describe('download tracker', () => {
  it('tracks a download from start to finish', async () => {
    render(<Harness />);
    expect(state()).toBe('idle');

    fireEvent.click(screen.getByRole('button', { name: 'start' }));
    expect(state()).toBe('Downloading GRMR-Q8.gguf');

    act(() => emit({ percent: 42, transferred: 44040192, total: 104857600 }));
    expect(state()).toBe('Downloading GRMR-Q8.gguf — 42% (42 MB of 100 MB)');

    await act(async () => {
      deferred.resolve();
    });
    expect(state()).toBe('idle');
  });

  it('keeps reporting after the component that started it unmounts', () => {
    const { rerender } = render(<Harness />);
    fireEvent.click(screen.getByRole('button', { name: 'start' }));
    act(() => emit({ percent: 12, transferred: 12582912, total: 104857600 }));

    rerender(<Harness showConsumer={false} />);
    rerender(<Harness showConsumer={true} />);

    expect(state()).toContain('GRMR-Q8.gguf');
    expect(state()).toContain('12%');
  });

  it('reports a download whose progress arrives before anyone claims it', () => {
    render(<Harness />);
    act(() => emit({ percent: 5, transferred: 5242880, total: 104857600 }));
    expect(state()).toContain('Downloading model');
    expect(state()).toContain('5%');
  });

  it('clears on cancel even while the request is pending', async () => {
    render(<Harness />);
    fireEvent.click(screen.getByRole('button', { name: 'start' }));
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'cancel' }));
    });
    expect(state()).toBe('idle');
    expect(apiStub.cancelDownload).toHaveBeenCalled();
  });

  it('unsubscribes from progress events on unmount', async () => {
    const { unmount } = render(<Harness />);
    expect(listeners.size).toBe(1);
    await act(async () => {
      unmount();
    });
    expect(listeners.size).toBe(0);
  });
});

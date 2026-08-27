// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { CorrectionResponse } from '../../electron/core/types';

// jsdom has no ResizeObserver; GrammarApp only uses it to re-measure the mirror.
globalThis.ResizeObserver = class {
  observe() {}
  unobserve() {}
  disconnect() {}
} as unknown as typeof ResizeObserver;

/** Fresh object per call: the component shifts offsets in place-free copies, but
 *  sharing one literal across tests would couple them. */
function response(): CorrectionResponse {
  return {
    suggestions: [
      {
        original: 'helo world',
        corrected: 'hello world',
        sentence: 'helo world',
        start_index: 0,
        end_index: 10,
        original_highlighted: '<span class="error-word">helo</span> world',
        corrected_highlighted: '<span class="corrected-word">hello</span> world',
        wordFixes: [{ original: 'helo', corrected: 'hello', start: 0, end: 4 }],
      },
    ],
    correctedText: 'hello world',
  };
}

// `src/ui/api.ts` snapshots `window.api` at import time, so the stub has to
// exist before GrammarApp (and thus api.ts) is imported.
const apiStub = {
  correct: vi.fn(async () => response()),
  applySuggestion: vi.fn(async () => ({ correctedText: 'hello world' })),
};
(window as unknown as { api: typeof apiStub }).api = apiStub;

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  apiStub.correct.mockClear();
  apiStub.applySuggestion.mockClear();
});

/** Check the text, then hover the highlight so the fix popup opens. */
async function openFixPopup() {
  const { default: GrammarApp } = await import('./GrammarApp');
  const { container } = render(<GrammarApp onOpenSettings={() => {}} wordLevelEnabled />);
  const textarea = container.querySelector('textarea') as HTMLTextAreaElement;
  fireEvent.change(textarea, { target: { value: 'helo world' } });
  fireEvent.click(screen.getByRole('button', { name: /check grammar/i }));
  await waitFor(() => expect(container.querySelector('.wfix')).toBeTruthy());

  // jsdom reports a zero rect for every element, so a pointer at the origin
  // hit-tests onto the first highlight span.
  const wrap = textarea.parentElement as HTMLElement;
  fireEvent.mouseMove(wrap, { clientX: 0, clientY: 0 });
  const popupButton = await screen.findByRole('button', { name: 'hello' });
  return { container, textarea, wrap, popupButton };
}

describe('word-fix popup', () => {
  it('writes the fix into the editor and keeps the overlay in sync', async () => {
    const { container, popupButton } = await openFixPopup();

    // A real click: press, then release on the same element.
    fireEvent.mouseDown(popupButton);
    fireEvent.mouseUp(popupButton);
    fireEvent.click(popupButton);

    // The editor text must carry the fix; if it does not, `lastCheckedText`
    // diverges from `text`, the overlay goes dark and every highlight vanishes.
    await waitFor(() =>
      expect((container.querySelector('textarea') as HTMLTextAreaElement).value).toBe('hello world'),
    );

    // Only the fixed sentence is re-checked.
    await waitFor(() => expect(apiStub.correct).toHaveBeenCalledTimes(2), { timeout: 2000 });
    expect(apiStub.correct).toHaveBeenLastCalledWith('hello world');
  });

  it('survives 500ms without the pointer, then closes', async () => {
    const { wrap } = await openFixPopup();
    vi.useFakeTimers();

    // Pointer leaves the word (no highlight under it) — the popup must linger.
    fireEvent.mouseMove(wrap, { clientX: 5000, clientY: 5000 });
    expect(screen.queryByRole('button', { name: 'hello' })).toBeTruthy();

    act(() => {
      vi.advanceTimersByTime(499);
    });
    expect(screen.queryByRole('button', { name: 'hello' })).toBeTruthy();

    act(() => {
      vi.advanceTimersByTime(2);
    });
    expect(screen.queryByRole('button', { name: 'hello' })).toBeNull();
  });

  it('confirms an applied suggestion, then hides the confirmation', async () => {
    const { default: GrammarApp } = await import('./GrammarApp');
    const { container } = render(<GrammarApp onOpenSettings={() => {}} wordLevelEnabled />);
    const textarea = container.querySelector('textarea') as HTMLTextAreaElement;
    fireEvent.change(textarea, { target: { value: 'helo world' } });
    fireEvent.click(screen.getByRole('button', { name: /check grammar/i }));
    await waitFor(() => expect(container.querySelector('.wfix')).toBeTruthy());

    // Fake timers only after the async check has settled: the toast timer has to
    // be created under the fake clock for this to be observable.
    vi.useFakeTimers();
    fireEvent.click(screen.getByRole('button', { name: 'Apply' }));
    await act(async () => {
      await Promise.resolve();
    });
    expect(screen.getByText(/Applied correction for helo world/)).toBeTruthy();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(3200);
    });
    expect(screen.queryByText(/Applied correction for/)).toBeNull();
  });

  it('stays open when the pointer moves onto the popup within the grace period', async () => {
    const { wrap, popupButton } = await openFixPopup();
    vi.useFakeTimers();

    fireEvent.mouseMove(wrap, { clientX: 5000, clientY: 5000 }); // schedules dismissal
    fireEvent.mouseMove(popupButton, { clientX: 1, clientY: 1 }); // pointer reaches the popup

    act(() => {
      vi.advanceTimersByTime(1500);
    });
    expect(screen.queryByRole('button', { name: 'hello' })).toBeTruthy();
  });
});

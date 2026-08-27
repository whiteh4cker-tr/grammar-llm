// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { CorrectionResponse, WordFix } from '../../electron/core/types';

// jsdom has no ResizeObserver; GrammarApp only uses it to re-measure the mirror.
globalThis.ResizeObserver = class {
  observe() {}
  unobserve() {}
  disconnect() {}
} as unknown as typeof ResizeObserver;

const TEXT = 'I run fast.';
const INSERT_VERY: WordFix = { original: '', corrected: 'very ', start: 6, end: 6 };

function response(wordFixes: WordFix[]): CorrectionResponse {
  return {
    suggestions: [
      {
        original: TEXT,
        corrected: 'I run very fast.',
        sentence: 'Sentence 1',
        start_index: 0,
        end_index: TEXT.length,
        original_highlighted: TEXT,
        corrected_highlighted: 'I run very fast.',
        wordFixes,
      },
    ],
    correctedText: 'I run very fast.',
  };
}

// `src/ui/api.ts` snapshots `window.api` at import time, so the stub has to
// exist before GrammarApp (and thus api.ts) is imported.
const apiStub = {
  correct: vi.fn(async () => response([{ original: '', corrected: 'very ', start: 6, end: 6 }])),
  applySuggestion: vi.fn(async () => ({ correctedText: 'I run very fast.' })),
};
(window as unknown as { api: typeof apiStub }).api = apiStub;

afterEach(() => {
  cleanup();
  apiStub.correct.mockClear();
});

async function checkWith(wordFixes: WordFix[], text: string = TEXT) {
  apiStub.correct.mockResolvedValueOnce(response(wordFixes));
  const { default: GrammarApp } = await import('./GrammarApp');
  const { container } = render(<GrammarApp onOpenSettings={() => {}} wordLevelEnabled />);
  const textarea = container.querySelector('textarea') as HTMLTextAreaElement;
  fireEvent.change(textarea, { target: { value: text } });
  fireEvent.click(screen.getByRole('button', { name: /check grammar/i }));
  await waitFor(() => expect(container.querySelector('.wfix')).toBeTruthy());
  const mirror = [...textarea.parentElement!.children].find((c) => c !== textarea) as HTMLElement;
  return { container, textarea, mirror };
}

describe('editor mirror alignment', () => {
  // The caret is drawn by the textarea at the real character offset while the
  // user reads the mirror. Any character the mirror adds (or drops) shifts the
  // visible text away from the caret, so the caret looks like it stopped short
  // of the end of the text.
  it('renders exactly the editor text, even with an insertion fix', async () => {
    const { textarea, mirror } = await checkWith([INSERT_VERY]);
    expect(mirror.textContent).toBe(textarea.value);
  });

  // The textarea paints a line for a trailing newline; `white-space: pre-wrap`
  // collapses the last one, which would leave the mirror a line short and put
  // the caret on the wrong visible line.
  it('paints the trailing blank line the textarea paints', async () => {
    const { textarea, mirror } = await checkWith([INSERT_VERY], 'I run fast.\n');
    expect(mirror.textContent).toBe(`${textarea.value}\n`);
  });

  it('renders exactly the editor text with replacement and deletion fixes', async () => {
    const { textarea, mirror } = await checkWith([
      { original: 'run', corrected: 'sprint', start: 2, end: 5 },
      { original: 'fast', corrected: '', start: 6, end: 10 },
    ]);
    expect(mirror.textContent).toBe(textarea.value);
  });

  it('keeps the insertion marker out of the text flow but hoverable', async () => {
    const { mirror } = await checkWith([{ original: '', corrected: 'very ', start: 6, end: 6 }]);
    const marker = mirror.querySelector('.wfix-insert');
    expect(marker).toBeTruthy();
    expect(marker!.textContent).toBe('');
    expect(marker!.getAttribute('data-fix-start')).toBe('6');
  });
});

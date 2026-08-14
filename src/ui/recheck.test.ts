import { describe, it, expect } from 'vitest';
import { applyWordFixToText, mergeSentenceRecheck, rebaseRecheckedSuggestions, shiftApplied } from './recheck';
import type { CorrectionResponse, Suggestion, WordFix } from '../electron/core/types';

function makeSuggestion(
  partial: Partial<Suggestion> & { original: string; corrected: string; start_index: number; end_index: number },
): Suggestion {
  return {
    sentence: 'Sentence 1',
    original_highlighted: '',
    corrected_highlighted: '',
    wordFixes: [],
    ...partial,
  };
}

describe('applyWordFixToText', () => {
  const text = 'She dont like the apples. this is a bad sentence';
  const s0 = makeSuggestion({
    original: 'She dont like the apples.',
    corrected: "She doesn't like the apples.",
    start_index: 0,
    end_index: 25,
  });
  const s1 = makeSuggestion({
    original: 'this is a bad sentence',
    corrected: 'This is a bad sentence',
    start_index: 26,
    end_index: 48,
  });

  it('applies the fix and locates its parent sentence', () => {
    const fix: WordFix = { original: 'dont', corrected: "doesn't", start: 4, end: 8 };
    const result = applyWordFixToText(text, fix, [s0, s1]);
    expect(result.text).toBe("She doesn't like the apples. this is a bad sentence");
    expect(result.parentIndex).toBe(0);
    expect(result.sentenceStart).toBe(0);
    expect(result.sentenceText).toBe("She doesn't like the apples.");
    expect(result.delta).toBe(3); // "doesn't" (7) vs "dont" (4)
  });

  it('handles a fix in a later sentence', () => {
    const fix: WordFix = { original: 'this', corrected: 'This', start: 26, end: 30 };
    const result = applyWordFixToText(text, fix, [s0, s1]);
    expect(result.text).toBe('She dont like the apples. This is a bad sentence');
    expect(result.parentIndex).toBe(1);
    expect(result.sentenceStart).toBe(26);
    expect(result.sentenceText).toBe('This is a bad sentence');
    expect(result.delta).toBe(0);
  });

  it('returns parentIndex -1 when no suggestion contains the fix', () => {
    const fix: WordFix = { original: ' ', corrected: '!', start: 25, end: 26 };
    const result = applyWordFixToText(text, fix, [s0, s1]);
    expect(result.parentIndex).toBe(-1);
    expect(result.text).toBe(text.slice(0, 25) + '!' + text.slice(26));
  });
});

describe('mergeSentenceRecheck', () => {
  const s0 = makeSuggestion({ original: 'Bad one.', corrected: 'Good one.', start_index: 0, end_index: 8 });
  const s1 = makeSuggestion({
    original: 'Worse one.',
    corrected: 'Better one.',
    start_index: 9,
    end_index: 19,
    wordFixes: [{ original: 'Worse', corrected: 'Better', start: 9, end: 14 }],
  });
  const s2 = makeSuggestion({ original: 'Last one.', corrected: 'Final one.', start_index: 20, end_index: 29 });

  it('replaces the parent with rebased recheck results and shifts the suffix', () => {
    const rechecked: CorrectionResponse = {
      correctedText: 'Best one.',
      suggestions: [
        makeSuggestion({
          original: 'Best one.',
          corrected: 'Best one.',
          start_index: 2,
          end_index: 6,
          wordFixes: [{ original: 'one', corrected: 'two', start: 4, end: 7 }],
        }),
      ],
    };
    const { suggestions, applied } = mergeSentenceRecheck([s0, s1, s2], new Set([2]), 1, 9, 1, rechecked);
    expect(suggestions.map((s) => s.original)).toEqual(['Bad one.', 'Best one.', 'Last one.']);
    expect(suggestions[1].start_index).toBe(11); // sentenceStart 9 + 2
    expect(suggestions[1].end_index).toBe(15); // sentenceStart 9 + 6
    expect(suggestions[1].wordFixes[0].start).toBe(13); // sentenceStart 9 + 4
    expect(suggestions[1].wordFixes[0].end).toBe(16); // sentenceStart 9 + 7
    expect(suggestions[2].start_index).toBe(21); // 20 + delta 1
    // applied mark for the suffix suggestion (index 2) survives the merge.
    expect(applied.has(2)).toBe(true);
  });

  it('handles a clean recheck (no suggestions) and shifts applied marks', () => {
    const rechecked: CorrectionResponse = { correctedText: 'Best one.', suggestions: [] };
    const { suggestions, applied } = mergeSentenceRecheck([s0, s1, s2], new Set([0, 2]), 1, 9, 2, rechecked);
    expect(suggestions.map((s) => s.original)).toEqual(['Bad one.', 'Last one.']);
    expect(suggestions[1].start_index).toBe(22); // 20 + delta 2
    expect(applied.has(0)).toBe(true); // prefix mark unchanged
    expect(applied.has(1)).toBe(true); // suffix mark: 2 -> 2 + (0 - 1)
  });
});

describe('rebaseRecheckedSuggestions', () => {
  it('rebases offsets and word fixes to the full text', () => {
    const rechecked: CorrectionResponse = {
      correctedText: 'Best one.',
      suggestions: [
        makeSuggestion({
          original: 'Best one.',
          corrected: 'Best one.',
          start_index: 2,
          end_index: 6,
          wordFixes: [{ original: 'one', corrected: 'two', start: 4, end: 7 }],
        }),
      ],
    };
    const rebased = rebaseRecheckedSuggestions(rechecked, 9);
    expect(rebased[0].start_index).toBe(11);
    expect(rebased[0].end_index).toBe(15);
    expect(rebased[0].wordFixes[0]).toEqual({ original: 'one', corrected: 'two', start: 13, end: 16 });
  });
});

describe('shiftApplied', () => {
  it('shifts marks at/after the threshold', () => {
    expect([...shiftApplied(new Set([0, 2, 4]), 2, -1)].sort()).toEqual([0, 1, 3]);
    expect([...shiftApplied(new Set([0, 1]), 1, 2)].sort()).toEqual([0, 3]);
  });
});

describe('provisional-then-final word fix flow', () => {
  it('keeps other suggestions during the re-check and merges the result', () => {
    const s0 = makeSuggestion({ original: 'Bad one.', corrected: 'Good one.', start_index: 0, end_index: 8 });
    const s1 = makeSuggestion({ original: 'Worse one.', corrected: 'Better one.', start_index: 9, end_index: 19 });
    const s2 = makeSuggestion({ original: 'Last one.', corrected: 'Final one.', start_index: 20, end_index: 29 });

    // Fix applied in sentence 2: parent removed, suffix shifted, mark remapped.
    const provisional = mergeSentenceRecheck([s0, s1, s2], new Set([2]), 1, 9, 1, {
      correctedText: '',
      suggestions: [],
    });
    expect(provisional.suggestions.map((s) => s.original)).toEqual(['Bad one.', 'Last one.']);
    expect(provisional.suggestions[1].start_index).toBe(21); // 20 + delta 1
    expect([...provisional.applied]).toEqual([1]); // mark 2 -> 2 + (0 - 1)

    // Re-check resolves with one suggestion: inserted at the parent position.
    const rechecked: CorrectionResponse = {
      correctedText: 'Best one.',
      suggestions: [
        makeSuggestion({ original: 'Best one.', corrected: 'Best one.', start_index: 2, end_index: 6 }),
      ],
    };
    const rebased = rebaseRecheckedSuggestions(rechecked, 9);
    const finalSuggestions = [
      ...provisional.suggestions.slice(0, 1),
      ...rebased,
      ...provisional.suggestions.slice(1),
    ];
    const finalApplied = shiftApplied(provisional.applied, 1, rebased.length);
    expect(finalSuggestions.map((s) => s.original)).toEqual(['Bad one.', 'Best one.', 'Last one.']);
    expect(finalSuggestions[1].start_index).toBe(11); // sentenceStart 9 + 2
    expect(finalSuggestions[2].start_index).toBe(21);
    expect([...finalApplied]).toEqual([2]);
  });
});

describe('applyWordFixToText with insertions and deletions', () => {
  it('inserts text at a zero-width fix point', () => {
    const s0 = makeSuggestion({ original: 'Hello John', corrected: 'Hello, John', start_index: 0, end_index: 10 });
    const fix: WordFix = { original: '', corrected: ',', start: 5, end: 5 };
    const result = applyWordFixToText('Hello John', fix, [s0]);
    expect(result.text).toBe('Hello, John');
    expect(result.parentIndex).toBe(0);
    expect(result.sentenceText).toBe('Hello, John');
    expect(result.delta).toBe(1);
  });

  it('deletes a word via an empty correction', () => {
    const s0 = makeSuggestion({ original: 'the the cat', corrected: 'the cat', start_index: 0, end_index: 11 });
    const fix: WordFix = { original: 'the ', corrected: '', start: 4, end: 8 };
    const result = applyWordFixToText('the the cat', fix, [s0]);
    expect(result.text).toBe('the cat');
    expect(result.parentIndex).toBe(0);
    expect(result.sentenceText).toBe('the cat');
    expect(result.delta).toBe(-4);
  });

  it('assigns an insertion on a sentence boundary to the sentence ending there', () => {
    const s0 = makeSuggestion({ original: 'One.', corrected: 'One!', start_index: 0, end_index: 4 });
    const s1 = makeSuggestion({ original: 'Two.', corrected: 'Two.', start_index: 5, end_index: 9 });
    const fix: WordFix = { original: '', corrected: '!', start: 4, end: 4 };
    const result = applyWordFixToText('One. Two.', fix, [s0, s1]);
    expect(result.parentIndex).toBe(0);
    expect(result.sentenceText).toBe('One.!');
  });
});

describe('applyWordFixToText defensive guards', () => {
  it('snaps an in-word insertion to the word end before applying', () => {
    const s0 = makeSuggestion({
      original: 'The team were very excited to receive the new software, but it\'s performance have been poorly.',
      corrected: "The team was very excited to receive the new software, but, it's performance have been poorly.",
      start_index: 0,
      end_index: 94,
    });
    const bad: WordFix = { original: '', corrected: ',', start: 57, end: 57 }; // inside 'but' (56..59)
    const result = applyWordFixToText(
      'The team were very excited to receive the new software, but it\'s performance have been poorly.',
      bad,
      [s0],
    );
    expect(result.text).toBe("The team were very excited to receive the new software, but, it's performance have been poorly.");
    expect(result.sentenceText).toBe("The team were very excited to receive the new software, but, it's performance have been poorly.");
  });

  it('falls back to the nearest occurrence for a stale span', () => {
    const s0 = makeSuggestion({ original: 'the the cat', corrected: 'the cat', start_index: 0, end_index: 11 });
    const stale: WordFix = { original: 'the ', corrected: '', start: 100, end: 104 };
    const result = applyWordFixToText('the the cat', stale, [s0]);
    expect(result.text).toBe('the cat');
  });

  it('refuses to apply when the original text no longer exists', () => {
    const s0 = makeSuggestion({ original: 'the cat', corrected: 'the cat', start_index: 0, end_index: 7 });
    const ghost: WordFix = { original: 'zzzz', corrected: 'aaaa', start: 2, end: 6 };
    const result = applyWordFixToText('the cat', ghost, [s0]);
    expect(result.text).toBe('the cat');
    expect(result.parentIndex).toBe(-1);
  });
});

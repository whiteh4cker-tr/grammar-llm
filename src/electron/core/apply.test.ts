import { describe, it, expect } from 'vitest';
import { applySuggestion, applySuggestionsBulk } from './apply';
import type { Suggestion } from './types';

function makeSuggestion(partial: Partial<Suggestion> & { original: string; corrected: string }): Suggestion {
  return {
    sentence: 'Sentence 1',
    start_index: 0,
    end_index: partial.original.length,
    original_highlighted: '',
    corrected_highlighted: '',
    ...partial,
  };
}

describe('applySuggestion', () => {
  it('replaces the indexed span', () => {
    const s = makeSuggestion({ original: 'dont', corrected: "don't", start_index: 4, end_index: 8 });
    expect(applySuggestion('She dont go.', 0, [s])).toBe("She don't go.");
  });

  it('falls back to nearest occurrence when span mismatches', () => {
    // approx start 99 -> nearest occurrence is the second 'dont' (Python: min by |sp[0]-start|)
    const s = makeSuggestion({ original: 'dont', corrected: "don't", start_index: 99, end_index: 103 });
    expect(applySuggestion('dont dont.', 0, [s])).toBe("dont don't.");
  });

  it('leaves text unchanged when original not found', () => {
    const s = makeSuggestion({ original: 'zzz', corrected: 'aaa' });
    expect(applySuggestion('nothing here.', 0, [s])).toBe('nothing here.');
  });

  it('throws on invalid index', () => {
    const s = makeSuggestion({ original: 'a', corrected: 'b' });
    expect(() => applySuggestion('a', 5, [s])).toThrow();
  });
});

describe('applySuggestionsBulk', () => {
  it('applies non-overlapping suggestions without index drift', () => {
    const s1 = makeSuggestion({ original: 'first', corrected: '1st', start_index: 0, end_index: 5 });
    const s2 = makeSuggestion({ original: 'third', corrected: '3rd', start_index: 13, end_index: 18 });
    expect(applySuggestionsBulk('first second third', [s1, s2])).toBe('1st second 3rd');
  });

  it('keeps rightmost replacement on overlap', () => {
    const s1 = makeSuggestion({ original: 'a b c', corrected: 'X', start_index: 0, end_index: 5 });
    const s2 = makeSuggestion({ original: 'b c', corrected: 'Y', start_index: 2, end_index: 5 });
    expect(applySuggestionsBulk('a b c', [s1, s2])).toBe('a Y');
  });

  it('skips invalid suggestions', () => {
    const s1 = makeSuggestion({ original: 'bad', corrected: 'good', start_index: -1, end_index: 3 });
    const s2 = makeSuggestion({ original: 'ok', corrected: 'fine', start_index: 0, end_index: 2 });
    expect(applySuggestionsBulk('ok', [s1, s2])).toBe('fine');
  });

  it('returns text unchanged for empty list', () => {
    expect(applySuggestionsBulk('hello', [])).toBe('hello');
  });
});

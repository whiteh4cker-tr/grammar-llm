import type { CorrectionResponse, Suggestion, WordFix } from '../electron/core/types';

export interface ApplyWordFixResult {
  /** Full text after the replacement. */
  text: string;
  /** Index of the suggestion whose sentence contains the fix, or -1. */
  parentIndex: number;
  /** Absolute start of that sentence in the (new) text. */
  sentenceStart: number;
  /** The sentence text with the fix applied — the only text that needs re-checking. */
  sentenceText: string;
  /** Length change introduced by the fix (corrected.length - span length). */
  delta: number;
}

/** Apply a word fix to the full text and locate the sentence it belongs to. */
export function applyWordFixToText(text: string, fix: WordFix, suggestions: Suggestion[]): ApplyWordFixResult {
  // Defensive: never apply an insertion inside a word (snap to the word's
  // end), and never apply a span that doesn't match the text at its position
  // (fall back to the nearest occurrence, then drop).
  let start = Math.min(fix.start, text.length);
  let end = Math.min(fix.end, text.length);
  if (fix.original === '') {
    // eslint-disable-next-line no-useless-escape -- the [ ] escapes are required here; unescaped [] matches nothing
    const wordChar = /[^\s.,!?;:'"()\[\]{}\u201C\u201D\u2018\u2019]/;
    if (start > 0 && start < text.length && wordChar.test(text[start - 1]) && wordChar.test(text[start])) {
      while (start < text.length && wordChar.test(text[start])) start++;
      end = start;
    }
  } else if (text.slice(start, end) !== fix.original) {
    let nearest = -1;
    let from = 0;
    while (from <= text.length - fix.original.length) {
      const idx = text.indexOf(fix.original, from);
      if (idx === -1) break;
      if (nearest === -1 || Math.abs(idx - fix.start) < Math.abs(nearest - fix.start)) nearest = idx;
      from = idx + 1;
    }
    if (nearest === -1) {
      return { text, parentIndex: -1, sentenceStart: 0, sentenceText: '', delta: 0 };
    }
    start = nearest;
    end = nearest + fix.original.length;
  }
  const replaced = text.slice(0, start) + fix.corrected + text.slice(end);
  const delta = fix.corrected.length - (end - start);

  // Insertions (zero-width fixes) may sit exactly on a sentence boundary
  // (e.g. a period added at the end of a sentence), so match their end
  // inclusively and prefer the sentence ENDING at the point.
  const parentIndex = suggestions.findIndex((s) =>
    start >= s.start_index &&
    (fix.original === '' ? start <= s.end_index : start < s.end_index),
  );
  if (parentIndex === -1) {
    return { text: replaced, parentIndex: -1, sentenceStart: 0, sentenceText: '', delta };
  }

  const parent = suggestions[parentIndex];
  const relStart = start - parent.start_index;
  const relEnd = end - parent.start_index;
  const sentenceText = parent.original.slice(0, relStart) + fix.corrected + parent.original.slice(relEnd);

  return {
    text: replaced,
    parentIndex,
    sentenceStart: parent.start_index,
    sentenceText,
    delta,
  };
}

/** Rebase re-check results (offsets relative to the sentence) onto the full text. */
export function rebaseRecheckedSuggestions(rechecked: CorrectionResponse, sentenceStart: number): Suggestion[] {
  return rechecked.suggestions.map((s) => ({
    ...s,
    start_index: sentenceStart + s.start_index,
    end_index: sentenceStart + s.end_index,
    wordFixes: s.wordFixes.map((w) => ({ ...w, start: sentenceStart + w.start, end: sentenceStart + w.end })),
  }));
}

/** Shift applied marks at/after `from` by `by` (e.g. after list insertions/removals). */
export function shiftApplied(applied: Set<number>, from: number, by: number): Set<number> {
  const next = new Set<number>();
  applied.forEach((j) => next.add(j >= from ? j + by : j));
  return next;
}

/**
 * Merge the re-check result of one sentence back into the full suggestion
 * list. The parent suggestion is replaced by the (rebased) re-check results;
 * suggestions after it get their offsets shifted by the fix's length delta;
 * applied marks are remapped to the new indices.
 *
 * Passing `{ suggestions: [] }` as the re-check result produces the
 * "provisional" state shown while the re-check is in flight: the parent
 * suggestion is removed, later suggestions are shifted, and the fixed
 * sentence contributes no suggestions until the re-check lands.
 */
export function mergeSentenceRecheck(
  suggestions: Suggestion[],
  applied: Set<number>,
  parentIndex: number,
  sentenceStart: number,
  delta: number,
  rechecked: CorrectionResponse,
): { suggestions: Suggestion[]; applied: Set<number> } {
  const rebased = rebaseRecheckedSuggestions(rechecked, sentenceStart);

  const prefix = suggestions.slice(0, parentIndex);
  const suffix = suggestions.slice(parentIndex + 1).map((s) => ({
    ...s,
    start_index: s.start_index + delta,
    end_index: s.end_index + delta,
    wordFixes: s.wordFixes.map((w) => ({ ...w, start: w.start + delta, end: w.end + delta })),
  }));

  return {
    suggestions: [...prefix, ...rebased, ...suffix],
    applied: shiftApplied(applied, parentIndex, rebased.length - 1),
  };
}

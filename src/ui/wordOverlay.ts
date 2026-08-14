import type { WordFix } from '../electron/core/types';

export interface Segment {
  text: string;
  isFix: boolean;
  fix?: WordFix;
}

/** Split text into plain/fix segments at the fix offsets. */
export function buildSegments(text: string, fixes: WordFix[]): Segment[] {
  if (text.length === 0) return [];
  const sorted = [...fixes].sort((a, b) => a.start - b.start);
  const segments: Segment[] = [];
  let cursor = 0;
  for (const fix of sorted) {
    if (fix.start > text.length) continue;
    if (fix.start === fix.end) {
      // Insertion point: zero-width segment rendered with the inserted text.
      if (fix.start < cursor) continue;
      if (fix.start > cursor) {
        segments.push({ text: text.slice(cursor, fix.start), isFix: false });
      }
      segments.push({ text: '', isFix: true, fix });
      cursor = fix.start;
      continue;
    }
    if (fix.end <= cursor) continue;
    if (fix.start > cursor) {
      segments.push({ text: text.slice(cursor, fix.start), isFix: false });
    }
    const end = Math.min(fix.end, text.length);
    segments.push({ text: text.slice(fix.start, end), isFix: true, fix });
    cursor = end;
  }
  if (cursor < text.length) {
    segments.push({ text: text.slice(cursor), isFix: false });
  }
  return segments;
}

// eslint-disable-next-line no-useless-escape -- the [ ] escapes are required here; unescaped [] matches nothing
const WORD_CHAR = /[^\s.,!?;:'"()\[\]{}\u201C\u201D\u2018\u2019]/;

function isWordChar(ch: string | undefined): boolean {
  return ch !== undefined && WORD_CHAR.test(ch);
}

/**
 * Defensive normalization of word fixes against the current text:
 * - Insertion points inside a word are snapped to the end of that word
 *   (an in-word comma like `bu,t` must never happen).
 * - Span fixes whose range no longer matches `fix.original` are re-located to
 *   the nearest occurrence; if none exists they are dropped.
 */
export function normalizeFixes(text: string, fixes: WordFix[]): WordFix[] {
  const out: WordFix[] = [];
  for (const fix of fixes) {
    if (fix.original === '') {
      let p = Math.min(fix.start, text.length);
      if (p > 0 && p < text.length && isWordChar(text[p - 1]) && isWordChar(text[p])) {
        // Inside a word — snap to its end.
        while (p < text.length && isWordChar(text[p])) p++;
      }
      out.push({ ...fix, start: p, end: p });
      continue;
    }
    const end = Math.min(fix.end, text.length);
    if (fix.start <= text.length && fix.start <= end && text.slice(fix.start, end) === fix.original) {
      out.push(fix);
      continue;
    }
    // Stale span: find the nearest occurrence of the original text.
    let nearest: [number, number] | null = null;
    let from = 0;
    while (from <= text.length - fix.original.length) {
      const idx = text.indexOf(fix.original, from);
      if (idx === -1) break;
      if (!nearest || Math.abs(idx - fix.start) < Math.abs(nearest[0] - fix.start)) {
        nearest = [idx, idx + fix.original.length];
      }
      from = idx + 1;
    }
    if (nearest) {
      out.push({ ...fix, start: nearest[0], end: nearest[1] });
    }
  }
  return out;
}

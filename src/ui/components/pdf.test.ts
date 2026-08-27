import { describe, it, expect } from 'vitest';
import { buildPdfDocument } from './pdf';
import { extractDraws, findCrossBaselineCollisions, rawContent } from './pdfTestHelpers';
import { highlightWordDifferences } from '../../electron/core/diff';
import type { Suggestion } from '../../electron/core/types';

const MM = 72 / 25.4;
const CONTENT_X = (20 + 24) * MM; // margin + label column
const LINE_H = 5.5 * MM; // row line height

function suggestion(original: string, corrected: string, sentence = 'Sentence 1'): Suggestion {
  const h = highlightWordDifferences(original, corrected);
  return {
    original,
    corrected,
    sentence,
    start_index: 0,
    end_index: original.length,
    original_highlighted: h.originalHighlighted,
    corrected_highlighted: h.correctedHighlighted,
    wordFixes: [],
  };
}

const WITH_NEWLINE = 'We are hoping that the company will soon release an update, which will have a good effect on its functionality.\nThe team were very excited to receive the new software, but its performance has been poor.';
const WITH_NEWLINE_FIXED = 'We are hoping that the company will soon release an update, which will have a good effect on its functionality.\nThe team was very excited to receive the new software, but its performance has been poor.';

describe('buildPdfDocument', () => {
  it('never overlaps text when a sentence contains a newline', () => {
    const doc = buildPdfDocument([suggestion(WITH_NEWLINE, WITH_NEWLINE_FIXED)], 72);
    expect(findCrossBaselineCollisions(extractDraws(doc))).toEqual([]);
  });

  it('starts a new line at the content margin after a newline', () => {
    const doc = buildPdfDocument([suggestion(WITH_NEWLINE, WITH_NEWLINE_FIXED)], 72);
    const draws = extractDraws(doc);
    const before = draws.find((d) => d.text === 'functionality.');
    const after = draws.find((d) => d.text === 'The');
    expect(before).toBeDefined();
    expect(after).toBeDefined();
    expect(after!.x).toBeCloseTo(CONTENT_X, 1);
    expect(before!.y - after!.y).toBeCloseTo(LINE_H, 1);
  });

  it('never hands a raw newline to jsPDF, which would draw it out of band', () => {
    const doc = buildPdfDocument(
      [suggestion('helo world.\nTeh next.', 'hello world.\nThe next.', 'Sentence 1.\nSentence 2.')],
      72,
    );
    // `T*` is how jsPDF advances a baseline itself — the layout code must do
    // all of its own line math instead, otherwise rows overlap each other.
    expect(rawContent(doc)).not.toMatch(/T\*/);
  });
});

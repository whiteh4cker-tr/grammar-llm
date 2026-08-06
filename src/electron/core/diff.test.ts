import { describe, it, expect } from 'vitest';
import { tokenize, highlightWordDifferences } from './diff';

describe('tokenize', () => {
  it('keeps punctuation and whitespace as separate tokens', () => {
    expect(tokenize('dont go.')).toEqual(['dont', ' ', 'go', '.']);
  });

  it('splits contractions on apostrophes', () => {
    expect(tokenize("doesn't")).toEqual(['doesn', "'", 't']);
  });
});

describe('highlightWordDifferences', () => {
  it('highlights replaced words in both directions', () => {
    const { originalHighlighted, correctedHighlighted } = highlightWordDifferences(
      'She dont like the apples. this is a bad sentence',
      "She doesn't like the apples. This is a bad sentence",
    );
    expect(originalHighlighted).toContain('<span class="error-word">dont</span>');
    expect(originalHighlighted).toContain('<span class="error-word">this</span>');
    expect(correctedHighlighted).toContain('<span class="corrected-word">doesn</span>');
    expect(correctedHighlighted).toContain('<span class="corrected-word">This</span>');
    expect(correctedHighlighted).toContain('like the apples');
  });

  it('highlights deleted words only in original', () => {
    const { originalHighlighted, correctedHighlighted } = highlightWordDifferences('a b c', 'a c');
    expect(originalHighlighted).toContain('<span class="error-word">b</span>');
    expect(correctedHighlighted).not.toContain('corrected-word');
  });

  it('preserves whitespace tokens unhighlighted', () => {
    const { originalHighlighted } = highlightWordDifferences('a\nb', 'a\nc');
    expect(originalHighlighted).toContain('\n');
    expect(originalHighlighted).not.toContain('error-word">\n<');
  });

  it('returns identical text for identical input', () => {
    const { originalHighlighted, correctedHighlighted } = highlightWordDifferences('same', 'same');
    expect(originalHighlighted).toBe('same');
    expect(correctedHighlighted).toBe('same');
  });
});

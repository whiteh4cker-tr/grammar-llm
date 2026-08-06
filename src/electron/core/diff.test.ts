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

describe('quote-aware diffing', () => {
  it('treats quote-style-only changes as equal (no spans, original chars kept)', () => {
    const result = highlightWordDifferences('He said \u201Chi\u201D.', 'He said "hi".');
    expect(result.originalHighlighted).toBe('He said \u201Chi\u201D.');
    expect(result.correctedHighlighted).toBe('He said \u201Chi\u201D.');
    expect(result.preservedCorrected).toBe('He said \u201Chi\u201D.');
    expect(result.originalHighlighted).not.toContain('error-word');
    expect(result.correctedHighlighted).not.toContain('corrected-word');
  });

  it('highlights only the real word change, not quote-adjacent words', () => {
    const result = highlightWordDifferences(
      'by \u201Cempovering\u201D users',
      'by "empowering" users',
    );
    expect(result.originalHighlighted).toBe('by \u201C<span class="error-word">empovering</span>\u201D users');
    expect(result.correctedHighlighted).toBe('by \u201C<span class="corrected-word">empowering</span>\u201D users');
    expect(result.preservedCorrected).toBe('by \u201Cempowering\u201D users');
  });

  it('treats curly apostrophes as equal to straight ones', () => {
    const result = highlightWordDifferences('It\u2019s fine', "It's fine");
    expect(result.originalHighlighted).toBe('It\u2019s fine');
    expect(result.preservedCorrected).toBe('It\u2019s fine');
  });

  it('preservedCorrected keeps corrected words with original quotes', () => {
    const result = highlightWordDifferences(
      'Future \u201CInteractive AI,\u201D works',
      'Future "Interactive AI," works',
    );
    expect(result.preservedCorrected).toBe('Future \u201CInteractive AI,\u201D works');
  });
});

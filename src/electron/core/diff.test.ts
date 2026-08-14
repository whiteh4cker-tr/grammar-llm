import { describe, it, expect } from 'vitest';
import { tokenize, highlightWordDifferences, extractWordFixes } from './diff';

describe('tokenize', () => {
  it('keeps punctuation and whitespace as separate tokens', () => {
    expect(tokenize('dont go.')).toEqual(['dont', ' ', 'go', '.']);
  });

  it('keeps contractions as single tokens', () => {
    expect(tokenize("doesn't")).toEqual(["doesn't"]);
  });

  it('keeps curly apostrophes inside words', () => {
    expect(tokenize('don\u2019t')).toEqual(['don\u2019t']);
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
    expect(correctedHighlighted).toContain('<span class="corrected-word">doesn\'t</span>');
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

describe('extractWordFixes', () => {
  it('pairs equal-length 1:1 replacements with absolute offsets', () => {
    expect(extractWordFixes('the cat sat', 'the dog sat', 10)).toEqual([
      { original: 'cat', corrected: 'dog', start: 14, end: 17 },
    ]);
  });

  it('joins unequal runs into a single fix spanning the run', () => {
    // "dont" (1 token) -> "doesn't" (3 tokens: doesn, ', t)
    expect(extractWordFixes('She dont like', "She doesn't like", 0)).toEqual([
      { original: 'dont', corrected: "doesn't", start: 4, end: 8 },
    ]);
  });

  it('joins multi-word removals into one fix', () => {
    expect(extractWordFixes('he is are', 'he was', 0)).toEqual([
      { original: 'is are', corrected: 'was', start: 3, end: 9 },
    ]);
  });

  it('skips pure deletions (no added tokens)', () => {
    // The separator space travels with the deleted word so the result stays
    // single-spaced.
    expect(extractWordFixes('a b c', 'a c', 0)).toEqual([
      { original: 'b ', corrected: '', start: 2, end: 4 },
    ]);
  });

  it('returns no fixes for identical text', () => {
    expect(extractWordFixes('same text', 'same text', 0)).toEqual([]);
  });

  it('treats quote-style-only changes as no fixes', () => {
    expect(extractWordFixes('He said \u201Chi\u201D.', 'He said "hi".', 0)).toEqual([]);
  });

  it('pairs a word adjacent to quotes as a single fix', () => {
    expect(extractWordFixes('by \u201Cempovering\u201D users', 'by \u201Cempowering\u201D users', 0)).toEqual([
      { original: 'empovering', corrected: 'empowering', start: 4, end: 14 },
    ]);
  });

  it('marks a pure insertion at its point', () => {
    expect(extractWordFixes('Hello John', 'Hello, John', 0)).toEqual([
      { original: '', corrected: ',', start: 5, end: 5 },
    ]);
  });

  it('marks an insertion at the end of the text', () => {
    expect(extractWordFixes('Hello John', 'Hello John!', 0)).toEqual([
      { original: '', corrected: '!', start: 10, end: 10 },
    ]);
  });

  it('marks a deleted word with an empty correction', () => {
    expect(extractWordFixes('the the cat', 'the cat', 0)).toEqual([
      { original: 'the ', corrected: '', start: 4, end: 8 },
    ]);
  });

  it('joins a multi-word deletion into one fix', () => {
    expect(extractWordFixes('a is are b', 'a b', 0)).toEqual([
      { original: 'is are ', corrected: '', start: 2, end: 9 },
    ]);
  });

  it('skips whitespace-only insertions and deletions', () => {
    expect(extractWordFixes('a b', 'a  b', 0)).toEqual([]);
    expect(extractWordFixes('a  b', 'a b', 0)).toEqual([]);
  });

  it('combines insertions, deletions and replacements in one pass', () => {
    const fixes = extractWordFixes('I like teh apples. the the end', 'I like the apples. the end', 0);
    expect(fixes).toContainEqual({ original: 'teh', corrected: 'the', start: 7, end: 10 });
    expect(fixes).toContainEqual({ original: 'the ', corrected: '', start: 23, end: 27 });
  });

  it('treats a contraction as a single word fix', () => {
    expect(extractWordFixes("She don't like apples.", "She doesn't like apples.", 0)).toEqual([
      { original: "don't", corrected: "doesn't", start: 4, end: 9 },
    ]);
  });
});

describe('extractWordFixes combined word+punct changes', () => {
  it('splits a word change plus a punctuation insertion into two fixes', () => {
    // Model: 'slow' -> 'slowly,' (word + comma). The comma must become its
    // own insertion marker so the editor matches the suggestions panel.
    const fixes = extractWordFixes(
      'The main issue is that it runs very slow and the interface is not very adaptable for the user.',
      'The main issue is that it runs very slowly, and the interface is not very adaptable for the user.',
      0,
    );
    expect(fixes).toContainEqual({ original: 'slow', corrected: 'slowly', start: 36, end: 40 });
    expect(fixes).toContainEqual({ original: '', corrected: ',', start: 40, end: 40 });
  });

  it('keeps a whole-run joined fix when the leftovers contain real words', () => {
    // 'is are' -> 'was': prefix pairing would leave ' are' behind, so the
    // change must stay a single unit.
    expect(extractWordFixes('he is are', 'he was', 0)).toEqual([
      { original: 'is are', corrected: 'was', start: 3, end: 9 },
    ]);
  });

  it('keeps a joined contraction fix when a leftover is a word fragment', () => {
    // "dont" -> "doesn't": leftover "'" + "t" contains a word char.
    expect(extractWordFixes('She dont like', "She doesn't like", 0)).toEqual([
      { original: 'dont', corrected: "doesn't", start: 4, end: 8 },
    ]);
  });
});

describe('extractWordFixes with non-zero base offset', () => {
  it('places a pure insertion at the absolute offset', () => {
    // Sentence starts at 155 in a multi-sentence text: the comma after
    // 'very slow' (rel 40) must be at 195, not 40.
    const fixes = extractWordFixes(
      'The main issue is that it runs very slow and the interface is not very adaptable for the user.',
      'The main issue is that it runs very slow, and the interface is not very adaptable for the user.',
      155,
    );
    expect(fixes).toEqual([{ original: '', corrected: ',', start: 195, end: 195 }]);
  });

  it('places the split word+punct insertion at the absolute offset', () => {
    const fixes = extractWordFixes(
      'The main issue is that it runs very slow and the interface is not very adaptable for the user.',
      'The main issue is that it runs very slowly, and the interface is not very adaptable for the user.',
      155,
    );
    expect(fixes).toContainEqual({ original: 'slow', corrected: 'slowly', start: 191, end: 195 });
    expect(fixes).toContainEqual({ original: '', corrected: ',', start: 195, end: 195 });
  });
});

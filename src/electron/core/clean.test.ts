import { describe, it, expect } from 'vitest';
import { cleanCorrectedText, isOnlyQuoteChange } from './clean';

describe('cleanCorrectedText', () => {
  it('removes template tags', () => {
    expect(cleanCorrectedText('<|im_start|>Hello there', 'Hello there')).toBe('Hello there');
  });

  it('strips instruction prefixes', () => {
    expect(cleanCorrectedText('Corrected: go now', 'go now')).toBe('go now');
    expect(cleanCorrectedText('Here is the corrected sentence: Go now.', 'go now.')).toBe('Go now.');
  });

  it('removes repeated 5-word segments (keeps first 5 words, matching Python)', () => {
    const input = 'the cat sat on the mat the cat sat on the mat and then left';
    expect(cleanCorrectedText(input, '')).toBe('the cat sat on the');
  });

  it('restores capitalization to match original', () => {
    expect(cleanCorrectedText('hello world', 'Hello world')).toBe('Hello world');
  });

  it('replicates Python quirk: quote cleanup then re-appends ending period', () => {
    // Bug-for-bug: after `".` -> `."` cleanup, the string ends with a quote,
    // so the ending-punctuation restore re-appends the period. Python does the same.
    expect(cleanCorrectedText('He said "stop.".', 'He said "stop".')).toBe('He said "stop.".');
  });

  it('restores ending punctuation from original', () => {
    expect(cleanCorrectedText('Go now', 'Go now.')).toBe('Go now.');
  });

  it('returns original when corrected is empty', () => {
    expect(cleanCorrectedText('', 'keep me')).toBe('keep me');
  });
});

describe('isOnlyQuoteChange', () => {
  it('detects curly-to-straight quote-only changes', () => {
    expect(isOnlyQuoteChange("He said 'hi'", 'He said \u2018hi\u2019')).toBe(true);
  });

  it('returns false when words actually changed', () => {
    expect(isOnlyQuoteChange('She dont go', "She doesn't go")).toBe(false);
  });

  it('returns false for identical strings', () => {
    expect(isOnlyQuoteChange('same', 'same')).toBe(false);
  });
});

import { describe, it, expect } from 'vitest';
import { cleanCorrectedText, isOnlyQuoteChange, fixMojibakeQuotes } from './clean';

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

describe('fixMojibakeQuotes', () => {
  it('fixes CP1252-style mojibake double quotes', () => {
    expect(fixMojibakeQuotes('by \u00e2\u20ac\u0153empowering\u00e2\u20ac\u009d end-users')).toBe('by \u201Cempowering\u201D end-users');
  });

  it('fixes CP1252-style mojibake single quotes', () => {
    expect(fixMojibakeQuotes('it\u00e2\u20ac\u2122s here')).toBe('it\u2019s here');
  });

  it('fixes Latin-1 style mojibake with raw control bytes', () => {
    expect(fixMojibakeQuotes('\u00e2\u0080\u009cempowering\u00e2\u0080\u009d')).toBe('\u201Cempowering\u201D');
  });

  it('fixes ellipsis mojibake', () => {
    expect(fixMojibakeQuotes('and so on\u00e2\u20ac\u00a6')).toBe('and so on\u2026');
  });

  it('leaves normal text untouched', () => {
    expect(fixMojibakeQuotes('The team were excited.')).toBe('The team were excited.');
  });
});

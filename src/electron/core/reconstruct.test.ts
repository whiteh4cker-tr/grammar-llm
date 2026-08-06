import { describe, it, expect } from 'vitest';
import { splitIntoSentences } from './sentences';
import { reconstructTextFromSentences } from './reconstruct';

describe('reconstructTextFromSentences', () => {
  it('preserves single-space gaps', () => {
    const original = 'One. Two.';
    const data = splitIntoSentences(original);
    const result = reconstructTextFromSentences(original, data, ['One!', 'Two?']);
    expect(result).toBe('One! Two?');
  });

  it('preserves newline gaps', () => {
    const original = 'First sentence.\n\nSecond sentence.';
    const data = splitIntoSentences(original);
    const result = reconstructTextFromSentences(original, data, ['First fixed.', 'Second fixed.']);
    expect(result).toBe('First fixed.\n\nSecond fixed.');
  });

  it('preserves leading whitespace before sentences', () => {
    const original = '  Indented start. Next.';
    const data = splitIntoSentences(original);
    const result = reconstructTextFromSentences(original, data, ['Indented fixed.', 'Next fixed.']);
    expect(result).toBe('  Indented fixed. Next fixed.');
  });

  it('returns original text when lengths mismatch', () => {
    const original = 'One. Two.';
    const data = splitIntoSentences(original);
    expect(reconstructTextFromSentences(original, data, ['One!'])).toBe(original);
  });

  it('drops trailing whitespace after the last sentence (Python quirk)', () => {
    const original = 'One. Two.\n\n\n';
    const data = splitIntoSentences(original);
    const result = reconstructTextFromSentences(original, data, ['One!', 'Two?']);
    expect(result).toBe('One! Two?');
  });
});

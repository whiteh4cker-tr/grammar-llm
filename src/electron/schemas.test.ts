import { describe, it, expect } from 'vitest';
import { correctRequestSchema, applyRequestSchema, downloadRequestSchema } from './schemas';

describe('IPC schemas', () => {
  it('accepts a valid correct request', () => {
    expect(correctRequestSchema.parse({ text: 'hello' })).toEqual({ text: 'hello' });
  });

  it('rejects a missing text field', () => {
    expect(() => correctRequestSchema.parse({})).toThrow();
  });

  it('rejects a non-url download request', () => {
    expect(() => downloadRequestSchema.parse({ url: 'not a url', fileName: 'x.gguf' })).toThrow();
  });

  it('accepts a valid apply request with suggestions', () => {
    const suggestion = {
      original: 'a', corrected: 'b', sentence: 'Sentence 1',
      start_index: 0, end_index: 1, original_highlighted: '', corrected_highlighted: '',
    };
    const parsed = applyRequestSchema.parse({
      originalText: 'a', suggestionIndex: 0, suggestions: [suggestion],
    });
    expect(parsed.suggestionIndex).toBe(0);
    expect(parsed.suggestions[0].start_index).toBe(0);
  });
});

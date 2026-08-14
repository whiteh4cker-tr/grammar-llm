import { describe, it, expect } from 'vitest';
import { correctRequestSchema, applyRequestSchema, downloadRequestSchema, contextSizeSchema, wordFixSchema } from './schemas';

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
      wordFixes: [],
    };
    const parsed = applyRequestSchema.parse({
      originalText: 'a', suggestionIndex: 0, suggestions: [suggestion],
    });
    expect(parsed.suggestionIndex).toBe(0);
    expect(parsed.suggestions[0].start_index).toBe(0);
  });
});

describe('contextSizeSchema', () => {
  it('accepts the default 8192', () => {
    expect(contextSizeSchema.parse(8192)).toBe(8192);
  });

  it('rejects values below 256 or above 131072', () => {
    expect(() => contextSizeSchema.parse(255)).toThrow();
    expect(() => contextSizeSchema.parse(131_073)).toThrow();
  });

  it('rejects non-integers', () => {
    expect(() => contextSizeSchema.parse(4096.5)).toThrow();
  });
});

describe('wordFixSchema', () => {
  it('accepts a valid word fix', () => {
    expect(wordFixSchema.parse({ original: 'dont', corrected: "doesn't", start: 4, end: 8 }))
      .toEqual({ original: 'dont', corrected: "doesn't", start: 4, end: 8 });
  });

  it('rejects a missing corrected field', () => {
    expect(() => wordFixSchema.parse({ original: 'dont', start: 4, end: 8 })).toThrow();
  });
});

import { describe, it, expect } from 'vitest';
import { splitIntoSentences } from './sentences';

describe('splitIntoSentences', () => {
  it('splits simple sentences', () => {
    const result = splitIntoSentences('First one. Second one.');
    expect(result.map((s) => s.text)).toEqual(['First one.', 'Second one.']);
  });

  it('does not split after abbreviations (Dr.)', () => {
    const result = splitIntoSentences('Dr. Smith went home. He slept.');
    expect(result.map((s) => s.text)).toEqual(['Dr. Smith went home.', 'He slept.']);
  });

  it('splits when decimal is beyond the last-10-chars window (matches Python)', () => {
    const result = splitIntoSentences('It costs 3.5 dollars. Really.');
    expect(result.map((s) => s.text)).toEqual(['It costs 3.5 dollars.', 'Really.']);
  });

  it('does not split when a decimal is within the last 10 chars (Python quirk: swallows following sentence)', () => {
    const result = splitIntoSentences('Total is 3.5. Next.');
    expect(result.map((s) => s.text)).toEqual(['Total is 3.5. Next.']);
  });

  it('splits after U.S. initials', () => {
    const result = splitIntoSentences('U.S. citizens vote. They do.');
    expect(result.map((s) => s.text)).toEqual(['U.S. citizens vote.', 'They do.']);
  });

  it('treats whole text as one sentence when nothing splits', () => {
    const result = splitIntoSentences('she is a lowercase start');
    expect(result.map((s) => s.text)).toEqual(['she is a lowercase start']);
  });

  it('tracks start/end/spanEnd indices', () => {
    const result = splitIntoSentences('Hi. There.');
    expect(result[0]).toMatchObject({ start: 0, end: 3, spanEnd: 4 });
    expect(result[1]).toMatchObject({ start: 4, end: 10, spanEnd: 11 });
  });

  it('returns empty array for blank text', () => {
    expect(splitIntoSentences('   ')).toEqual([]);
  });
});

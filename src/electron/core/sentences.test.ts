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

  it('splits when no space follows the terminal punctuation', () => {
    const text = 'We are hoping that the company will soon release an update, which will have a good effect on its functionality.The team were very excited to receive the new software, but its performance has been poor.';
    expect(splitIntoSentences(text).map((s) => s.text)).toEqual([
      'We are hoping that the company will soon release an update, which will have a good effect on its functionality.',
      'The team were very excited to receive the new software, but its performance has been poor.',
    ]);
  });

  it('splits after ! and ? with no space', () => {
    expect(splitIntoSentences('Stop!Run away.').map((s) => s.text)).toEqual(['Stop!', 'Run away.']);
    expect(splitIntoSentences('Really?Yes indeed.').map((s) => s.text)).toEqual(['Really?', 'Yes indeed.']);
  });

  it('tracks start/end indices across a space-less boundary', () => {
    const result = splitIntoSentences('Hi.There.');
    expect(result.map((s) => s.text)).toEqual(['Hi.', 'There.']);
    expect(result[0]).toMatchObject({ start: 0, end: 3, spanEnd: 3 });
    expect(result[1]).toMatchObject({ start: 3, end: 9, spanEnd: 10 });
  });

  it('does not split single-letter initials without a space (U.S.Today)', () => {
    expect(splitIntoSentences('He lives in the U.S.Today is Monday.').map((s) => s.text)).toEqual([
      'He lives in the U.S.Today is Monday.',
    ]);
  });

  it('does not split abbreviations or decimals without a space', () => {
    expect(splitIntoSentences('Bring apples, oranges, etc.They are fresh.').map((s) => s.text)).toEqual([
      'Bring apples, oranges, etc.They are fresh.',
    ]);
    expect(splitIntoSentences('It costs 3.5.Next time.').map((s) => s.text)).toEqual([
      'It costs 3.5.Next time.',
    ]);
    expect(splitIntoSentences('Open the file README.TXT now.').map((s) => s.text)).toEqual([
      'Open the file README.TXT now.',
    ]);
  });

  it('does not split a space-less lowercase continuation', () => {
    expect(splitIntoSentences('She is happy.enough for now.').map((s) => s.text)).toEqual([
      'She is happy.enough for now.',
    ]);
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

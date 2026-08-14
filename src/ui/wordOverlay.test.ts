import { describe, it, expect } from 'vitest';
import { buildSegments, normalizeFixes } from './wordOverlay';

describe('buildSegments', () => {
  it('returns a single non-fix segment when there are no fixes', () => {
    expect(buildSegments('hello world', [])).toEqual([
      { text: 'hello world', isFix: false },
    ]);
  });

  it('splits around fixes with absolute offsets', () => {
    const fixes = [{ original: 'dont', corrected: "doesn't", start: 4, end: 8 }];
    const segments = buildSegments('She dont go', fixes);
    expect(segments).toEqual([
      { text: 'She ', isFix: false },
      { text: 'dont', isFix: true, fix: fixes[0] },
      { text: ' go', isFix: false },
    ]);
  });

  it('handles empty text', () => {
    expect(buildSegments('', [])).toEqual([]);
  });

  it('skips fixes out of range', () => {
    const fixes = [{ original: 'x', corrected: 'y', start: 100, end: 101 }];
    expect(buildSegments('abc', fixes)).toEqual([{ text: 'abc', isFix: false }]);
  });

  it('sorts unsorted fixes and clips overlapping ones', () => {
    const a = { original: 'aa', corrected: 'bb', start: 2, end: 4 };
    const b = { original: 'cc', corrected: 'dd', start: 5, end: 7 };
    const segments = buildSegments('0123456789', [b, a]);
    expect(segments.map((s) => s.text)).toEqual(['01', '23', '4', '56', '789']);
    expect(segments.filter((s) => s.isFix).map((s) => s.fix?.original)).toEqual(['aa', 'cc']);
  });
});

describe('buildSegments with insertions', () => {
  it('emits a zero-width segment at the insertion point', () => {
    const fix = { original: '', corrected: ',', start: 5, end: 5 };
    const segments = buildSegments('Hello John', [fix]);
    expect(segments).toEqual([
      { text: 'Hello', isFix: false },
      { text: '', isFix: true, fix },
      { text: ' John', isFix: false },
    ]);
  });

  it('handles an insertion at the start of the text', () => {
    const fix = { original: '', corrected: 'Hi ', start: 0, end: 0 };
    const segments = buildSegments('John', [fix]);
    expect(segments.map((s) => s.text)).toEqual(['', 'John']);
    expect(segments[0].isFix).toBe(true);
  });

  it('sorts insertions and replacements together by position', () => {
    const ins = { original: '', corrected: ',', start: 5, end: 5 };
    const rep = { original: 'teh', corrected: 'the', start: 6, end: 9 };
    const segments = buildSegments('Hello teh', [rep, ins]);
    expect(segments.map((s) => s.text)).toEqual(['Hello', '', ' ', 'teh']);
    expect(segments.map((s) => s.isFix)).toEqual([false, true, false, true]);
  });
});

describe('normalizeFixes', () => {
  const text = 'The team were very excited to receive the new software, but it\'s performance have been poorly.';

  it('snaps an in-word insertion point to the end of the word', () => {
    const bad = { original: '', corrected: ',', start: 57, end: 57 }; // inside 'but' (56..59)
    const fixed = normalizeFixes(text, [bad]);
    expect(fixed[0].start).toBe(59);
    expect(fixed[0].end).toBe(59);
  });

  it('keeps a valid insertion point unchanged', () => {
    const good = { original: '', corrected: ',', start: 59, end: 59 };
    expect(normalizeFixes(text, [good])).toEqual([good]);
  });

  it('keeps a span that matches the text', () => {
    const fix = { original: 'were', corrected: 'was', start: 9, end: 13 };
    expect(normalizeFixes(text, [fix])).toEqual([fix]);
  });

  it('re-locates a stale span to the nearest occurrence', () => {
    const fix = { original: 'were', corrected: 'was', start: 100, end: 104 }; // stale
    const fixed = normalizeFixes(text, [fix]);
    expect(fixed[0].start).toBe(9);
    expect(fixed[0].end).toBe(13);
  });

  it('drops a span whose original no longer exists', () => {
    const fix = { original: 'zzzz', corrected: 'aaaa', start: 10, end: 14 };
    expect(normalizeFixes(text, [fix])).toEqual([]);
  });

  it('keeps insertions at the very start of the text', () => {
    const fix = { original: '', corrected: 'Hi ', start: 0, end: 0 };
    expect(normalizeFixes(text, [fix])).toEqual([fix]);
  });
});

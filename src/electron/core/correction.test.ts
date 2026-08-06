import { describe, it, expect } from 'vitest';
import { correctText } from './correction';
import type { SentenceCorrector } from './types';

function fakeCorrector(map: Record<string, string>): SentenceCorrector {
  return {
    async correct(sentence: string): Promise<string> {
      return map[sentence] ?? sentence;
    },
  };
}

describe('correctText', () => {
  it('reproduces the README example output', async () => {
    const text = 'She dont like the apples. this is a bad sentence';
    const corrector = fakeCorrector({
      [text]: "She doesn't like the apples. This is a bad sentence",
    });
    const result = await correctText(text, corrector);

    expect(result.correctedText).toBe("She doesn't like the apples. This is a bad sentence");
    expect(result.suggestions).toHaveLength(1);
    expect(result.suggestions[0]).toMatchObject({
      original: text,
      corrected: "She doesn't like the apples. This is a bad sentence",
      sentence: 'Sentence 1',
      start_index: 0,
      end_index: 48,
    });
    expect(result.suggestions[0].original_highlighted).toContain('<span class="error-word">dont</span>');
    expect(result.suggestions[0].original_highlighted).toContain('<span class="error-word">this</span>');
  });

  it('returns empty response for blank text', async () => {
    const result = await correctText('   ', fakeCorrector({}));
    expect(result).toEqual({ suggestions: [], correctedText: '' });
  });

  it('skips sentences shorter than 2 chars', async () => {
    const corrector = fakeCorrector({});
    const result = await correctText('x', corrector);
    expect(result.suggestions).toHaveLength(0);
    expect(result.correctedText).toBe('x');
  });

  it('rejects corrections longer than 2x the original', async () => {
    const text = 'Go now.';
    const corrector = fakeCorrector({ [text]: 'Go now immediately because it is very important to leave.' });
    const result = await correctText(text, corrector);
    expect(result.correctedText).toBe('Go now.');
    expect(result.suggestions).toHaveLength(0);
  });

  it('omits quote-only changes from suggestions and keeps original text', async () => {
    const text = "He said 'hi'.";
    const corrector = fakeCorrector({ [text]: 'He said \u2018hi\u2019.' });
    const result = await correctText(text, corrector);
    expect(result.suggestions).toHaveLength(0);
    // Python: correct_sentence returns the ORIGINAL sentence for quote-only changes
    expect(result.correctedText).toBe(text);
  });

  it('omits suggestions when correction is >1.5x original length', async () => {
    const text = 'Fix this.';
    const corrector = fakeCorrector({ [text]: 'Fix this nicely.' }); // 16 chars: >1.5x, <2x
    const result = await correctText(text, corrector);
    expect(result.correctedText).toBe('Fix this nicely.');
    expect(result.suggestions).toHaveLength(0);
  });

  it('corrects each sentence with preserved indices', async () => {
    const text = 'Bad one. Worse one.';
    const corrector = fakeCorrector({
      'Bad one.': 'Good one.',
      'Worse one.': 'Better one.',
    });
    const result = await correctText(text, corrector);
    expect(result.correctedText).toBe('Good one. Better one.');
    expect(result.suggestions.map((s) => s.sentence)).toEqual(['Sentence 1', 'Sentence 2']);
    expect(result.suggestions[1].start_index).toBe(9);
  });
});

  it('preserves original quote style and highlights only real changes', async () => {
    const text = 'He said \u201Cempovering\u201D is wrong.';
    const corrector = fakeCorrector({ [text]: 'He said "empowering" is wrong.' });
    const result = await correctText(text, corrector);

    expect(result.suggestions).toHaveLength(1);
    expect(result.suggestions[0].corrected).toBe('He said \u201Cempowering\u201D is wrong.');
    expect(result.correctedText).toBe('He said \u201Cempowering\u201D is wrong.');

    const originalHighlighted = result.suggestions[0].original_highlighted;
    expect(originalHighlighted).toContain('<span class="error-word">empovering</span>');
    expect(originalHighlighted).not.toContain('error-word">He');
    expect(originalHighlighted).not.toContain('error-word">said');
    expect(result.suggestions[0].corrected_highlighted).toContain('<span class="corrected-word">empowering</span>');
  });

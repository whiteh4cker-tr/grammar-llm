import type { SentenceData } from './types.js';

const ABBREVIATIONS = new Set([
  'etc', 'eg', 'e.g', 'ie', 'i.e', 'vs', 'viz', 'cf', 'ca', 'approx',
  'no', 'vol', 'fig', 'p', 'pp', 'ch', 'sec', 'ex', 'al', 'et', 'seq',
  'etc.', 'e.g.', 'i.e.', 'vs.', 'viz.', 'cf.', 'ca.', 'approx.',
  'no.', 'vol.', 'fig.', 'p.', 'pp.', 'ch.', 'sec.', 'ex.', 'et al.', 'seq.',
  'mr', 'mrs', 'ms', 'dr', 'prof', 'rev', 'sr', 'jr', 'st',
]);

// Python: (?<=[.!?])(?!\w)(?<!\d\.\d)(?<!\s[A-Za-z]\.)\s+(?=[A-Z"'])|(?<=[.!?])\s*$
// (re.VERBOSE + re.IGNORECASE; \w is ASCII in JS — acceptable for English text)
const SENTENCE_BOUNDARY_RE = /(?<=[.!?])(?!\w)(?<!\d\.\d)(?<!\s[A-Za-z]\.)\s+(?=[A-Z"'])|(?<=[.!?])\s*$/gi;

export function splitIntoSentences(text: string): SentenceData[] {
  const sentences: SentenceData[] = [];
  if (!text.trim()) return sentences;

  let lastEnd = 0;
  const potentialSplits: RegExpExecArray[] = [];
  const re = new RegExp(SENTENCE_BOUNDARY_RE.source, 'gi');
  let match: RegExpExecArray | null;
  while ((match = re.exec(text)) !== null) {
    potentialSplits.push(match);
    // JS exec() loops forever on zero-width matches; Python's finditer()
    // auto-advances. Guard like findAllOccurrences does.
    if (match.index === re.lastIndex) re.lastIndex++;
  }

  for (const match of potentialSplits) {
    const splitPos = match.index; // start of whitespace right after punctuation
    const sentenceText = text.slice(lastEnd, splitPos + 1).trim();
    if (!sentenceText) {
      lastEnd = splitPos + 1;
      continue;
    }

    let isTrueBoundary = true;

    const prevWords = sentenceText.toLowerCase().split(/\s+/);
    if (prevWords.length > 0) {
      const lastWord = prevWords[prevWords.length - 1].replace(/^[.,!?;:"']+|[.,!?;:"']+$/g, '');
      if (ABBREVIATIONS.has(lastWord)) {
        isTrueBoundary = false;
      } else if (/\d\.\d/.test(sentenceText.slice(-10))) {
        // Python quirk replicated: the single-letter-initial check was dead
        // code in main.py (misspelled variable), so initials ARE split here.
        isTrueBoundary = false;
      }
    }

    if (splitPos + 2 < text.length) {
      const nextChars = text.slice(splitPos + 1, splitPos + 3);
      // Python: (next_chars and next_chars[0].islower()) or next_chars[0].isdigit()
      if (/[a-z\d]/.test(nextChars[0])) {
        isTrueBoundary = false;
      }
    }

    if (!isTrueBoundary) continue;

    let startNoWs = lastEnd;
    while (startNoWs < splitPos + 1 && /\s/.test(text[startNoWs])) startNoWs++;

    let spanEnd = splitPos;
    while (spanEnd < text.length && /\s/.test(text[spanEnd])) spanEnd++;
    if (spanEnd === splitPos) spanEnd = splitPos + 1;

    sentences.push({ text: sentenceText, start: startNoWs, end: splitPos, spanEnd, gapBeforeStart: lastEnd });
    lastEnd = spanEnd;
  }

  if (lastEnd < text.length) {
    const remaining = text.slice(lastEnd).trim();
    if (remaining) {
      let startNoWs = lastEnd;
      while (startNoWs < text.length && /\s/.test(text[startNoWs])) startNoWs++;
      sentences.push({ text: remaining, start: startNoWs, end: text.length, spanEnd: text.length, gapBeforeStart: lastEnd });
    }
  }

  if (sentences.length === 0) {
    const content = text.trim();
    if (content) {
      let startNoWs = 0;
      while (startNoWs < text.length && /\s/.test(text[startNoWs])) startNoWs++;
      sentences.push({ text: content, start: startNoWs, end: text.length, spanEnd: text.length, gapBeforeStart: 0 });
    }
  }

  return sentences;
}

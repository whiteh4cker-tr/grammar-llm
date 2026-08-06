import { diffArrays } from 'diff';

export function tokenize(text: string): string[] {
  const tokens: string[] = [];
  let current = '';
  for (const char of text) {
    if (/\s/.test(char)) {
      if (current) {
        tokens.push(current);
        current = '';
      }
      tokens.push(char);
    } else if ('.,!?;:\'"()[]{}\u201C\u201D\u2018\u2019'.includes(char)) {
      if (current) {
        tokens.push(current);
        current = '';
      }
      tokens.push(char);
    } else {
      current += char;
    }
  }
  if (current) tokens.push(current);
  return tokens;
}

// Curly quotes compare equal to their straight counterparts so that
// quote-style changes alone never trigger word highlighting.
function normalizeToken(token: string): string {
  return token
    .replace(/[\u201C\u201D]/g, '"')
    .replace(/[\u2018\u2019]/g, "'");
}

export function highlightWordDifferences(
  original: string,
  corrected: string,
): {
  originalHighlighted: string;
  correctedHighlighted: string;
  /** Corrected text with the original's quote characters restored. */
  preservedCorrected: string;
} {
  const originalTokens = tokenize(original);
  const correctedTokens = tokenize(corrected);
  const parts = diffArrays(originalTokens.map(normalizeToken), correctedTokens.map(normalizeToken));

  let originalHighlighted = '';
  let correctedHighlighted = '';
  let preservedCorrected = '';
  let originalIndex = 0;
  let correctedIndex = 0;

  for (const part of parts) {
    const value = Array.isArray(part.value) ? part.value : [part.value];
    const length = value.length;

    if (part.added) {
      for (let k = 0; k < length; k++) {
        const token = correctedTokens[correctedIndex + k];
        correctedHighlighted += token.trim()
          ? `<span class="corrected-word">${token}</span>`
          : token;
        preservedCorrected += token;
      }
      correctedIndex += length;
    } else if (part.removed) {
      for (let k = 0; k < length; k++) {
        const token = originalTokens[originalIndex + k];
        originalHighlighted += token.trim()
          ? `<span class="error-word">${token}</span>`
          : token;
      }
      originalIndex += length;
    } else {
      // Equal after quote normalization: emit the ORIGINAL characters on both
      // sides so quote style stays untouched and words aren't highlighted.
      const originalSlice = originalTokens.slice(originalIndex, originalIndex + length).join('');
      originalHighlighted += originalSlice;
      correctedHighlighted += originalSlice;
      preservedCorrected += originalSlice;
      originalIndex += length;
      correctedIndex += length;
    }
  }

  return { originalHighlighted, correctedHighlighted, preservedCorrected };
}

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
    } else if ('.,!?;:\'"()[]{}'.includes(char)) {
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

export function highlightWordDifferences(
  original: string,
  corrected: string,
): { originalHighlighted: string; correctedHighlighted: string } {
  const originalTokens = tokenize(original);
  const correctedTokens = tokenize(corrected);
  const parts = diffArrays(originalTokens, correctedTokens);

  let originalHighlighted = '';
  let correctedHighlighted = '';

  for (const part of parts) {
    const value = Array.isArray(part.value) ? part.value : [part.value];
    if (part.added) {
      for (const token of value) {
        correctedHighlighted += token.trim()
          ? `<span class="corrected-word">${token}</span>`
          : token;
      }
    } else if (part.removed) {
      for (const token of value) {
        originalHighlighted += token.trim()
          ? `<span class="error-word">${token}</span>`
          : token;
      }
    } else {
      originalHighlighted += value.join('');
      correctedHighlighted += value.join('');
    }
  }

  return { originalHighlighted, correctedHighlighted };
}

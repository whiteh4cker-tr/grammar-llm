const TEMPLATE_TAG_RE = /<\|.*?\|>/g;
const QUOTE_PERIOD_RE = /([.!?])(["'])\s*\./g;
const QUOTE_DUP_RE = /([.!?])(["'])\s*\1/g;

const INSTRUCTION_PREFIXES = [
  'correct the grammar and spelling of this sentence:',
  'here is the corrected sentence:',
  'corrected sentence:',
  'the corrected version is:',
  'grammar correction:',
  'corrected:',
];

export function cleanCorrectedText(corrected: string, original: string): string {
  if (!corrected) return original;

  let result = corrected.replace(TEMPLATE_TAG_RE, '').trim();

  for (const prefix of INSTRUCTION_PREFIXES) {
    if (result.toLowerCase().startsWith(prefix)) {
      result = result.slice(prefix.length).trim();
      result = result.replace(/^[:]\s*/, '');
    }
  }

  const words = result.split(/\s+/);
  if (words.length > 10) {
    for (let i = 0; i < words.length - 5; i++) {
      const segment = words.slice(i, i + 5).join(' ');
      if (words.slice(i + 5).join(' ').includes(segment)) {
        result = words.slice(0, i + 5).join(' ');
        break;
      }
    }
  }

  if (original && /^[A-Z]/.test(original) && result && /^[a-z]/.test(result)) {
    result = result[0].toUpperCase() + result.slice(1);
  }

  for (let i = 0; i < 3; i++) {
    const before = result;
    result = result.replace(QUOTE_PERIOD_RE, '$1$2');
    result = result.replace(QUOTE_DUP_RE, '$1$2');
    if (result === before) break;
  }

  if (original && /[.!?]$/.test(original) && result && !/[.!?]$/.test(result)) {
    result += original[original.length - 1];
  }

  return result.trim();
}

export function isOnlyQuoteChange(original: string, corrected: string): boolean {
  if (original === corrected) return false;
  const normalize = (s: string): string => s.replace(/[\u2018\u2019]/g, "'").replace(/[\u201C\u201D]/g, '"');
  if (normalize(original) === normalize(corrected)) return true;
  if (normalize(original.trim()) === normalize(corrected.trim())) return true;
  return false;
}

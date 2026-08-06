# GrammarLLM Electron Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate GrammarLLM (Python FastAPI + vanilla JS web app) into the existing Electron + React + TypeScript project with full feature parity, using node-llama-cpp for local inference with auto-detected GPU backends.

**Architecture:** Three layers — pure TS core engine (`src/electron/core/`, fully unit-tested, no Electron imports), main process (IPC handlers, model manager, node-llama-cpp service, secure preload), React renderer (port of the vanilla UI). No REST API; renderer talks to main only via typed `window.api` exposed through `contextBridge` in a CommonJS preload.

**Tech Stack:** Electron 43, React 19, TypeScript 6, Vite 8, node-llama-cpp, zod, jsdiff, jspdf, vitest, electron-builder.

## Global Constraints

- Reference implementation: `migration/grammar-llm/` — `main.py` (engine), `static/script.js` (UI logic), `static/index.html` + `static/style.css` (UI). Port behavior faithfully, including the Python quirks noted inline.
- Model download URLs (verbatim, repo `icecubetr`): `https://huggingface.co/icecubetr/GRMR-V3-G4B-GGUF/resolve/main/GRMR-V3-G4B-Q4_K_M.gguf` and `https://huggingface.co/icecubetr/GRMR-V3-G4B-GGUF/resolve/main/GRMR-V3-G4B-Q8_0.gguf`. Q4_K_M labeled "Recommended — faster, smaller download" (default-selected); Q8_0 labeled "Highest quality — slower".
- **NEVER download the model file.** No test or verification step may fetch a GGUF. Model download is tested by the user personally after completion.
- Do not run `npm run build` or electron-builder packaging without explicit user approval. Verification uses `npx vitest run` and `npx tsc -b` (type-check + test only). `npm run dev:electron` is run by the user.
- Suggestion data shape keeps Python field names: `original, corrected, sentence, start_index, end_index, original_highlighted, corrected_highlighted`.
- `node-llama-cpp` goes in `dependencies` (electron-builder bundles prod deps only). Native binaries must be unpacked from asar.
- Renderer stays secure: `contextIsolation: true, sandbox: true, nodeIntegration: false`, preload is CommonJS (`preload.cjs`).
- Every task ends with a git commit (task 1 runs `git init`).

## File Structure

**Create:**
- `src/electron/core/types.ts` — `SentenceData`, `Suggestion`, `CorrectionResponse`, `SentenceCorrector`
- `src/electron/core/sentences.ts` — `splitIntoSentences`
- `src/electron/core/clean.ts` — `cleanCorrectedText`, `isOnlyQuoteChange`
- `src/electron/core/diff.ts` — `tokenize`, `highlightWordDifferences`
- `src/electron/core/reconstruct.ts` — `reconstructTextFromSentences`
- `src/electron/core/apply.ts` — `applySuggestion`, `applySuggestionsBulk`, `findAllOccurrences`
- `src/electron/core/correction.ts` — `correctText`, `cleanQuotePunctuation`
- `src/electron/core/*.test.ts` — vitest suites (one per module)
- `src/electron/ipc-types.ts` — `IpcApi`, `ModelStatus`, `DownloadProgress`, request types
- `src/electron/ipc.ts` — zod schemas + `registerIpcHandlers`
- `src/electron/modelManager.ts` — `ModelManager` + `ModelDownloader` interface + `createNodeLlamaDownloader` adapter
- `src/electron/llamaService.ts` — `LlamaCorrectionService`
- `src/electron/preload.cjs` — contextBridge bridge
- `src/ui/electron-api.d.ts` — `window.api` global declaration
- `src/ui/api.ts` — renderer-side API wrapper
- `src/ui/components/ModelGate.tsx`, `GrammarApp.tsx`, `SuggestionsList.tsx`, `ScoreBadge.tsx`, `ReportButton.tsx`, `ThemeToggle.tsx`, `pdf.ts`
- `src/ui/App.css` — ported styles
- `vitest.config.ts` — vitest config (node environment)
- `models/.gitkeep` — dev models dir placeholder

**Modify:**
- `package.json` — deps, `test` scripts
- `src/electron/main.ts` — window + preload + service wiring
- `src/electron/tsconfig.json` — exclude `**/*.test.ts`
- `src/ui/App.tsx`, `src/ui/main.tsx` — new app tree
- `electron-builder.json` — `asarUnpack`
- `.gitignore` — add `/models`
- `index.html` — title

---

### Task 1: Project Setup — deps, vitest, git

**Files:**
- Modify: `package.json`, `vite.config.ts`, `src/electron/tsconfig.json`, `.gitignore`
- Create: `vitest.config.ts`, `models/.gitkeep`

**Interfaces:** none (foundation)

- [ ] **Step 1: git init + baseline commit**

```bash
git init && git add -A && git commit -m "chore: baseline before grammarllm migration"
```

- [ ] **Step 2: Install dependencies**

```bash
npm i node-llama-cpp zod diff jspdf
npm i -D vitest
```

- [ ] **Step 3: Add test script + vitest config**

`package.json` scripts — add:
```json
"test": "vitest run",
"test:watch": "vitest"
```

Create `vitest.config.ts`:
```ts
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'node',
    include: ['src/electron/**/*.test.ts'],
  },
});
```

- [ ] **Step 4: Exclude tests from electron compile + ignore models**

`src/electron/tsconfig.json` — add `"exclude": ["**/*.test.ts"]`.
`.gitignore` — add `/models` line after `dist-electron`.

- [ ] **Step 5: Verify setup**

Run: `npx vitest run` — expected: "No test files found" (or 0 tests), exit 0.
Run: `npx tsc -b` — expected: PASS.
Run: `node -e "console.log(Object.keys(require('./package.json').dependencies))"` — verify `node-llama-cpp`, `zod`, `jsdiff`, `jspdf` in dependencies; `vitest` in devDependencies.

- [ ] **Step 6: Record node-llama-cpp API surface**

Inspect installed types:
```bash
grep -rn "createModelDownloader\|stopGenerationTriggers\|repeatPenalty\|contextSize\|resolveModelFile" node_modules/node-llama-cpp/dist/index.d.ts | head -30
```
Note any signature differences from this plan's usage (Task 10 adapts if needed). Record findings in a comment at the top of `src/electron/llamaService.ts` when created.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "chore: setup vitest, deps for grammarllm migration"
```
### Task 2: Core types + sentence splitting

**Files:**
- Create: `src/electron/core/types.ts`, `src/electron/core/sentences.ts`, `src/electron/core/sentences.test.ts`

**Interfaces:**
- Produces: `SentenceData { text, start, end, spanEnd, gapBeforeStart }`, `Suggestion`, `CorrectionResponse`, `SentenceCorrector` (all in types.ts), `splitIntoSentences(text: string): SentenceData[]`

- [ ] **Step 1: Write types.ts**

```ts
export interface SentenceData {
  text: string;
  start: number;
  end: number;
  spanEnd: number;
  gapBeforeStart: number;
}

export interface Suggestion {
  original: string;
  corrected: string;
  sentence: string;
  start_index: number;
  end_index: number;
  original_highlighted: string;
  corrected_highlighted: string;
}

export interface CorrectionResponse {
  suggestions: Suggestion[];
  correctedText: string;
}

export interface SentenceCorrector {
  correct(sentence: string): Promise<string>;
}
```

- [ ] **Step 2: Write the failing tests** (`sentences.test.ts`)

```ts
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `npx vitest run src/electron/core/sentences.test.ts`
Expected: FAIL — "Cannot find module './sentences'"

- [ ] **Step 4: Implement sentences.ts** (port of `split_into_sentences`; Python quirk replicated)

```ts
import type { SentenceData } from './types';

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
    // auto-advances. Guard required because the second alternative
    // `(?<=[.!?])\s*$` can match zero-width.
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `npx vitest run src/electron/core/sentences.test.ts`
Expected: PASS (all 7)

- [ ] **Step 6: Commit**

```bash
git add src/electron/core && git commit -m "feat: port sentence splitting to TS"
```
### Task 3: Correction cleaning

**Files:**
- Create: `src/electron/core/clean.ts`, `src/electron/core/clean.test.ts`

**Interfaces:**
- Produces: `cleanCorrectedText(corrected: string, original: string): string`, `isOnlyQuoteChange(original: string, corrected: string): boolean`

- [ ] **Step 1: Write the failing tests** (`clean.test.ts`)

```ts
import { describe, it, expect } from 'vitest';
import { cleanCorrectedText, isOnlyQuoteChange } from './clean';

describe('cleanCorrectedText', () => {
  it('removes template tags', () => {
    expect(cleanCorrectedText('<|im_start|>Hello there', 'Hello there')).toBe('Hello there');
  });

  it('strips instruction prefixes', () => {
    expect(cleanCorrectedText('Corrected: go now', 'go now')).toBe('go now');
    expect(cleanCorrectedText('Here is the corrected sentence: Go now.', 'go now.')).toBe('Go now.');
  });

  it('removes repeated 5-word segments (keeps first 5 words, matching Python)', () => {
    const input = 'the cat sat on the mat the cat sat on the mat and then left';
    expect(cleanCorrectedText(input, '')).toBe('the cat sat on the');
  });

  it('restores capitalization to match original', () => {
    expect(cleanCorrectedText('hello world', 'Hello world')).toBe('Hello world');
  });

  it('replicates Python quirk: quote cleanup then re-appends ending period', () => {
    // Bug-for-bug: after `".` -> `."` cleanup, the string ends with a quote,
    // so the ending-punctuation restore re-appends the period. Python does the same.
    expect(cleanCorrectedText('He said "stop.".', 'He said "stop".')).toBe('He said "stop.".');
  });

  it('restores ending punctuation from original', () => {
    expect(cleanCorrectedText('Go now', 'Go now.')).toBe('Go now.');
  });

  it('returns original when corrected is empty', () => {
    expect(cleanCorrectedText('', 'keep me')).toBe('keep me');
  });
});

describe('isOnlyQuoteChange', () => {
  it('detects curly-to-straight quote-only changes', () => {
    expect(isOnlyQuoteChange("He said 'hi'", 'He said \u2018hi\u2019')).toBe(true);
  });

  it('returns false when words actually changed', () => {
    expect(isOnlyQuoteChange('She dont go', "She doesn't go")).toBe(false);
  });

  it('returns false for identical strings', () => {
    expect(isOnlyQuoteChange('same', 'same')).toBe(false);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/electron/core/clean.test.ts`
Expected: FAIL — "Cannot find module './clean'"

- [ ] **Step 3: Implement clean.ts**

```ts
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx vitest run src/electron/core/clean.test.ts`
Expected: PASS (all 11)

- [ ] **Step 5: Commit**

```bash
git add src/electron/core && git commit -m "feat: port correction cleaning to TS"
```
### Task 4: Word-diff highlighting

**Files:**
- Create: `src/electron/core/diff.ts`, `src/electron/core/diff.test.ts`

**Note:** the jsdiff library is published on npm as `diff` (the `jsdiff` package is an abandoned v1.1.1). Import: `import { diffArrays } from 'diff';`

**Interfaces:**
- Produces: `tokenize(text: string): string[]`, `highlightWordDifferences(original: string, corrected: string): { originalHighlighted: string; correctedHighlighted: string }`

- [ ] **Step 1: Write the failing tests** (`diff.test.ts`)

```ts
import { describe, it, expect } from 'vitest';
import { tokenize, highlightWordDifferences } from './diff';

describe('tokenize', () => {
  it('keeps punctuation and whitespace as separate tokens', () => {
    expect(tokenize("dont go.")).toEqual(['dont', ' ', 'go', '.']);
  });

  it('splits contractions on apostrophes', () => {
    expect(tokenize("doesn't")).toEqual(['doesn', "'", 't']);
  });
});

describe('highlightWordDifferences', () => {
  it('highlights replaced words in both directions', () => {
    const { originalHighlighted, correctedHighlighted } = highlightWordDifferences(
      'She dont like the apples. this is a bad sentence',
      "She doesn't like the apples. This is a bad sentence",
    );
    expect(originalHighlighted).toContain('<span class="error-word">dont</span>');
    expect(originalHighlighted).toContain('<span class="error-word">this</span>');
    expect(correctedHighlighted).toContain('<span class="corrected-word">doesn</span>');
    expect(correctedHighlighted).toContain('<span class="corrected-word">This</span>');
    expect(correctedHighlighted).toContain('like the apples');
  });

  it('highlights deleted words only in original', () => {
    const { originalHighlighted, correctedHighlighted } = highlightWordDifferences('a b c', 'a c');
    expect(originalHighlighted).toContain('<span class="error-word">b</span>');
    expect(correctedHighlighted).not.toContain('corrected-word');
  });

  it('preserves whitespace tokens unhighlighted', () => {
    const { originalHighlighted } = highlightWordDifferences('a\nb', 'a\nc');
    expect(originalHighlighted).toContain('\n');
    expect(originalHighlighted).not.toContain('error-word">\n<');
  });

  it('returns empty strings for identical text', () => {
    const { originalHighlighted, correctedHighlighted } = highlightWordDifferences('same', 'same');
    expect(originalHighlighted).toBe('same');
    expect(correctedHighlighted).toBe('same');
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/electron/core/diff.test.ts`
Expected: FAIL — "Cannot find module './diff'"

- [ ] **Step 3: Implement diff.ts**

```ts
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx vitest run src/electron/core/diff.test.ts`
Expected: PASS (all 7)

- [ ] **Step 5: Commit**

```bash
git add src/electron/core && git commit -m "feat: port word-diff highlighting to TS"
```

### Task 5: Text reconstruction

**Files:**
- Create: `src/electron/core/reconstruct.ts`, `src/electron/core/reconstruct.test.ts`

**Interfaces:**
- Consumes: `SentenceData` (types.ts), `splitIntoSentences` in tests
- Produces: `reconstructTextFromSentences(originalText: string, sentenceData: SentenceData[], correctedSentences: string[]): string`

- [ ] **Step 1: Write the failing tests** (`reconstruct.test.ts`)

```ts
import { describe, it, expect } from 'vitest';
import { splitIntoSentences } from './sentences';
import { reconstructTextFromSentences } from './reconstruct';

describe('reconstructTextFromSentences', () => {
  it('preserves single-space gaps', () => {
    const original = 'One. Two.';
    const data = splitIntoSentences(original);
    const result = reconstructTextFromSentences(original, data, ['One!', 'Two?']);
    expect(result).toBe('One! Two?');
  });

  it('preserves newline gaps', () => {
    const original = 'First sentence.\n\nSecond sentence.';
    const data = splitIntoSentences(original);
    const result = reconstructTextFromSentences(original, data, ['First fixed.', 'Second fixed.']);
    expect(result).toBe('First fixed.\n\nSecond fixed.');
  });

  it('preserves leading whitespace before sentences', () => {
    const original = '  Indented start. Next.';
    const data = splitIntoSentences(original);
    const result = reconstructTextFromSentences(original, data, ['Indented fixed.', 'Next fixed.']);
    expect(result).toBe('  Indented fixed. Next fixed.');
  });

  it('returns original text when lengths mismatch', () => {
    const original = 'One. Two.';
    const data = splitIntoSentences(original);
    expect(reconstructTextFromSentences(original, data, ['One!'])).toBe(original);
  });

  it('preserves trailing whitespace after the last sentence', () => {
    const original = 'One. Two.\n\n\n';
    const data = splitIntoSentences(original);
    const result = reconstructTextFromSentences(original, data, ['One!', 'Two?']);
    expect(result).toBe('One! Two?\n\n\n');
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/electron/core/reconstruct.test.ts`
Expected: FAIL — "Cannot find module './reconstruct'"

- [ ] **Step 3: Implement reconstruct.ts**

```ts
import type { SentenceData } from './types';

export function reconstructTextFromSentences(
  originalText: string,
  sentenceData: SentenceData[],
  correctedSentences: string[],
): string {
  if (sentenceData.length !== correctedSentences.length) return originalText;

  const resultParts: string[] = [];
  let lastSpanEnd = 0;

  for (let i = 0; i < sentenceData.length; i++) {
    const sent = sentenceData[i];
    const corrected = correctedSentences[i];
    const { start, end } = sent;
    const spanEnd = sent.spanEnd ?? end;
    const gapBeforeStart = sent.gapBeforeStart ?? lastSpanEnd;

    if (gapBeforeStart > lastSpanEnd) {
      resultParts.push(originalText.slice(lastSpanEnd, gapBeforeStart));
    } else if (start > lastSpanEnd) {
      resultParts.push(originalText.slice(lastSpanEnd, start));
    }

    resultParts.push(corrected);

    if (spanEnd > end) {
      resultParts.push(originalText.slice(end, spanEnd));
    } else if (i < sentenceData.length - 1) {
      if (end < originalText.length) {
        let whitespaceEnd = end;
        while (whitespaceEnd < originalText.length && /\s/.test(originalText[whitespaceEnd])) {
          whitespaceEnd++;
        }
        if (whitespaceEnd > end) {
          resultParts.push(originalText.slice(end, whitespaceEnd));
        } else if (corrected && /[.!?]$/.test(corrected) && whitespaceEnd < originalText.length) {
          const nextChar = originalText[whitespaceEnd];
          // Python: next_char.isalpha() and next_char.isupper()
          if (nextChar && /[A-Z]/.test(nextChar)) {
            resultParts.push(' ');
          }
        }
      }
    }

    lastSpanEnd = spanEnd;
  }

  if (lastSpanEnd < originalText.length) {
    resultParts.push(originalText.slice(lastSpanEnd));
  }

  return resultParts.join('');
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx vitest run src/electron/core/reconstruct.test.ts`
Expected: PASS (all 5)

- [ ] **Step 5: Commit**

```bash
git add src/electron/core && git commit -m "feat: port text reconstruction to TS"
```
### Task 6: Apply suggestion logic

**Files:**
- Create: `src/electron/core/apply.ts`, `src/electron/core/apply.test.ts`

**Interfaces:**
- Consumes: `Suggestion` (types.ts)
- Produces: `applySuggestion(text: string, suggestionIndex: number, suggestions: Suggestion[]): string`, `applySuggestionsBulk(text: string, suggestions: Suggestion[]): string`, `findAllOccurrences(haystack: string, needle: string): Array<[number, number]>`

- [ ] **Step 1: Write the failing tests** (`apply.test.ts`)

```ts
import { describe, it, expect } from 'vitest';
import { applySuggestion, applySuggestionsBulk } from './apply';
import type { Suggestion } from './types';

function makeSuggestion(partial: Partial<Suggestion> & { original: string; corrected: string }): Suggestion {
  return {
    sentence: 'Sentence 1',
    start_index: 0,
    end_index: partial.original.length,
    original_highlighted: '',
    corrected_highlighted: '',
    ...partial,
  };
}

describe('applySuggestion', () => {
  it('replaces the indexed span', () => {
    const s = makeSuggestion({ original: 'dont', corrected: "don't", start_index: 4, end_index: 8 });
    expect(applySuggestion('She dont go.', 0, [s])).toBe("She don't go.");
  });

  it('falls back to nearest occurrence when span mismatches', () => {
    // approx start 99 -> nearest occurrence is the second 'dont' (Python: min by |sp[0]-start|)
    const s = makeSuggestion({ original: 'dont', corrected: "don't", start_index: 99, end_index: 103 });
    expect(applySuggestion('dont dont.', 0, [s])).toBe("dont don't.");
  });

  it('leaves text unchanged when original not found', () => {
    const s = makeSuggestion({ original: 'zzz', corrected: 'aaa' });
    expect(applySuggestion('nothing here.', 0, [s])).toBe('nothing here.');
  });

  it('throws on invalid index', () => {
    const s = makeSuggestion({ original: 'a', corrected: 'b' });
    expect(() => applySuggestion('a', 5, [s])).toThrow();
  });
});

describe('applySuggestionsBulk', () => {
  it('applies non-overlapping suggestions without index drift', () => {
    const s1 = makeSuggestion({ original: 'first', corrected: '1st', start_index: 0, end_index: 5 });
    const s2 = makeSuggestion({ original: 'third', corrected: '3rd', start_index: 13, end_index: 18 });
    expect(applySuggestionsBulk('first second third', [s1, s2])).toBe('1st second 3rd');
  });

  it('keeps rightmost replacement on overlap', () => {
    const s1 = makeSuggestion({ original: 'a b c', corrected: 'X', start_index: 0, end_index: 5 });
    const s2 = makeSuggestion({ original: 'b c', corrected: 'Y', start_index: 2, end_index: 5 });
    expect(applySuggestionsBulk('a b c', [s1, s2])).toBe('a Y');
  });

  it('skips invalid suggestions', () => {
    const s1 = makeSuggestion({ original: 'bad', corrected: 'good', start_index: -1, end_index: 3 });
    const s2 = makeSuggestion({ original: 'ok', corrected: 'fine', start_index: 0, end_index: 2 });
    expect(applySuggestionsBulk('ok', [s1, s2])).toBe('fine');
  });

  it('returns text unchanged for empty list', () => {
    expect(applySuggestionsBulk('hello', [])).toBe('hello');
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/electron/core/apply.test.ts`
Expected: FAIL — "Cannot find module './apply'"

- [ ] **Step 3: Implement apply.ts**

```ts
import type { Suggestion } from './types';

export function findAllOccurrences(haystack: string, needle: string): Array<[number, number]> {
  if (!needle || !haystack) return [];
  const escaped = needle.replace(/[.*+?^${}()|[\]\]/g, '\$&');
  const re = new RegExp(escaped, 'g');
  const matches: Array<[number, number]> = [];
  let match: RegExpExecArray | null;
  while ((match = re.exec(haystack)) !== null) {
    matches.push([match.index, match.index + needle.length]);
    if (match.index === re.lastIndex) re.lastIndex++;
  }
  return matches;
}

function nearestOccurrence(occurrences: Array<[number, number]>, approxStart: number): [number, number] {
  return occurrences.reduce((best, occ) =>
    Math.abs(occ[0] - approxStart) < Math.abs(best[0] - approxStart) ? occ : best,
  );
}

export function applySuggestion(text: string, suggestionIndex: number, suggestions: Suggestion[]): string {
  if (suggestionIndex < 0 || suggestionIndex >= suggestions.length) {
    throw new RangeError('Invalid suggestion index');
  }
  const suggestion = suggestions[suggestionIndex];
  const { start_index: start, end_index: end } = suggestion;

  if (start >= 0 && start <= end && end <= text.length && text.slice(start, end) === suggestion.original) {
    return text.slice(0, start) + suggestion.corrected + text.slice(end);
  }

  const occurrences = findAllOccurrences(text, suggestion.original);
  if (occurrences.length > 0) {
    const [tStart, tEnd] = nearestOccurrence(occurrences, start);
    return text.slice(0, tStart) + suggestion.corrected + text.slice(tEnd);
  }

  return text;
}

export function applySuggestionsBulk(text: string, suggestions: Suggestion[]): string {
  if (suggestions.length === 0) return text;

  const sorted = suggestions
    .filter((s) => s.start_index >= 0 && s.end_index >= 0 && s.start_index <= s.end_index)
    .sort((a, b) => b.start_index - a.start_index);

  const appliedIntervals: Array<[number, number]> = [];
  let result = text;

  for (const s of sorted) {
    const { start_index: start, end_index: end } = s;
    let candidate: [number, number] | null = null;

    if (end <= result.length && result.slice(start, end) === s.original) {
      candidate = [start, end];
    } else {
      const occurrences = findAllOccurrences(result, s.original);
      if (occurrences.length > 0) {
        candidate = nearestOccurrence(occurrences, start);
      }
    }

    if (!candidate) continue;

    const [cStart, cEnd] = candidate;
    const overlaps = appliedIntervals.some(([aStart, aEnd]) => !(cEnd <= aStart || cStart >= aEnd));
    if (overlaps) continue;

    result = result.slice(0, cStart) + s.corrected + result.slice(cEnd);
    appliedIntervals.push([cStart, cEnd]);
  }

  return result;
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx vitest run src/electron/core/apply.test.ts`
Expected: PASS (all 11)

- [ ] **Step 5: Commit**

```bash
git add src/electron/core && git commit -m "feat: port apply-suggestion logic to TS"
```
### Task 7: Correction orchestrator

**Files:**
- Create: `src/electron/core/correction.ts`, `src/electron/core/correction.test.ts`

**Interfaces:**
- Consumes: `splitIntoSentences`, `cleanCorrectedText`, `isOnlyQuoteChange`, `highlightWordDifferences`, `reconstructTextFromSentences`, `SentenceCorrector`
- Produces: `correctText(text: string, corrector: SentenceCorrector): Promise<CorrectionResponse>`, `cleanQuotePunctuation(s: string): string`

- [ ] **Step 1: Write the failing tests** (`correction.test.ts`)

```ts
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/electron/core/correction.test.ts`
Expected: FAIL — "Cannot find module './correction'"

- [ ] **Step 3: Implement correction.ts**

```ts
import { splitIntoSentences } from './sentences';
import { cleanCorrectedText, isOnlyQuoteChange } from './clean';
import { highlightWordDifferences } from './diff';
import { reconstructTextFromSentences } from './reconstruct';
import type { CorrectionResponse, SentenceCorrector } from './types';

export function cleanQuotePunctuation(s: string): string {
  let result = s;
  for (let i = 0; i < 3; i++) {
    const before = result;
    result = result.replace(/([.!?])(["'])\s*\./g, '$1$2');
    result = result.replace(/([.!?])(["'])\s*\1/g, '$1$2');
    if (result === before) break;
  }
  return result;
}

export async function correctText(text: string, corrector: SentenceCorrector): Promise<CorrectionResponse> {
  const trimmed = text.trim();
  if (!trimmed) return { suggestions: [], correctedText: '' };

  const sentenceData = splitIntoSentences(text);
  const suggestions: CorrectionResponse['suggestions'] = [];
  const correctedSentences: string[] = [];

  for (let i = 0; i < sentenceData.length; i++) {
    const sent = sentenceData[i];
    const sentence = sent.text;

    if (sentence.length < 2) {
      correctedSentences.push(sentence);
      continue;
    }

    let corrected = await corrector.correct(sentence);
    corrected = cleanCorrectedText(corrected, sentence);

    if (isOnlyQuoteChange(sentence, corrected)) {
      corrected = sentence;
    }
    if (corrected.length > sentence.length * 2) {
      corrected = sentence;
    }

    corrected = cleanQuotePunctuation(corrected);
    correctedSentences.push(corrected);

    if (
      corrected.toLowerCase().trim() !== sentence.toLowerCase().trim() &&
      corrected.trim() !== sentence.trim() &&
      corrected.length <= sentence.length * 1.5 &&
      !isOnlyQuoteChange(sentence, corrected)
    ) {
      const highlighted = highlightWordDifferences(sentence, corrected);
      suggestions.push({
        original: sentence,
        corrected,
        sentence: `Sentence ${i + 1}`,
        start_index: sent.start,
        end_index: sent.end,
        original_highlighted: highlighted.originalHighlighted,
        corrected_highlighted: highlighted.correctedHighlighted,
      });
    }
  }

  const correctedText = cleanQuotePunctuation(
    reconstructTextFromSentences(text, sentenceData, correctedSentences),
  );
  return { suggestions, correctedText };
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx vitest run src/electron/core/correction.test.ts`
Expected: PASS (all 8)

- [ ] **Step 5: Commit**

```bash
git add src/electron/core && git commit -m "feat: port correction orchestrator to TS"
```
### Task 8: IPC contract, preload, and typed window.api

**Files:**
- Create: `src/electron/ipc-types.ts`, `src/electron/preload.cjs`, `src/ui/electron-api.d.ts`, `src/ui/api.ts`

**Interfaces:**
- Consumes: `Suggestion`, `CorrectionResponse` (core/types.ts)
- Produces: `ModelDownloadRequest { url, fileName }`, `DownloadProgress { percent, transferred, total }`, `ModelState`, `ModelStatus { state, modelName? }`, `IpcApi` (all in ipc-types.ts); `window.api: IpcApi` in renderer

- [ ] **Step 1: Write ipc-types.ts**

```ts
import type { CorrectionResponse, Suggestion } from './core/types';

export interface ModelDownloadRequest {
  url: string;
  fileName: string;
}

export interface DownloadProgress {
  percent: number;
  transferred: number;
  total: number;
}

export type ModelState = 'ready' | 'missing' | 'downloading' | 'error';

export interface ModelStatus {
  state: ModelState;
  modelName?: string;
}

export interface ApplySuggestionRequest {
  originalText: string;
  suggestionIndex: number;
  suggestions: Suggestion[];
}

export interface ApplyManyRequest {
  originalText: string;
  suggestions: Suggestion[];
}

export interface IpcApi {
  correct(text: string): Promise<CorrectionResponse>;
  applySuggestion(args: ApplySuggestionRequest): Promise<{ correctedText: string }>;
  applyMany(args: ApplyManyRequest): Promise<{ correctedText: string }>;
  modelStatus(): Promise<ModelStatus>;
  downloadModel(args: ModelDownloadRequest): Promise<void>;
  cancelDownload(): Promise<void>;
  onDownloadProgress(cb: (progress: DownloadProgress) => void): () => void;
}
```

- [ ] **Step 2: Write preload.cjs** (plain CommonJS — sandboxed preloads cannot be ESM)

```js
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('api', {
  correct: (text) => ipcRenderer.invoke('text:correct', { text }),
  applySuggestion: (args) => ipcRenderer.invoke('suggestion:apply', args),
  applyMany: (args) => ipcRenderer.invoke('suggestion:applyMany', args),
  modelStatus: () => ipcRenderer.invoke('model:status'),
  downloadModel: (args) => ipcRenderer.invoke('model:download', args),
  cancelDownload: () => ipcRenderer.invoke('model:cancel-download'),
  onDownloadProgress: (cb) => {
    const listener = (_event, progress) => cb(progress);
    ipcRenderer.on('model:download-progress', listener);
    return () => ipcRenderer.removeListener('model:download-progress', listener);
  },
});
```

- [ ] **Step 3: Write the renderer type declaration** (`src/ui/electron-api.d.ts`)

```ts
import type { IpcApi } from '../electron/ipc-types';

declare global {
  interface Window {
    api: IpcApi;
  }
}

export {};
```

- [ ] **Step 4: Write the renderer API wrapper** (`src/ui/api.ts`)

```ts
export const api = window.api;
```

- [ ] **Step 5: Verify types compile on both sides**

Run: `npx tsc -b`
Expected: PASS — ipc-types.ts type-checks under the electron project; electron-api.d.ts type-checks under the app project (import pulls the file into the program; both configs are noEmit on the app side).

- [ ] **Step 6: Commit**

```bash
git add src/electron/ipc-types.ts src/electron/preload.cjs src/ui/electron-api.d.ts src/ui/api.ts && git commit -m "feat: add typed IPC contract and preload bridge"
```
### Task 9: Model manager

**Files:**
- Create: `src/electron/modelManager.ts`, `src/electron/modelManager.test.ts`

**Interfaces:**
- Consumes: `ModelStatus`, `ModelState`, `DownloadProgress` (ipc-types.ts)
- Produces: `ModelDownloader { download(): Promise<unknown>; cancel(): Promise<void>; onProgress(cb: (p: { transferredBytes: number; totalBytes: number }) => void): void }`, `DownloaderFactory { create(model: { url: string; dir: string; fileName: string }): ModelDownloader | Promise<ModelDownloader> }` (async because node-llama-cpp is ESM-with-TLA and must be loaded via dynamic `import()`), `ModelManager` class with `getStatus(): ModelStatus`, `download(url: string, fileName: string): Promise<void>`, `cancelDownload(): Promise<void>`, `getModelPath(): string | null`

- [ ] **Step 1: Write the failing tests** (`modelManager.test.ts`) — no real downloader, no network

```ts
import { describe, it, expect, vi } from 'vitest';
import { ModelManager, type ModelDownloader, type DownloaderFactory } from './modelManager';

function fakeDownloader() {
  const callbacks = new Set<(p: { transferredBytes: number; totalBytes: number }) => void>();
  return {
    download: vi.fn().mockResolvedValue(undefined),
    cancel: vi.fn().mockResolvedValue(undefined),
    onProgress: vi.fn().mockImplementation((cb) => {
      callbacks.add(cb);
    }),
    emit(transferredBytes: number, totalBytes: number) {
      callbacks.forEach((cb) => cb({ transferredBytes, totalBytes }));
    },
  };
}

function makeManager(overrides: { files?: string[]; downloader?: ReturnType<typeof fakeDownloader> } = {}) {
  const downloader = overrides.downloader ?? fakeDownloader();
  const factory: DownloaderFactory = {
    create: vi.fn().mockReturnValue(downloader),
  };
  const manager = new ModelManager({
    modelsDir: '/fake/models',
    listModels: async () => overrides.files ?? [],
    factory,
  });
  return { manager, downloader, factory };
}

describe('ModelManager', () => {
  it('reports missing when no model files exist', async () => {
    const { manager } = makeManager({ files: [] });
    expect(await manager.getStatus()).toEqual({ state: 'missing' });
  });

  it('reports ready when a gguf exists', async () => {
    const { manager } = makeManager({ files: ['GRMR-V3-G4B-Q4_K_M.gguf'] });
    const status = await manager.getStatus();
    expect(status.state).toBe('ready');
    expect(status.modelName).toBe('GRMR-V3-G4B-Q4_K_M.gguf');
  });

  it('downloads with progress updates', async () => {
    const downloader = fakeDownloader();
    const { manager, factory } = makeManager({ downloader });

    const progressEvents: Array<{ percent: number }> = [];
    manager.onDownloadProgress((p) => progressEvents.push(p));

    const promise = manager.download('https://example.com/model.gguf', 'model.gguf');
    expect(factory.create).toHaveBeenCalledWith({
      url: 'https://example.com/model.gguf',
      dir: '/fake/models',
      fileName: 'model.gguf',
    });
    expect((await manager.getStatus()).state).toBe('downloading');

    downloader.emit(100, 200);
    await promise;
    expect(progressEvents).toEqual([{ percent: 50, transferred: 100, total: 200 }]);
    expect((await manager.getStatus()).state).toBe('ready');
  });

  it('forwards cancel to the downloader', async () => {
    const downloader = fakeDownloader();
    const { manager } = makeManager({ downloader });
    await manager.download('https://example.com/model.gguf', 'model.gguf');
    await manager.cancelDownload();
    expect(downloader.cancel).toHaveBeenCalled();
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/electron/modelManager.test.ts`
Expected: FAIL — "Cannot find module './modelManager'"

- [ ] **Step 3: Implement modelManager.ts**

```ts
import path from 'path';
import type { DownloadProgress, ModelStatus } from './ipc-types';

export interface ModelDownloader {
  download(): Promise<unknown>;
  cancel(): Promise<void>;
  onProgress(cb: (p: { transferredBytes: number; totalBytes: number }) => void): void;
}

export interface DownloaderFactory {
  create(model: { url: string; dir: string; fileName: string }): ModelDownloader;
}

export interface ModelManagerOptions {
  modelsDir: string;
  listModels?: () => Promise<string[]>;
  factory?: DownloaderFactory;
}

export class ModelManager {
  private readonly modelsDir: string;
  private readonly listModels: () => Promise<string[]>;
  private readonly factory: DownloaderFactory;
  private currentDownload: ModelDownloader | null = null;
  private progressListeners = new Set<(p: DownloadProgress) => void>();
  private state: ModelStatus['state'] = 'missing';
  private modelName: string | undefined;

  constructor(options: ModelManagerOptions) {
    this.modelsDir = options.modelsDir;
    this.listModels = options.listModels ?? (async () => {
      const fs = await import('fs/promises');
      try {
        const entries = await fs.readdir(this.modelsDir);
        return entries.filter((name) => name.endsWith('.gguf'));
      } catch {
        return [];
      }
    });
    this.factory = options.factory ?? createNodeLlamaDownloaderFactory();
  }

  getModelPath(): string | null {
    return this.modelName ? path.join(this.modelsDir, this.modelName) : null;
  }

  async getStatus(): Promise<ModelStatus> {
    if (this.state === 'downloading') return { state: 'downloading', modelName: this.modelName };
    if (this.state === 'error') return { state: 'error', modelName: this.modelName };
    if (this.state === 'ready' && this.modelName) return { state: 'ready', modelName: this.modelName };
    // Initial state: scan the models dir (modelName presence alone is NOT
    // proof of readiness — a canceled download leaves modelName set).
    const files = await this.listModels();
    if (files.length > 0) {
      this.modelName = files[0];
      return { state: 'ready', modelName: files[0] };
    }
    return { state: 'missing' };
  }

  onDownloadProgress(cb: (p: DownloadProgress) => void): () => void {
    this.progressListeners.add(cb);
    return () => this.progressListeners.delete(cb);
  }

  async download(url: string, fileName: string): Promise<void> {
    const downloader = await this.factory.create({ url, dir: this.modelsDir, fileName });
    this.currentDownload = downloader;
    this.state = 'downloading';
    this.modelName = fileName;

    downloader.onProgress(({ transferredBytes, totalBytes }) => {
      const percent = totalBytes > 0 ? Math.round((transferredBytes / totalBytes) * 100) : 0;
      this.emitProgress({ percent, transferred: transferredBytes, total: totalBytes });
    });

    try {
      await downloader.download();
      this.state = 'ready';
    } catch (error) {
      this.state = 'error';
      throw error;
    } finally {
      this.currentDownload = null;
    }
  }

  async cancelDownload(): Promise<void> {
    if (this.currentDownload) {
      await this.currentDownload.cancel();
      this.currentDownload = null;
      this.state = 'missing';
    }
  }

  private emitProgress(p: DownloadProgress): void {
    this.progressListeners.forEach((cb) => cb(p));
  }
}

// Adapter around node-llama-cpp's createModelDownloader (v3.19 API: modelUri/dirPath,
// async factory, progress via onProgress option). node-llama-cpp is ESM with top-level
// await, so it must be loaded via dynamic import().
export function createNodeLlamaDownloaderFactory(): DownloaderFactory {
  return {
    async create({ url, dir, fileName }) {
      const { createModelDownloader } = await import('node-llama-cpp');
      const callbacks = new Set<(p: { transferredBytes: number; totalBytes: number }) => void>();
      const downloader = await createModelDownloader({
        modelUri: url,
        dirPath: dir,
        fileName,
        onProgress: (status) => {
          const progress = {
            transferredBytes: status.downloadedSize,
            totalBytes: status.totalSize,
          };
          callbacks.forEach((cb) => cb(progress));
        },
      });
      return {
        download: () => downloader.download(),
        cancel: () => downloader.cancel(),
        onProgress: (cb) => {
          callbacks.add(cb);
          return () => callbacks.delete(cb);
        },
      };
    },
  };
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx vitest run src/electron/modelManager.test.ts`
Expected: PASS (all 4)

- [ ] **Step 5: Commit**

```bash
git add src/electron/modelManager.ts src/electron/modelManager.test.ts && git commit -m "feat: add model manager with download lifecycle"
```
### Task 10: Llama inference service

**Files:**
- Create: `src/electron/llamaService.ts`

**Interfaces:**
- Consumes: `SentenceCorrector` (core/types.ts), `ModelManager`
- Produces: `LlamaCorrectionService implements SentenceCorrector` — constructor `(modelManager: ModelManager)`, `ensureLoaded(): Promise<void>`, `correct(sentence: string): Promise<string>` (returns original sentence on any error), `getStatus(): ModelStatus`

- [ ] **Step 1: Implement llamaService.ts** (compile-only — no model, no inference in tests)

```ts
import { getLlama, LlamaChatSession } from 'node-llama-cpp';
import type { ModelStatus } from './ipc-types';
import type { ModelManager } from './modelManager';
import type { SentenceCorrector } from './core/types';

const SYSTEM_PROMPT =
  'You are a grammar correction assistant. Correct the grammar, spelling, and punctuation of the given text. Return only the corrected text without any additional explanations or prefixes.';

const STOP_SEQUENCES = ['<|endoftext|>', '<|corrected_end|>', '\n\n', 'Corrected:', 'Here is'];

// NOTE: sampling params verified against installed node-llama-cpp v3.19 types (Task 1 step 6).
// Python mapping: temperature=0.7, top_p=0.95, top_k=40, min_p=0.01,
// frequency_penalty=0.0/presence_penalty=0.0 -> repeatPenalty=1.0, max_tokens=len+20.
// Stop sequences are passed per-prompt via `customStopTriggers` (no constructor option).
export class LlamaCorrectionService implements SentenceCorrector {
  private session: LlamaChatSession | null = null;
  private loadError: unknown = null;
  private loading: Promise<void> | null = null;

  constructor(private readonly modelManager: ModelManager) {}

  async ensureLoaded(): Promise<void> {
    if (this.session) return;
    if (this.loading) return this.loading;

    this.loading = (async () => {
      const modelPath = this.modelManager.getModelPath();
      if (!modelPath) throw new Error('No model found');
      const llama = await getLlama(); // auto-detects CUDA / Metal / Vulkan / CPU
      const model = await llama.loadModel({ modelPath, contextSize: 4096 });
      const context = await model.createContext();
      this.session = new LlamaChatSession({
        contextSequence: context.createContextSequence(),
        systemPrompt: SYSTEM_PROMPT,
        temperature: 0.7,
        topK: 40,
        topP: 0.95,
        minP: 0.01,
        repeatPenalty: 1.0,
      });
    })();

    try {
      await this.loading;
      this.loadError = null;
    } catch (error) {
      this.loadError = error;
      this.loading = null;
      throw error;
    }
  }

  async correct(sentence: string): Promise<string> {
    try {
      await this.ensureLoaded();
      const response = await this.session!.prompt(sentence, {
        maxTokens: sentence.length + 20,
        customStopTriggers: STOP_SEQUENCES,
      });
      return response.trim();
    } catch (error) {
      console.error(`Error correcting sentence '${sentence}':`, error);
      return sentence;
    }
  }

  getStatus(): ModelStatus {
    if (this.loadError) return { state: 'error', modelName: this.modelManager.getModelPath() ?? undefined };
    if (this.session) return { state: 'ready', modelName: this.modelManager.getModelPath() ?? undefined };
    return { state: 'missing' };
  }
}
```

- [ ] **Step 2: Verify against installed node-llama-cpp API**

Check every member used above against the installed type definitions (`node_modules/node-llama-cpp/dist/index.d.ts`): `getLlama`, `LlamaChatSession` constructor options, `loadModel({ modelPath, contextSize })`, `createContextSequence`, `prompt(message, { maxTokens })`, `stopGenerationTriggers`, `repeatPenalty`.
If any name differs (e.g. `maxTokens` vs `maxTokens` in prompt options, or `stopGenerationTriggers` renamed), adapt the code and record the difference in a comment.

- [ ] **Step 3: Verify it compiles**

Run: `npx tsc -b`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/electron/llamaService.ts && git commit -m "feat: add llama inference service"
```

### Task 11: IPC handlers + main process wiring

**Files:**
- Create: `src/electron/schemas.ts`, `src/electron/schemas.test.ts`, `src/electron/ipc.ts`
- Modify: `src/electron/main.ts`

Note: zod schemas live in `schemas.ts` (no electron import) so vitest can test them in a plain node environment; `ipc.ts` only wires handlers.

**Interfaces:**
- Consumes: `correctText`, `applySuggestion`, `applySuggestionsBulk`, `ModelManager`, `SentenceCorrector`, `IpcApi` channel names
- Produces: `schemas.ts` (zod schemas, electron-free), `registerIpcHandlers(modelManager: ModelManager, corrector: SentenceCorrector): void`; main.ts wires window + preload + services

- [ ] **Step 1: Write the failing tests** (`schemas.test.ts`) — zod schemas only, no Electron import

```ts
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/electron/schemas.test.ts`
Expected: FAIL — "Cannot find module './schemas'"

- [ ] **Step 3: Implement schemas.ts** (electron-free)

```ts
import { z } from 'zod';

export const suggestionSchema = z.object({
  original: z.string(),
  corrected: z.string(),
  sentence: z.string(),
  start_index: z.number(),
  end_index: z.number(),
  original_highlighted: z.string(),
  corrected_highlighted: z.string(),
});

export const correctRequestSchema = z.object({ text: z.string() });

export const applyRequestSchema = z.object({
  originalText: z.string(),
  suggestionIndex: z.number().int().nonnegative(),
  suggestions: z.array(suggestionSchema),
});

export const applyManySchema = z.object({
  originalText: z.string(),
  suggestions: z.array(suggestionSchema),
});

export const downloadRequestSchema = z.object({
  url: z.string().url(),
  fileName: z.string(),
});
```

- [ ] **Step 4: Implement ipc.ts** (imports schemas; not covered by unit tests — thin wiring)

```ts
import { ipcMain } from 'electron';
import { correctText } from './core/correction';
import { applySuggestion, applySuggestionsBulk } from './core/apply';
import { correctRequestSchema, applyRequestSchema, applyManySchema, downloadRequestSchema } from './schemas';
import type { ModelManager } from './modelManager';
import type { SentenceCorrector } from './core/types';

export const suggestionSchema = z.object({
  original: z.string(),
  corrected: z.string(),
  sentence: z.string(),
  start_index: z.number(),
  end_index: z.number(),
  original_highlighted: z.string(),
  corrected_highlighted: z.string(),
});

export const correctRequestSchema = z.object({ text: z.string() });

export const applyRequestSchema = z.object({
  originalText: z.string(),
  suggestionIndex: z.number().int().nonnegative(),
  suggestions: z.array(suggestionSchema),
});

export const applyManySchema = z.object({
  originalText: z.string(),
  suggestions: z.array(suggestionSchema),
});

export const downloadRequestSchema = z.object({
  url: z.string().url(),
  fileName: z.string(),
});

export function registerIpcHandlers(modelManager: ModelManager, corrector: SentenceCorrector): void {
  ipcMain.handle('model:status', () => modelManager.getStatus());

  ipcMain.handle('model:download', async (_event, raw) => {
    const { url, fileName } = downloadRequestSchema.parse(raw);
    await modelManager.download(url, fileName);
  });

  ipcMain.handle('model:cancel-download', () => modelManager.cancelDownload());

  ipcMain.handle('text:correct', async (_event, raw) => {
    const { text } = correctRequestSchema.parse(raw);
    return correctText(text, corrector);
  });

  ipcMain.handle('suggestion:apply', async (_event, raw) => {
    const { originalText, suggestionIndex, suggestions } = applyRequestSchema.parse(raw);
    return { correctedText: applySuggestion(originalText, suggestionIndex, suggestions) };
  });

  ipcMain.handle('suggestion:applyMany', async (_event, raw) => {
    const { originalText, suggestions } = applyManySchema.parse(raw);
    return { correctedText: applySuggestionsBulk(originalText, suggestions) };
  });
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `npx vitest run src/electron/schemas.test.ts`
Expected: PASS (all 4)

- [ ] **Step 6: Rewrite main.ts** (full replacement)

```ts
import { app, BrowserWindow } from 'electron';
import path from 'path';
import { registerIpcHandlers } from './ipc';
import { ModelManager } from './modelManager';
import { LlamaCorrectionService } from './llamaService';
import type { DownloadProgress } from './ipc-types';

function getModelsDir(): string {
  return app.isPackaged
    ? path.join(app.getPath('userData'), 'models')
    : path.join(app.getAppPath(), 'models');
}

const modelManager = new ModelManager({ modelsDir: getModelsDir() });
const llamaService = new LlamaCorrectionService(modelManager);

app.on('ready', async () => {
  registerIpcHandlers(modelManager, llamaService);

  modelManager.onDownloadProgress((progress: DownloadProgress) => {
    BrowserWindow.getAllWindows().forEach((win) => {
      win.webContents.send('model:download-progress', progress);
    });
  });

  const mainWindow = new BrowserWindow({
    width: 1100,
    height: 800,
    webPreferences: {
      preload: path.join(import.meta.dirname, 'preload.cjs'), // ESM: no __dirname; Electron 43 has import.meta.dirname
      contextIsolation: true,
      sandbox: true,
      nodeIntegration: false,
    },
  });

  mainWindow.loadFile(path.join(app.getAppPath(), 'dist-react', 'index.html'));

  const status = await modelManager.getStatus();
  if (status.state === 'ready' && status.modelName) {
    llamaService.ensureLoaded().catch((error) => {
      console.error('Failed to preload model:', error);
    });
  }
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});
```

- [ ] **Step 7: Verify**

Run: `npx tsc -b`
Expected: PASS. Note: `app.getPath` is used at module scope via `getModelsDir()` — verify this is safe (it is: `app.getPath` works before `ready` for `userData`; `getAppPath` also works pre-ready). If not, move construction into the `ready` handler.

- [ ] **Step 8: Commit**

```bash
git add src/electron/schemas.ts src/electron/schemas.test.ts src/electron/ipc.ts src/electron/main.ts && git commit -m "feat: wire IPC handlers and main process"
```

**Notes from execution (already applied):**
- `tsc -b` (root) does NOT compile the electron project (root tsconfig excludes `src/electron`). Always verify with `npx tsc -p src/electron/tsconfig.json` or `npm run build:electron`.
- nodenext requires `.js` extensions on all relative imports in emitted ESM.
- `preload.cjs` is not emitted by tsc — `build:electron` copies it: `tsc -p src/electron/tsconfig.json && node -e "require('fs').copyFileSync('src/electron/preload.cjs','dist-electron/preload.cjs')"`
- ESM main process: no `__dirname` — use `import.meta.dirname`.
- node-llama-cpp v3.19: sampling params go on `session.prompt()`, `contextSize` on `model.createContext()`, sequences via `context.getSequence()`, `repeatPenalty` is `{ penalty: 1 }` to disable, `erasableSyntaxOnly` forbids constructor parameter properties.
### Task 12: Model download screen

**Files:**
- Create: `src/ui/components/ModelGate.tsx`, `src/ui/components/ModelGate.css`
- Modify: `src/ui/App.tsx` (rewrite), `src/ui/main.tsx` (import App.css)

**Interfaces:**
- Consumes: `api` (src/ui/api.ts), `ModelStatus`, `DownloadProgress` types
- Produces: `ModelGate` component with props `{ status: ModelStatus }`

- [ ] **Step 1: Write ModelGate.tsx**

```tsx
import { useEffect, useState } from 'react';
import { api } from '../api';
import type { DownloadProgress, ModelStatus } from '../../electron/ipc-types';
import './ModelGate.css';

const MODELS = [
  {
    fileName: 'GRMR-V3-G4B-Q4_K_M.gguf',
    url: 'https://huggingface.co/icecubetr/GRMR-V3-G4B-GGUF/resolve/main/GRMR-V3-G4B-Q4_K_M.gguf',
    label: 'Q4_K_M — Recommended',
    detail: 'Faster, smaller download',
  },
  {
    fileName: 'GRMR-V3-G4B-Q8_0.gguf',
    url: 'https://huggingface.co/icecubetr/GRMR-V3-G4B-GGUF/resolve/main/GRMR-V3-G4B-Q8_0.gguf',
    label: 'Q8_0 — Highest quality',
    detail: 'Slower, larger download',
  },
];

export function ModelGate({ status }: { status: ModelStatus }) {
  const [selected, setSelected] = useState(0);
  const [progress, setProgress] = useState<DownloadProgress | null>(null);
  const [downloading, setDownloading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!downloading) return;
    const unsubscribe = api.onDownloadProgress(setProgress);
    return unsubscribe;
  }, [downloading]);

  async function handleDownload() {
    const model = MODELS[selected];
    setDownloading(true);
    setError(null);
    try {
      await api.downloadModel({ url: model.url, fileName: model.fileName });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setDownloading(false);
      setProgress(null);
    }
  }

  async function handleCancel() {
    await api.cancelDownload();
    setDownloading(false);
    setProgress(null);
  }

  return (
    <div className="model-gate">
      <h1>GrammarLLM</h1>
      <p className="model-gate-subtitle">
        No model detected. Choose a model to download and get started.
      </p>

      <div className="model-options">
        {MODELS.map((model, index) => (
          <label key={model.fileName} className={`model-option ${selected === index ? 'selected' : ''}`}>
            <input
              type="radio"
              name="model"
              checked={selected === index}
              onChange={() => setSelected(index)}
              disabled={downloading}
            />
            <div>
              <strong>{model.label}</strong>
              <span>{model.detail}</span>
            </div>
          </label>
        ))}
      </div>

      {status.state === 'error' && <p className="model-gate-error">Model failed to load: {status.modelName}</p>}
      {error && <p className="model-gate-error">{error}</p>}

      {progress ? (
        <div className="download-area">
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: `${progress.percent}%` }} />
          </div>
          <p>
            {progress.percent}% — {(progress.transferred / 1024 / 1024).toFixed(0)} MB /{' '}
            {(progress.total / 1024 / 1024).toFixed(0)} MB
          </p>
          <button onClick={handleCancel}>Cancel</button>
        </div>
      ) : (
        <button className="download-btn" onClick={handleDownload} disabled={downloading}>
          {downloading ? 'Downloading…' : 'Download model'}
        </button>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Write ModelGate.css** (compact styles; full theme ported in Task 13)

```css
.model-gate {
  max-width: 560px;
  margin: 0 auto;
  padding: 48px 24px;
  text-align: center;
}
.model-gate-subtitle { opacity: 0.8; margin-bottom: 24px; }
.model-options { display: flex; flex-direction: column; gap: 12px; margin-bottom: 24px; }
.model-option {
  display: flex; gap: 12px; align-items: flex-start;
  padding: 16px; border-radius: 10px; cursor: pointer;
  border: 2px solid transparent; background: rgba(128, 128, 128, 0.08);
}
.model-option.selected { border-color: #667eea; }
.model-option div { display: flex; flex-direction: column; text-align: left; }
.model-option span { opacity: 0.7; font-size: 0.9em; }
.download-btn {
  padding: 12px 28px; border: none; border-radius: 8px;
  background: #667eea; color: white; font-size: 1em; cursor: pointer;
}
.download-btn:disabled { opacity: 0.6; cursor: not-allowed; }
.progress-bar {
  height: 10px; border-radius: 5px; overflow: hidden;
  background: rgba(128, 128, 128, 0.2); margin-bottom: 8px;
}
.progress-fill { height: 100%; background: #667eea; transition: width 0.2s; }
.model-gate-error { color: #e53e3e; }
```

- [ ] **Step 3: Rewrite App.tsx** — status polling until a model is ready

```tsx
import { useEffect, useState } from 'react';
import { api } from './api';
import { ModelGate } from './components/ModelGate';
import type { ModelStatus } from '../electron/ipc-types';

export default function App() {
  const [status, setStatus] = useState<ModelStatus | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function poll() {
      const next = await api.modelStatus();
      if (!cancelled) setStatus(next);
    }
    poll();
    const timer = setInterval(poll, 2000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  if (!status) return <div className="app-loading">Loading…</div>;
  if (status.state === 'missing' || status.state === 'downloading' || status.state === 'error') {
    return <ModelGate status={status} />;
  }
  return <GrammarApp />;
}

// Placeholder until Task 13 — keeps the app compiling.
function GrammarApp() {
  return <div>Grammar app coming in Task 13</div>;
}
```

- [ ] **Step 4: Verify compile**

Run: `npx tsc -b`
Expected: PASS. (Renderer still shows placeholder until Task 13.)

- [ ] **Step 5: Commit**

```bash
git add src/ui && git commit -m "feat: add model download screen"
```
### Task 13: Main grammar UI (editor, suggestions, score, PDF, theme)

**Files:**
- Create: `src/ui/components/GrammarApp.tsx`, `SuggestionsList.tsx`, `ScoreBadge.tsx`, `ReportButton.tsx`, `ThemeToggle.tsx`, `pdf.ts`
- Modify: `src/ui/App.tsx` (use real GrammarApp), `src/ui/App.css` (full port), `src/ui/main.tsx`, `index.html` (title)

**Interfaces:**
- Consumes: `api`, `CorrectionResponse`, `Suggestion`, `applySuggestion`/`applyMany` IPC
- Produces: `GrammarApp` default export; `generatePdfReport(suggestions: Suggestion[], score: number): void`

- [ ] **Step 1: Write GrammarApp.tsx** (port of script.js state logic)

```tsx
import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../api';
import type { CorrectionResponse, Suggestion } from '../../electron/core/types';
import { SuggestionsList } from './SuggestionsList';
import { ScoreBadge } from './ScoreBadge';
import { ReportButton } from './ReportButton';
import { ThemeToggle } from './ThemeToggle';

export default function GrammarApp() {
  const [text, setText] = useState('');
  const [corrections, setCorrections] = useState<CorrectionResponse | null>(null);
  const [applied, setApplied] = useState<Set<number>>(new Set());
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<{ message: string; isError?: boolean } | null>(null);
  const [score, setScore] = useState<number | null>(null);
  const [lastCheckedText, setLastCheckedText] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const lastCaretRef = useRef(0);

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        event.preventDefault();
        void handleCheck();
      }
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  });
    const input = text.trim();
    if (!input) {
      setToast({ message: 'Please enter some text to check grammar.', isError: true });
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const result = await api.correct(input);
      setCorrections(result);
      setApplied(new Set());
      setLastCheckedText(input);
      setScore(computeScore(input, result.suggestions));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setScore(null);
    } finally {
      setLoading(false);
    }
  }

  function handleTextChange(next: string) {
    setText(next);
    const isSuggestionApply = next !== lastCheckedText && Math.abs(next.length - (lastCheckedText?.length ?? 0)) < 20;
    if (!isSuggestionApply && corrections && next !== lastCheckedText) {
      setCorrections(null);
      setApplied(new Set());
      setScore(null);
    }
  }

  async function handleApply(index: number) {
    if (!corrections?.suggestions[index]) return;
    try {
      const result = await api.applySuggestion({
        originalText: text,
        suggestionIndex: index,
        suggestions: corrections.suggestions,
      });
      setText(result.correctedText);
      setApplied((prev) => new Set(prev).add(index));
      setToast({ message: `Applied correction for ${corrections.suggestions[index].sentence}` });
    } catch (e) {
      setToast({ message: e instanceof Error ? e.message : 'Failed to apply', isError: true });
    }
  }

  const handleSuggestionHover = useCallback((suggestion: Suggestion) => {
    const el = textareaRef.current;
    if (!el) return;
    const approxStart = suggestion.start_index;
    const bestSpan = findBestOccurrence(el.value, suggestion.original, approxStart);
    if (bestSpan) {
      el.focus();
      el.setSelectionRange(bestSpan[0], bestSpan[1]);
    }
  }, []);

  const handleSuggestionLeave = useCallback(() => {
    const el = textareaRef.current;
    if (el) {
      const caret = Math.min(lastCaretRef.current, el.value.length);
      el.setSelectionRange(caret, caret);
    }
  }, []);

  return (
    <div className="app-shell">
      <header className="app-header">
        <h1>GrammarLLM</h1>
        <div className="header-actions">
          {score !== null && <ScoreBadge score={score} />}
          <ReportButton suggestions={corrections?.suggestions ?? []} score={score} />
          <ThemeToggle />
        </div>
      </header>

      <main className="app-main">
        <section className="editor-section">
          <textarea
            ref={textareaRef}
            value={text}
            placeholder="Type or paste your text here, then press Ctrl+Enter or click Check Grammar"
            onChange={(e) => handleTextChange(e.target.value)}
            onKeyUp={(e) => { lastCaretRef.current = e.currentTarget.selectionStart; }}
            onClick={(e) => { lastCaretRef.current = e.currentTarget.selectionStart; }}
          />
          <div className="editor-toolbar">
            <button className="check-btn" onClick={() => void handleCheck()} disabled={loading}>
              {loading ? 'Checking…' : 'Check Grammar'}
            </button>
            <button className="clear-btn" onClick={() => {
              setText(''); setCorrections(null); setApplied(new Set()); setScore(null); setError(null);
            }}>
              Clear
            </button>
          </div>
        </section>

        <section className="suggestions-section">
          <SuggestionsList
            suggestions={corrections?.suggestions ?? []}
            applied={applied}
            loading={loading}
            error={error}
            onApply={handleApply}
            onHover={handleSuggestionHover}
            onLeave={handleSuggestionLeave}
          />
        </section>
      </main>

      {toast && <Toast message={toast.message} isError={toast.isError} onDone={() => setToast(null)} />}
    </div>
  );
}

export function computeScore(text: string, suggestions: Suggestion[]): number {
  const words = text.trim().split(/\s+/).filter((w) => w.length > 0);
  const totalWords = words.length;
  let errorCount = 0;
  for (const s of suggestions) {
    const matches = s.original_highlighted.match(/<span class="error-word">/g);
    if (matches) errorCount += matches.length;
  }
  if (totalWords === 0) return 100;
  return Math.max(0, Math.round(100 * (1 - errorCount / totalWords)));
}

export function findBestOccurrence(haystack: string, needle: string, approxIndex: number): [number, number] | null {
  if (!needle || !haystack) return null;
  const escaped = needle.replace(/[.*+?^${}()|[\]\]/g, '\$&');
  const re = new RegExp(escaped, 'g');
  const matches: Array<[number, number]> = [];
  let m: RegExpExecArray | null;
  while ((m = re.exec(haystack)) !== null) {
    matches.push([m.index, m.index + needle.length]);
    if (m.index === re.lastIndex) re.lastIndex++;
  }
  if (matches.length === 0) return null;
  return matches.reduce((best, occ) =>
    Math.abs(occ[0] - approxIndex) < Math.abs(best[0] - approxIndex) ? occ : best,
  );
}

function Toast({ message, isError, onDone }: { message: string; isError?: boolean; onDone: () => void }) {
  useEffect(() => {
    const timer = setTimeout(onDone, 3000);
    return () => clearTimeout(timer);
  }, [onDone]);
  return <div className={`toast ${isError ? 'toast-error' : ''}`}>{message}</div>;
}
```
- [ ] **Step 2: Write SuggestionsList.tsx**

```tsx
import type { Suggestion } from '../../electron/core/types';

interface Props {
  suggestions: Suggestion[];
  applied: Set<number>;
  loading: boolean;
  error: string | null;
  onApply: (index: number) => void;
  onHover: (suggestion: Suggestion) => void;
  onLeave: () => void;
}

export function SuggestionsList({ suggestions, applied, loading, error, onApply, onHover, onLeave }: Props) {
  if (loading) {
    return <div className="empty-state"><p>Checking grammar…</p></div>;
  }
  if (error) {
    return <div className="empty-state error-state"><p>Error checking grammar</p><small>{error}</small></div>;
  }

  const unapplied = suggestions.filter((_, index) => !applied.has(index));

  if (suggestions.length === 0) {
    return <div className="empty-state"><p>No grammar issues found</p><small>Your text looks great!</small></div>;
  }
  if (unapplied.length === 0) {
    return <div className="empty-state"><p>All suggestions applied!</p><small>Your text looks great</small></div>;
  }

  return (
    <div className="suggestions-list">
      {suggestions.map((suggestion, index) => {
        if (applied.has(index)) return null;
        return (
          <div
            key={index}
            className="suggestion-item"
            onMouseEnter={() => onHover(suggestion)}
            onMouseLeave={onLeave}
            onFocus={() => onHover(suggestion)}
            onBlur={onLeave}
          >
            <div className="suggestion-header">
              <span className="suggestion-sentence">{suggestion.sentence}</span>
              <button className="apply-btn" onClick={() => onApply(index)}>Apply</button>
            </div>
            <div className="original-text">
              <strong>Original:</strong>{' '}
              <span dangerouslySetInnerHTML={{ __html: suggestion.original_highlighted || escapeHtml(suggestion.original) }} />
            </div>
            <div className="corrected-text-suggestion">
              <strong>Suggested:</strong>{' '}
              <span dangerouslySetInnerHTML={{ __html: suggestion.corrected_highlighted || escapeHtml(suggestion.corrected) }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function escapeHtml(unsafe: string): string {
  return unsafe
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
```

- [ ] **Step 3: Write ScoreBadge.tsx + ThemeToggle.tsx**

```tsx
export function ScoreBadge({ score }: { score: number }) {
  return (
    <div className="score-badge" title="Writing quality score">
      <span className="score-value">{score}</span>
      <span className="score-max">/ 100</span>
    </div>
  );
}
```

```tsx
import { useEffect, useState } from 'react';

export function ThemeToggle() {
  const [dark, setDark] = useState(() => {
    try {
      return localStorage.getItem('theme') !== 'light';
    } catch {
      return true;
    }
  });

  useEffect(() => {
    document.body.classList.toggle('dark-mode', dark);
    try {
      localStorage.setItem('theme', dark ? 'dark' : 'light');
    } catch {
      // ignore storage errors
    }
  }, [dark]);

  return (
    <button
      className="theme-toggle"
      onClick={() => setDark((d) => !d)}
      title={dark ? 'Dark Mode (Click to switch to Light Mode)' : 'Light Mode (Click to switch to Dark Mode)'}
    >
      {dark ? '🌙' : '☀️'}
    </button>
  );
}
```

- [ ] **Step 4: Write pdf.ts + ReportButton.tsx** (port of downloadReport)

```ts
import { jsPDF } from 'jspdf';
import type { Suggestion } from '../../electron/core/types';

// WCAG 2.0 AA compliant colors on white background (contrast >= 4.5:1)
const COLOR_TITLE = [41, 41, 41] as const;
const COLOR_BODY = [51, 51, 51] as const;
const COLOR_ERROR = [163, 28, 28] as const;
const COLOR_CORRECT = [21, 111, 56] as const;
const COLOR_LABEL = [80, 80, 80] as const;
const COLOR_LINE = [180, 180, 180] as const;
const COLOR_SCORE = [21, 111, 56] as const;
const COLOR_BG_ORIG = [254, 226, 226] as const;
const COLOR_BG_CORR = [220, 252, 231] as const;

interface Token { text: string; highlighted: boolean }

function parseHighlightedTokens(html: string, spanClass: string): Token[] {
  if (!html) return [{ text: '', highlighted: false }];
  const tokens: Token[] = [];
  const regex = new RegExp(`<span class="${spanClass.replace(/[.*+?^${}()|[\]\]/g, '\$&')}">([^<]*)</span>`, 'g');
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = regex.exec(html)) !== null) {
    if (match.index > lastIndex) {
      tokens.push({ text: html.substring(lastIndex, match.index), highlighted: false });
    }
    tokens.push({ text: match[1], highlighted: true });
    lastIndex = regex.lastIndex;
  }
  if (lastIndex < html.length) {
    tokens.push({ text: html.substring(lastIndex), highlighted: false });
  }
  tokens.forEach((t) => { t.text = t.text.replace(/<[^>]*>/g, ''); });
  return tokens;
}

export function generatePdfReport(suggestions: Suggestion[], score: number | null): void {
  const doc = new jsPDF({ unit: 'mm', format: 'a4' });
  const pageW = doc.internal.pageSize.getWidth();
  const marginL = 20;
  const marginR = 20;
  const usableW = pageW - marginL - marginR;
  let y = 20;

  const checkPage = (needed: number) => {
    if (y + needed > doc.internal.pageSize.getHeight() - 15) {
      doc.addPage();
      y = 20;
    }
  };

  doc.setFont('helvetica', 'bold');
  doc.setFontSize(22);
  doc.setTextColor(...COLOR_TITLE);
  doc.text('Writing Quality Report', pageW / 2, y, { align: 'center' });
  y += 10;

  doc.setDrawColor(102, 126, 234);
  doc.setLineWidth(0.8);
  doc.line(marginL, y, pageW - marginR, y);
  y += 10;

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(14);
  doc.setTextColor(...COLOR_BODY);
  doc.text('Writing Quality Score:', marginL, y);
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(20);
  doc.setTextColor(...COLOR_SCORE);
  doc.text(`${score ?? '—'} / 100`, marginL + 58, y);
  y += 12;

  doc.setDrawColor(...COLOR_LINE);
  doc.setLineWidth(0.3);
  doc.line(marginL, y, pageW - marginR, y);
  y += 8;

  doc.setFont('helvetica', 'bold');
  doc.setFontSize(16);
  doc.setTextColor(...COLOR_TITLE);
  doc.text('Suggestions', marginL, y);
  y += 8;

  if (suggestions.length === 0) {
    checkPage(10);
    doc.setFont('helvetica', 'italic');
    doc.setFontSize(12);
    doc.setTextColor(...COLOR_BODY);
    doc.text('No grammar issues found. Great writing!', marginL, y);
    y += 10;
  } else {
    suggestions.forEach((s, i) => {
      checkPage(40);
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(12);
      doc.setTextColor(...COLOR_TITLE);
      doc.text(`${i + 1}. ${s.sentence}`, marginL, y);
      y += 7;

      y = renderHighlightedRow(doc, 'Original:', parseHighlightedTokens(s.original_highlighted || s.original, 'error-word'), marginL, y, usableW, COLOR_LABEL, COLOR_BODY, COLOR_ERROR, COLOR_BG_ORIG);
      y += 2;
      y = renderHighlightedRow(doc, 'Suggested:', parseHighlightedTokens(s.corrected_highlighted || s.corrected, 'corrected-word'), marginL, y, usableW, COLOR_LABEL, COLOR_BODY, COLOR_CORRECT, COLOR_BG_CORR);
      y += 6;

      if (i < suggestions.length - 1) {
        doc.setDrawColor(...COLOR_LINE);
        doc.setLineWidth(0.15);
        doc.line(marginL + 5, y, pageW - marginR - 5, y);
        y += 6;
      }
    });
  }

  checkPage(16);
  y += 4;
  doc.setDrawColor(102, 126, 234);
  doc.setLineWidth(0.5);
  doc.line(marginL, y, pageW - marginR, y);
  y += 6;
  doc.setFont('helvetica', 'italic');
  doc.setFontSize(9);
  doc.setTextColor(...COLOR_LABEL);
  doc.text('Generated by GrammarLLM', pageW / 2, y, { align: 'center' });

  doc.save('writing-quality-report.pdf');

function renderHighlightedRow(
  doc: jsPDF, label: string, tokens: Token[], marginL: number, y: number,
  usableW: number, colorLabel: readonly number[], colorNormal: readonly number[],
  colorHighlight: readonly number[], bgColor: readonly number[],
): number {
  const fontSize = 11;
  const lineH = 5.5;
  const labelW = 24;
  const contentX = marginL + labelW;
  const contentW = usableW - labelW;

  doc.setFont('helvetica', 'bold');
  doc.setFontSize(fontSize);
  doc.setTextColor(...colorLabel);
  doc.text(label, marginL + 2, y);

  const pieces: Array<{ text: string; highlighted: boolean; width: number }> = [];
  tokens.forEach((tok) => {
    tok.text.split(/( )/).forEach((part) => {
      if (part.length > 0) {
        doc.setFont('helvetica', tok.highlighted ? 'bold' : 'normal');
        pieces.push({ text: part, highlighted: tok.highlighted, width: doc.getTextWidth(part) });
      }
    });
  });

  let curX = contentX;
  let lineY = y;

  pieces.forEach((p) => {
    if (curX + p.width > contentX + contentW && p.text.trim() !== '') {
      lineY += lineH;
      curX = contentX;
      if (lineY > doc.internal.pageSize.getHeight() - 15) {
        doc.addPage();
        lineY = 20;
      }
    }
    if (p.highlighted) {
      doc.setFillColor(...bgColor);
      doc.roundedRect(curX - 0.5, lineY - 3.5, p.width + 1, lineH, 0.8, 0.8, 'F');
      doc.setFont('helvetica', 'bold');
      doc.setTextColor(...colorHighlight);
      doc.text(p.text, curX, lineY);
      doc.setFont('helvetica', 'normal');
    } else {
      doc.setTextColor(...colorNormal);
      doc.text(p.text, curX, lineY);
    }
    curX += p.width;
  });

  return lineY + lineH;
}
```

```tsx
import { generatePdfReport } from './pdf';
import type { Suggestion } from '../../electron/core/types';

export function ReportButton({ suggestions, score }: { suggestions: Suggestion[]; score: number | null }) {
  return (
    <button className="report-btn" onClick={() => generatePdfReport(suggestions, score)}>
      Download Report
    </button>
  );
}
```

- [ ] **Step 5: Port styles and wire up App**

Copy `migration/grammar-llm/static/style.css` into `src/ui/App.css`, keeping the same class names used by the components above (`.app-shell`, `.app-header`, `.app-main`, `.editor-section`, `.suggestions-section`, `.suggestion-item`, `.original-text`, `.corrected-text-suggestion`, `.empty-state`, `.error-state`, `.apply-btn`, `.check-btn`, `.clear-btn`, `.report-btn`, `.theme-toggle`, `.score-badge`, `.toast`, `.toast-error`, `.dark-mode`). Adjust selectors that referenced body-level layout (the vanilla app used a centered column) to fit the `.app-shell` grid. Keep the dark-mode default with the same CSS variables.

In `src/ui/main.tsx`:
```tsx
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import './App.css';
import App from './App';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
```

In `src/ui/App.tsx`, replace the placeholder `GrammarApp` with:
```tsx
import GrammarApp from './components/GrammarApp';
```
and use `<GrammarApp />` in the ready branch.

Update `index.html` title to `GrammarLLM`.

- [ ] **Step 6: Verify compile**

Run: `npx tsc -b`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/ui index.html && git commit -m "feat: port grammar UI to React"
```
### Task 14: Packaging config for node-llama-cpp

**Files:**
- Modify: `electron-builder.json`

**Interfaces:** none

- [ ] **Step 1: Add asarUnpack to electron-builder.json**

```json
{
  "appId": "tr.alperendemir.grammarllm",
  "files": ["dist-electron", "dist-react"],
  "asarUnpack": [
    "**/node_modules/node-llama-cpp/**",
    "**/node_modules/@node-llama-cpp/**"
  ],
  "mac": {
    "target": "dmg",
    "icon": "./desktopIcon.png"
  },
  "linux": {
    "target": "AppImage",
    "category": "Utility",
    "icon": "./desktopIcon.png"
  },
  "win": {
    "target": ["portable", "msi"],
    "icon": "./desktopIcon.png"
  }
}
```

- [ ] **Step 2: Verify config is valid JSON**

Run: `node -e "JSON.parse(require('fs').readFileSync('electron-builder.json','utf8')); console.log('valid')"`
Expected: `valid`

- [ ] **Step 3: Commit**

```bash
git add electron-builder.json && git commit -m "chore: unpack node-llama-cpp binaries from asar"
```

### Task 15: Full verification

**Files:** none (verification only)

- [ ] **Step 1: Run the complete test suite**

Run: `npx vitest run`
Expected: ALL PASS (sentences 7, clean 11, diff 7, reconstruct 5, apply 11, correction 8, modelManager 4, ipc 4 = 57 tests)

- [ ] **Step 2: Type-check everything**

Run: `npx tsc -b`
Expected: PASS (all three projects)

- [ ] **Step 3: Lint**

Run: `npm run lint`
Expected: no errors (fix any warnings that indicate real bugs; ignore style nits)

- [ ] **Step 4: Verify no model file was downloaded**

Run: `ls models/` — expected: only `.gitkeep` (or empty). The `.gguf` download is intentionally left to the user.

- [ ] **Step 5: Final commit**

```bash
git add -A && git commit -m "chore: final verification" || echo "nothing to commit"
```

- [ ] **Step 6: Hand off to user for manual E2E (no builds run by the implementer)**

User runs:
```bash
npm run dev:electron
```
Expected: app opens → "No model detected" screen → user downloads Q4_K_M → progress bar → editor appears → paste `She dont like the apples. this is a bad sentence` → Check Grammar → suggestion with highlighted `dont`/`this` → Apply → score updates → Download Report → PDF opens.
Then optionally: `npm run dist:win` to verify the packaged app (with unpacked binaries) works.

# GrammarLLM Electron Migration — Design Spec

**Date:** 2026-08-06
**Status:** Approved by user (2026-08-06), pending spec review

## Goal

Migrate the GrammarLLM app (Python FastAPI backend + vanilla JS web UI) into the existing `grammarllm-electron` project (Electron + React + TypeScript + Vite), at full feature parity: grammar checking, suggestions with apply, writing quality score, PDF report, and first-run model download. No REST API. Chrome extension is out of scope.

## Source of truth

Reference implementation lives in `migration/grammar-llm/`:
- `main.py` — correction engine (sentence splitting, prompt/clean, diff highlighting, reconstruction, apply logic) + FastAPI routes
- `static/script.js` — UI logic (score formula, apply flows, hover highlighting, PDF generation, theme, toasts, smart-edit detection)
- `static/index.html`, `static/style.css` — UI markup/styles
- `README.md` — API shapes and example input/output fixtures (use `"She dont like the apples. this is a bad sentence"` as test fixture)

## Architecture

Three layers:

```
Renderer (React + TS)  src/ui/
   │ IPC via window.api (preload contextBridge)
Main process  src/electron/
   │ imports
Core engine  src/electron/core/ (pure TS, no electron/node-llama-cpp imports)
```

### Core engine (`src/electron/core/`)

Direct, typed port of main.py logic, bug-for-bug where sensible:
- `sentences.ts` — `splitIntoSentences(text)` with abbreviation handling, decimals, initials, whitespace span tracking. Returns sentence objects `{ text, start, end, spanEnd, gapBeforeStart }`.
- `clean.ts` — `cleanCorrectedText(corrected, original)` (template tags, instruction prefixes, repetition removal, capitalization, quote-punctuation cleanup) + `isOnlyQuoteChange(original, corrected)`.
- `diff.ts` — `highlightWordDifferences(original, corrected)` using **jsdiff** `diffArrays` on the same tokenizer; returns `{ originalHighlighted, correctedHighlighted }` with `error-word` / `corrected-word` spans (same HTML as Python).
- `reconstruct.ts` — `reconstructTextFromSentences(originalText, sentenceData, correctedSentences)` preserving original spacing.
- `apply.ts` — `applySuggestion(text, suggestionIndex, suggestions)` and `applySuggestionsBulk(text, suggestions)` with rightmost-replacement overlap resolution (port of `apply_suggestions_bulk`).
- `correction.ts` — orchestrator: split → per-sentence prompt/clean/validate → build `CorrectionResponse { suggestions[], correctedText }`. Same validation rules: skip <2 chars, skip quote-only changes, reject if corrected > 2× original length, skip if > 1.5× in suggestion list.

### Main process (`src/electron/`)

- `ipc.ts` — registers `ipcMain.handle` for all channels, validates payloads with **zod**.
- `modelManager.ts` — model detection (file exists in models dir), download via node-llama-cpp `createModelDownloader` with progress + cancel, model dir = `<project>/models` in dev, `app.getPath('userData')/models` when packaged.
- `llamaService.ts` — `getLlama()` (auto GPU backend detection: CUDA/Metal/Vulkan/CPU), load model `contextSize: 4096`, `LlamaChatSession` with system prompt: *"You are a grammar correction assistant. Correct the grammar, spelling, and punctuation of the given text. Return only the corrected text without any additional explanations or prefixes."* Sampling: `temperature 0.7, topK 40, topP 0.95, minP 0.01`, `repeatPenalty` mapped from Python's frequency/presence penalty 0.0, stop sequences `["<|endoftext|>", "<|corrected_end|>", "\n\n", "Corrected:", "Here is"]`, max tokens = sentence length + 20.
- `main.ts` — window creation with `webPreferences: { preload: dist-electron/preload.cjs, contextIsolation: true, sandbox: true, nodeIntegration: false }`; loads `dist-react/index.html`.
- `preload.cjs` — plain CommonJS (sandboxed preloads cannot be ESM; package.json has `"type": "module"`). `contextBridge.exposeInMainWorld('api', ...)` exposing only: `correct`, `applySuggestion`, `applyMany`, `modelStatus`, `downloadModel`, `cancelDownload`, `onDownloadProgress`.
- `ipc-types.ts` — shared type contract (`IpcApi`, request/response types), imported by `ipc.ts` and by `src/ui/electron-api.d.ts` (renderer type declaration for `window.api`).

### IPC contract

| Channel | Type | Payload → Result |
|---|---|---|
| `model:status` | invoke | → `{ state: 'ready'\|'missing'\|'downloading'\|'error', modelName? }` |
| `model:download` | invoke | `{ url, fileName }` → starts download |
| `model:download-progress` | event | `{ percent, transferred, total }` |
| `model:cancel-download` | invoke | — |
| `text:correct` | invoke | `{ text }` → `{ suggestions[], correctedText }` |
| `suggestion:apply` | invoke | `{ originalText, suggestionIndex, suggestions }` → `{ correctedText }` |
| `suggestion:applyMany` | invoke | `{ originalText, suggestions }` → `{ correctedText }` |

`CorrectionResponse` shape identical to Python: `suggestions: [{ original, corrected, sentence, start_index, end_index, original_highlighted, corrected_highlighted }]`.

### Renderer (`src/ui/`)

React port of the vanilla UI:
- `ModelGate` — shown when `model:status` is `missing`/`error`: two download options:
  - **Q4_K_M** — "Recommended — faster, smaller download" (default-selected)
  - **Q8_0** — "Highest quality — slower"
  - URLs: `https://huggingface.co/icecubetr/GRMR-V3-G4B-GGUF/resolve/main/GRMR-V3-G4B-Q4_K_M.gguf` and `.../GRMR-V3-G4B-Q8_0.gguf` (repo `icecubetr`, not the original `qingy2024`)
  - Progress bar + cancel.
- `Editor` — textarea, Ctrl/Cmd+Enter to check, hover on suggestion selects the sentence span in the textarea (port of `findBestOccurrence` + `setSelectionRange`), smart-edit invalidation (clear suggestions when user edits text).
- `SuggestionsList` — cards with `original_highlighted` / `corrected_highlighted` rendered as HTML (sanitized server-side by construction — engine only emits its own spans; escape plain text), Apply button, applied-state filtering, toast feedback.
- `ScoreBadge` — same formula: `round(100 × (1 − errorWords/totalWords))`, errorWords = count of `error-word` spans.
- `ReportButton` — jsPDF port of `downloadReport` (same WCAG colors, layout, `parseHighlightedTokens` + `renderHighlightedRow`).
- `ThemeToggle` — dark default, localStorage persistence.

## Dependencies

Runtime (`dependencies`): `node-llama-cpp` (must be runtime dep so electron-builder bundles it), `zod`, `diff` (jsdiff — published as `diff` on npm; the `jsdiff` package is an unrelated abandoned v1.1.1), `jspdf`.
Dev: `vitest` only (jsdiff v9 bundles its own types).

Removed/replaced: fastapi, uvicorn, pydantic, huggingface-hub, python-multipart (never used), llama-cpp-python.

## Packaging

- `electron-builder.json`: add `asarUnpack` for node-llama-cpp native binaries (`**/node_modules/node-llama-cpp/**` and `**/node_modules/@node-llama-cpp/**`); models dir excluded from package (runtime download).
- Cross-platform caveats already known: per-OS builds, GPU backends auto-detected, CUDA needs NVIDIA driver + CUDA Toolkit.

## Testing

- vitest for core engine: sentence splitting edge cases (abbreviations, decimals, initials, quotes), cleaning rules, quote-only change detection, diff highlighting, reconstruction fidelity (spacing preservation), apply + bulk-apply overlap resolution. Fixtures from README example.
- Manual E2E (user-performed): download model → check grammar on sample → apply → score → PDF → packaged app runs.
- **Constraint: no automated model download in tests, and the model file must not be downloaded by the implementer. User tests model download personally after completion.**

## Out of scope

- REST API (user decision: no)
- Chrome extension (user decision: later/separate)
- Model switching UI (v1 downloads one model; settings screen later if wanted)
- GitHub Actions CI (mentioned in node-llama-cpp docs; not requested)

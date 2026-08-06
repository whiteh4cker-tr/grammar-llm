# GrammarLLM — Electron Desktop

A local-first grammar correction and writing quality assessment desktop application built with **Electron, React, TypeScript, and Vite**. It runs a fine-tuned LLM (GRMR-V3-G4B) entirely on your machine — no internet connection, no servers, no API keys, no data leaving your computer.

This is the **`electron` branch** — a full migration of the original [GrammarLLM](https://github.com/whiteh4cker-tr/grammar-llm) (Python FastAPI + vanilla JavaScript web app) into a native desktop app. See [Differences from the main branch](#differences-from-the-main-branch) below.

---

## ✨ Features

- **Real-time grammar & spelling correction** — sentence-level error detection using the GRMR-V3-G4B LLM (quantized Q4_K_M or Q8_0)
- **AI-powered suggestions** — per-sentence suggestion cards with word-level highlighted diffs (red = error, green = correction)
- **Apply suggestions individually** — one click replaces the error span in your text; applied suggestions are tracked and filtered
- **Writing quality score (0–100)** — computed from the error-to-word ratio, updated after every check
- **PDF report generation** — WCAG 2.0 AA–compliant, downloadable report with the score, all suggestions, and highlighted original/corrected sentences (jsPDF)
- **First-run model manager** — if no model is present, choose between:
  - **Q4_K_M** — *Recommended*: faster, smaller download
  - **Q8_0** — *Highest quality*: slower, larger download (~4 GB)
  - Progress bar with cancel; resumable high-speed downloads
- **Multi-backend GPU support** — llama.cpp via [node-llama-cpp](https://github.com/withcatai/node-llama-cpp) with **automatic detection**: CUDA (NVIDIA), Metal (Apple Silicon), Vulkan (AMD/Intel/NVIDIA), CPU fallback — zero configuration
- **Fully offline & private** — model runs locally; nothing is ever sent to a server
- **Secure by design** — sandboxed renderer, `contextIsolation`, typed IPC bridge via `contextBridge` (no raw `ipcRenderer` in the UI)
- **Dark/light theme** — dark mode default, persisted across sessions
- **Keyboard shortcut** — `Ctrl+Enter` (or `Cmd+Enter`) to check grammar
- **Smart editing detection** — editing your text automatically clears stale suggestions

## 🖥️ Tech Stack

| Layer | Technology |
|---|---|
| Desktop shell | Electron 43 |
| UI | React 19 + TypeScript + Vite 8 |
| LLM inference | [node-llama-cpp](https://github.com/withcatai/node-llama-cpp) (llama.cpp) — CPU / CUDA / Metal / Vulkan |
| Model downloads | node-llama-cpp `createModelDownloader` (resumable, parallel) |
| Validation | zod (IPC payloads) |
| Diff highlighting | jsdiff (`diff`) |
| PDF reports | jsPDF |
| Tests | vitest (53 unit tests) |
| Linting | oxlint |
| Packaging | electron-builder (dmg / AppImage / portable / msi) |

## 📁 Project Structure

```
src/
├── electron/            # Main process (Node)
│   ├── main.ts          # Window creation, service wiring
│   ├── ipc.ts           # ipcMain handlers (no REST API — pure IPC)
│   ├── schemas.ts       # zod validation schemas
│   ├── preload.cjs      # contextBridge → window.api (CommonJS, sandbox-compatible)
│   ├── ipc-types.ts     # typed IPC contract shared with the renderer
│   ├── modelManager.ts  # model detection / download / cancel / progress
│   ├── llamaService.ts  # node-llama-cpp chat session (fresh context per sentence)
│   └── core/            # Pure, unit-tested port of the original engine
│       ├── sentences.ts     # sentence splitting (abbreviations, decimals, quotes)
│       ├── clean.ts         # model-output cleaning (prefixes, repetition, punctuation)
│       ├── diff.ts          # word-level diff → error-word/corrected-word spans
│       ├── reconstruct.ts   # whitespace-preserving text reconstruction
│       ├── apply.ts         # single + bulk suggestion application (overlap-safe)
│       └── correction.ts    # orchestrator: split → correct → validate → suggest
└── ui/                  # Renderer (React)
    ├── App.tsx          # model-status gate → ModelGate or GrammarApp
    └── components/      # ModelGate, GrammarApp, SuggestionsList, ScoreBadge,
                         # ReportButton, pdf.ts (report builder), ThemeToggle
```

## 🚀 Getting Started

### Requirements

- Node.js 20+
- npm

### Development

```bash
npm install
npm run dev:electron     # builds React + compiles main process, then launches the app
```

On first launch you'll see the model download screen. Pick a model (~2–4 GB, resumable) and the editor opens.

> The model is stored in `./models` (dev) or the OS user-data directory (packaged app). Never committed.

### Scripts

| Script | Purpose |
|---|---|
| `npm run dev:electron` | Build everything and launch the Electron app |
| `npm run dev:react` | Vite dev server only (http://localhost:8000) |
| `npm run test` | Run the vitest suite |
| `npm run lint` | oxlint |
| `npm run build` | Full build: `tsc -b` + `vite build` + electron compile |
| `npm run dist` | Package for the current platform (electron-builder) |
| `npm run dist:mac` | macOS arm64 dmg (run on macOS) |
| `npm run dist:win` | Windows x64 portable + msi (run on Windows) |
| `npm run dist:linux` | Linux x64 AppImage (run on Linux) |

## 🔍 How It Works

1. **Model gate** — on startup the app checks for a `.gguf` model; if none exists, the download screen is shown.
2. **Correction pipeline** (per check): split text into sentences (abbreviation/decimal-aware) → for each sentence, prompt the LLM with a fixed system prompt and a **fresh chat history** (each sentence corrected independently) → clean the model output (strip prefixes, template tags, repeated segments) → validate (reject quote-only changes, >2× length explosions, >1.5× suggestion bloat) → diff words → reconstruct the full text preserving original spacing.
3. **Suggestions** — only meaningful corrections become cards; hovering a card highlights the sentence in your editor; Apply replaces the span.
4. **Score** — `round(100 × (1 − errorWords / totalWords))`.
5. **Report** — jsPDF builds the A4 report with WCAG-compliant colors and word-wrap.

## 🧪 Testing

```bash
npm run test
```

53 unit tests cover the core engine (sentence splitting edge cases, cleaning rules, diff highlighting, reconstruction fidelity, apply/overlap logic), the model manager lifecycle (download/cancel/error states), and IPC schema validation. Tests never download a model — the LLM layer is exercised manually.

## 📦 Packaging

```bash
npm run dist:win    # or dist:mac / dist:linux (build on the target OS)
```

- Output lands in `dist/`
- node-llama-cpp native binaries are unpacked from the asar archive (`asarUnpack`) — required for the packaged app to run
- App icon: `desktopIcon.png` (512×512)
- Installs are unsigned by default (`CSC_IDENTITY_AUTO_DISCOVERY=false`); set `CSC_LINK`/`CSC_KEY_PASSWORD` when you have a signing certificate

## 🔄 Differences from the main branch

| Aspect | `main` branch (original) | `electron` branch (this) |
|---|---|---|
| **Form factor** | Web app — browser + Python server | Native desktop app (Electron) |
| **Backend** | FastAPI + uvicorn REST server | No server — main-process IPC (`ipcMain`/`contextBridge`) |
| **Language** | Python (backend) + vanilla JS (frontend) | TypeScript end-to-end (Electron, React, Node) |
| **Inference** | `llama-cpp-python` (CPU-only) | `node-llama-cpp` — **CPU, CUDA, Metal, Vulkan** auto-detected |
| **Model loading** | Auto-downloads fixed Q8_0 model from Hugging Face | Download manager with **choice of Q4_K_M or Q8_0**, progress bar, cancel, resume |
| **Frontend** | Vanilla HTML/CSS/JS (`script.js`, `jspdf.umd.min.js`) | React 19 + TypeScript components |
| **API surface** | REST endpoints (`/correct`, `/apply-suggestion`, `/health`) | Typed `window.api` over IPC (zod-validated) |
| **Data validation** | Pydantic models | zod schemas |
| **Word diff** | Python `difflib` | jsdiff (`diffArrays`) |
| **PDF report** | jsPDF via CDN script | jsPDF npm package (same WCAG layout, unit-tested builder) |
| **Privacy** | Client-server (data crosses the network) | **Fully offline** — model runs locally, no data leaves the machine |
| **Packaging** | Docker / `uvicorn` | electron-builder installers: dmg, AppImage, portable exe, msi |
| **Testing** | None | 53 vitest unit tests (engine, model manager, schemas) |
| **Chrome extension** | Included in repo | Out of scope (separate artifact) |

### What was preserved

- The correction engine behavior is a **bug-for-bug port** of `main.py` (sentence splitting heuristics, cleaning rules, validation thresholds, whitespace-preserving reconstruction, overlap-safe bulk apply) — verified with the original README's example output as a test fixture
- The UI design language: same colors, dark-mode default, WCAG-compliant highlights and report
- The model: same GRMR-V3-G4B GGUF family (repo: `icecubetr/GRMR-V3-G4B-GGUF`)

## 📄 License

Original project: [whiteh4cker-tr/grammar-llm](https://github.com/whiteh4cker-tr/grammar-llm) (see its `LICENSE`). This migration retains the original's license terms.

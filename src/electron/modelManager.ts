import path from 'path';
import type { DownloadProgress, ModelStatus } from './ipc-types.js';

export interface ModelDownloader {
  download(): Promise<unknown>;
  cancel(): Promise<void>;
  onProgress(cb: (p: { transferredBytes: number; totalBytes: number }) => void): void;
}

export interface DownloaderFactory {
  create(model: { url: string; dir: string; fileName: string }): ModelDownloader | Promise<ModelDownloader>;
}

export interface ModelManagerOptions {
  modelsDir: string;
  listModels?: () => Promise<string[]>;
  factory?: DownloaderFactory;
  loadSelection?: () => Promise<string | null>;
  saveSelection?: (name: string | null) => Promise<void>;
  deleteFile?: (fileName: string) => Promise<void>;
}

const SELECTION_FILE = 'selected.json';

export class ModelManager {
  private readonly modelsDir: string;
  private readonly listModels: () => Promise<string[]>;
  private readonly factory: DownloaderFactory;
  private readonly loadSelection: () => Promise<string | null>;
  private readonly saveSelection: (name: string | null) => Promise<void>;
  private readonly deleteFile: (fileName: string) => Promise<void>;
  private currentDownload: ModelDownloader | null = null;
  private progressListeners = new Set<(p: DownloadProgress) => void>();
  private state: ModelStatus['state'] = 'missing';
  private modelName: string | undefined;
  private cancelRequested = false;
  private selectedModel: string | null = null;
  private selectionLoaded: Promise<void> | null = null;

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
    this.loadSelection = options.loadSelection ?? (async () => {
      const fs = await import('fs/promises');
      try {
        const raw = await fs.readFile(path.join(this.modelsDir, SELECTION_FILE), 'utf8');
        const parsed = JSON.parse(raw) as { selected?: string | null };
        return parsed.selected ?? null;
      } catch {
        return null;
      }
    });
    this.saveSelection = options.saveSelection ?? (async (name) => {
      const fs = await import('fs/promises');
      await fs.writeFile(path.join(this.modelsDir, SELECTION_FILE), JSON.stringify({ selected: name }), 'utf8');
    });
    this.deleteFile = options.deleteFile ?? (async (fileName) => {
      const fs = await import('fs/promises');
      await fs.unlink(path.join(this.modelsDir, fileName));
    });
  }

  /** Lazily read the persisted selection once per process. */
  private ensureSelectionLoaded(): Promise<void> {
    if (!this.selectionLoaded) {
      this.selectionLoaded = this.loadSelection().then((name) => {
        this.selectedModel = name;
      }).catch(() => {
        this.selectedModel = null;
      });
    }
    return this.selectionLoaded;
  }

  getModelPath(): string | null {
    return this.modelName ? path.join(this.modelsDir, this.modelName) : null;
  }

  async listModelFiles(): Promise<string[]> {
    return this.listModels();
  }

  async getStatus(): Promise<ModelStatus> {
    if (this.state === 'downloading') return { state: 'downloading', modelName: this.modelName };
    if (this.state === 'error') return { state: 'error', modelName: this.modelName };
    if (this.state === 'ready' && this.modelName) return { state: 'ready', modelName: this.modelName };

    await this.ensureSelectionLoaded();
    const files = await this.listModels();
    if (files.length === 0) return { state: 'missing' };

    // Prefer the persisted selection; fall back to the first installed model.
    const chosen = this.selectedModel && files.includes(this.selectedModel)
      ? this.selectedModel
      : files[0];
    this.modelName = chosen;
    this.state = 'ready';
    return { state: 'ready', modelName: chosen };
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
    this.cancelRequested = false;

    downloader.onProgress(({ transferredBytes, totalBytes }) => {
      const percent = totalBytes > 0 ? Math.round((transferredBytes / totalBytes) * 100) : 0;
      this.emitProgress({ percent, transferred: transferredBytes, total: totalBytes });
    });

    try {
      await downloader.download();
      if (!this.cancelRequested) {
        this.state = 'ready';
        this.selectedModel = fileName;
        await this.saveSelection(fileName).catch((error) => {
          console.error('Failed to persist model selection:', error);
        });
      }
    } catch (error) {
      if (!this.cancelRequested) {
        this.state = 'error';
        throw error;
      }
    } finally {
      this.currentDownload = null;
      if (this.cancelRequested) this.state = 'missing';
    }
  }

  async cancelDownload(): Promise<void> {
    this.cancelRequested = true;
    if (this.currentDownload) {
      await this.currentDownload.cancel();
    }
  }

  /** Switch the active model to an installed one. */
  async select(fileName: string): Promise<void> {
    await this.ensureSelectionLoaded();
    const files = await this.listModels();
    if (!files.includes(fileName)) {
      throw new Error(`Model not found: ${fileName}`);
    }
    this.modelName = fileName;
    this.selectedModel = fileName;
    this.state = 'ready';
    await this.saveSelection(fileName).catch((error) => {
      console.error('Failed to persist model selection:', error);
    });
  }

  /** Delete an installed model file; clears the selection if it was selected. */
  async deleteModel(fileName: string): Promise<void> {
    await this.ensureSelectionLoaded();
    await this.deleteFile(fileName);
    if (this.modelName === fileName || this.selectedModel === fileName) {
      this.modelName = undefined;
      this.selectedModel = null;
      this.state = 'missing';
      await this.saveSelection(null).catch((error) => {
        console.error('Failed to clear model selection:', error);
      });
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

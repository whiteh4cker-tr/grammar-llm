import path from 'path';
import type { DownloadProgress, ModelStatus } from './ipc-types';

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
    if (this.modelName) return { state: 'ready', modelName: this.modelName };
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

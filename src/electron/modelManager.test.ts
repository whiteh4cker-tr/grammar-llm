import { describe, it, expect, vi } from 'vitest';
import { ModelManager, type DownloaderFactory } from './modelManager';

function fakeDownloader() {
  const callbacks = new Set<(p: { transferredBytes: number; totalBytes: number }) => void>();
  let resolveDownload!: (value: unknown) => void;
  let rejectDownload!: (error: unknown) => void;
  const download = vi.fn().mockImplementation(
    () => new Promise((resolve, reject) => {
      resolveDownload = resolve;
      rejectDownload = reject;
    }),
  );
  return {
    download,
    cancel: vi.fn().mockResolvedValue(undefined),
    onProgress: vi.fn().mockImplementation((cb: (p: { transferredBytes: number; totalBytes: number }) => void) => {
      callbacks.add(cb);
    }),
    emit(transferredBytes: number, totalBytes: number) {
      callbacks.forEach((cb) => cb({ transferredBytes, totalBytes }));
    },
    resolveDownload: () => resolveDownload(undefined),
    rejectDownload: (error: unknown) => rejectDownload(error),
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

// Let pending microtasks (download continuation) run before asserting state.
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

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
    await tick();
    expect((await manager.getStatus()).state).toBe('downloading');

    downloader.emit(100, 200);
    downloader.resolveDownload();
    await promise;
    expect(progressEvents).toEqual([{ percent: 50, transferred: 100, total: 200 }]);
    expect((await manager.getStatus()).state).toBe('ready');
  });

  it('forwards cancel to the downloader and stays missing', async () => {
    const downloader = fakeDownloader();
    const { manager } = makeManager({ downloader });
    const promise = manager.download('https://example.com/model.gguf', 'model.gguf');
    await tick();
    await manager.cancelDownload();
    expect(downloader.cancel).toHaveBeenCalled();
    downloader.resolveDownload();
    await promise;
    expect((await manager.getStatus()).state).toBe('missing');
  });

  it('reports error when download fails', async () => {
    const downloader = fakeDownloader();
    const { manager } = makeManager({ downloader });
    const promise = manager.download('https://example.com/model.gguf', 'model.gguf');
    await tick();
    downloader.rejectDownload(new Error('network down'));
    await expect(promise).rejects.toThrow('network down');
    expect((await manager.getStatus()).state).toBe('error');
  });
});

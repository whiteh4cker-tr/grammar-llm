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
    loadSettings: async () => ({}),
    saveSettings: async () => {},
  });
  return { manager, downloader, factory };
}

// Let pending microtasks (download continuation) run before asserting state.
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

  function makeSelectionManager(overrides: {
    files?: string[];
    selection?: string | null;
    contextSize?: number;
  } = {}) {
    let saved: { selected?: string | null; contextSize?: number } = {
      selected: overrides.selection ?? null,
      contextSize: overrides.contextSize,
    };
    const downloader = fakeDownloader();
    const factory: DownloaderFactory = { create: vi.fn().mockReturnValue(downloader) };
    const deleteFile = vi.fn().mockResolvedValue(undefined);
    const manager = new ModelManager({
      modelsDir: '/fake/models',
      listModels: async () => overrides.files ?? [],
      factory,
      loadSettings: async () => saved,
      saveSettings: vi.fn().mockImplementation(async (settings) => { saved = { ...saved, ...settings }; }),
      deleteFile,
    });
    return { manager, deleteFile, downloader, getSaved: () => saved };
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
    await tick();
    expect(factory.create).toHaveBeenCalledWith({
      url: 'https://example.com/model.gguf',
      dir: '/fake/models',
      fileName: 'model.gguf',
    });
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

describe('ModelManager selection & deletion', () => {

  it('loads the persisted selection when its file exists', async () => {
    const { manager } = makeSelectionManager({ files: ['A.gguf', 'B.gguf'], selection: 'B.gguf' });
    const status = await manager.getStatus();
    expect(status).toEqual({ state: 'ready', modelName: 'B.gguf' });
  });

  it('falls back to the first model when selection is stale', async () => {
    const { manager } = makeSelectionManager({ files: ['A.gguf'], selection: 'gone.gguf' });
    expect((await manager.getStatus()).modelName).toBe('A.gguf');
  });

  it('select persists and switches the active model', async () => {
    const { manager } = makeSelectionManager({ files: ['A.gguf', 'B.gguf'], selection: 'A.gguf' });
    await manager.select('B.gguf');
    const status = await manager.getStatus();
    expect(status.modelName).toBe('B.gguf');
    expect(manager.getModelPath()).toContain('B.gguf');
  });

  it('select rejects unknown models', async () => {
    const { manager } = makeSelectionManager({ files: ['A.gguf'] });
    await expect(manager.select('nope.gguf')).rejects.toThrow();
  });

  it('delete removes the file, clears selection, falls back to remaining model', async () => {
    const files = ['A.gguf', 'B.gguf'];
    let saved: { selected?: string | null } = { selected: 'A.gguf' };
    const deleteFile = vi.fn().mockImplementation(async (name: string) => {
      const i = files.indexOf(name);
      if (i >= 0) files.splice(i, 1);
    });
    const manager = new ModelManager({
      modelsDir: '/fake/models',
      listModels: async () => files,
      factory: { create: vi.fn() },
      loadSettings: async () => saved,
      saveSettings: vi.fn().mockImplementation(async (settings) => { saved = { ...saved, ...settings }; }),
      deleteFile,
    });
    await manager.deleteModel('A.gguf');
    expect(deleteFile).toHaveBeenCalledWith('A.gguf');
    expect(saved.selected).toBeNull();
    expect(await manager.getStatus()).toEqual({ state: 'ready', modelName: 'B.gguf' });
  });

  it('delete of the last model leaves the app missing', async () => {
    const files = ['A.gguf'];
    let saved: { selected?: string | null } = { selected: 'A.gguf' };
    const deleteFile = vi.fn().mockImplementation(async (name: string) => {
      const i = files.indexOf(name);
      if (i >= 0) files.splice(i, 1);
    });
    const manager = new ModelManager({
      modelsDir: '/fake/models',
      listModels: async () => files,
      factory: { create: vi.fn() },
      loadSettings: async () => saved,
      saveSettings: vi.fn().mockImplementation(async (settings) => { saved = { ...saved, ...settings }; }),
      deleteFile,
    });
    await manager.deleteModel('A.gguf');
    expect((await manager.getStatus()).state).toBe('missing');
  });

  it('download success persists the new model as selection', async () => {
    const { manager, downloader, getSaved } = makeSelectionManager({ files: [] });
    const promise = manager.download('https://example.com/new.gguf', 'new.gguf');
    await tick();
    downloader.resolveDownload();
    await promise;
    expect(getSaved().selected).toBe('new.gguf');
    expect((await manager.getStatus()).modelName).toBe('new.gguf');
  });
});

describe('ModelManager context size', () => {
  it('defaults to 8192', async () => {
    const { manager } = makeSelectionManager({ files: [] });
    expect(await manager.getContextSize()).toBe(8192);
  });

  it('loads a persisted context size', async () => {
    const { manager } = makeSelectionManager({ files: [], contextSize: 4096 });
    expect(await manager.getContextSize()).toBe(4096);
  });

  it('setContextSize persists the value', async () => {
    const { manager, getSaved } = makeSelectionManager({ files: [] });
    await manager.setContextSize(16384);
    expect(await manager.getContextSize()).toBe(16384);
    expect(getSaved().contextSize).toBe(16384);
  });

  it('setContextSize rejects invalid values', async () => {
    const { manager } = makeSelectionManager({ files: [] });
    await expect(manager.setContextSize(100)).rejects.toThrow();
    await expect(manager.setContextSize(1_000_000)).rejects.toThrow();
    await expect(manager.setContextSize(4096.5)).rejects.toThrow();
  });

  it('download success keeps the configured context size in settings', async () => {
    const { manager, downloader, getSaved } = makeSelectionManager({ files: [], contextSize: 4096 });
    const promise = manager.download('https://example.com/new.gguf', 'new.gguf');
    await tick();
    downloader.resolveDownload();
    await promise;
    expect(getSaved().contextSize).toBe(4096);
  });
});

describe('ModelManager word-level correction setting', () => {
  it('defaults to enabled when the setting is missing', async () => {
    const { manager } = makeSelectionManager({ files: [] });
    expect(await manager.getWordLevelCorrection()).toBe(true);
  });

  it('loads a persisted disabled value', async () => {
    const saved = { selected: null, wordLevelCorrection: false };
    const manager = new ModelManager({
      modelsDir: '/fake/models',
      listModels: async () => [],
      factory: { create: vi.fn() },
      loadSettings: async () => saved,
      saveSettings: vi.fn(),
    });
    expect(await manager.getWordLevelCorrection()).toBe(false);
  });

  it('setWordLevelCorrection persists the value', async () => {
    const { manager, getSaved } = makeSelectionManager({ files: [] });
    await manager.setWordLevelCorrection(false);
    expect(await manager.getWordLevelCorrection()).toBe(false);
    expect(getSaved().wordLevelCorrection).toBe(false);
  });
});

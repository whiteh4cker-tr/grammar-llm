import { app, BrowserWindow } from 'electron';
import path from 'path';
import { registerIpcHandlers } from './ipc.js';
import { ModelManager } from './modelManager.js';
import { LlamaCorrectionService } from './llamaService.js';
import type { DownloadProgress } from './ipc-types.js';

function getModelsDir(): string {
  // Portable builds: electron-builder sets PORTABLE_EXECUTABLE_DIR to the
  // folder containing the exe — store models right next to it.
  const portableDir = process.env.PORTABLE_EXECUTABLE_DIR;
  if (app.isPackaged && portableDir) {
    return path.join(portableDir, 'models');
  }
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
      preload: path.join(import.meta.dirname, 'preload.cjs'),
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

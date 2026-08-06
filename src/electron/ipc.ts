import { ipcMain } from 'electron';
import { correctText } from './core/correction.js';
import { applySuggestion, applySuggestionsBulk } from './core/apply.js';
import { correctRequestSchema, applyRequestSchema, applyManySchema, downloadRequestSchema, modelFileSchema } from './schemas.js';
import type { ModelManager } from './modelManager.js';
import type { LlamaCorrectionService } from './llamaService.js';

export function registerIpcHandlers(modelManager: ModelManager, corrector: LlamaCorrectionService): void {
  ipcMain.handle('model:status', () => modelManager.getStatus());

  ipcMain.handle('model:list', () => modelManager.listModelFiles());

  ipcMain.handle('model:select', async (_event, raw) => {
    const { fileName } = modelFileSchema.parse(raw);
    await modelManager.select(fileName);
    await corrector.reset();
    // Reload the newly selected model in the background; corrections await it lazily.
    void corrector.ensureLoaded().catch((error) => {
      console.error('Failed to load selected model:', error);
    });
    return modelManager.getStatus();
  });

  ipcMain.handle('model:delete', async (_event, raw) => {
    const { fileName } = modelFileSchema.parse(raw);
    const status = await modelManager.getStatus();
    if (status.modelName === fileName) {
      await corrector.reset();
    }
    await modelManager.deleteModel(fileName);
    return modelManager.getStatus();
  });
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

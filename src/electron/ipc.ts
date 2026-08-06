import { ipcMain } from 'electron';
import { correctText } from './core/correction';
import { applySuggestion, applySuggestionsBulk } from './core/apply';
import { correctRequestSchema, applyRequestSchema, applyManySchema, downloadRequestSchema } from './schemas';
import type { ModelManager } from './modelManager';
import type { SentenceCorrector } from './core/types';

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

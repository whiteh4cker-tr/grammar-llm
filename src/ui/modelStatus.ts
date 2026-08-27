import type { ModelStatus } from '../electron/ipc-types';

/**
 * Decide what the UI should treat as the model situation.
 *
 * While a *second* model downloads, `modelManager` reports
 * `{ state: 'downloading', modelName: <file being downloaded> }` even though
 * the already-loaded model is still usable. Taken literally that means "no
 * usable model", which used to unmount the editor and the open Settings dialog
 * (and with them, the download progress). A download is therefore only
 * blocking while nothing has been loaded yet.
 */
export function effectiveModelStatus(status: ModelStatus, readyModel: string | null): ModelStatus {
  return status.state === 'downloading' && readyModel ? { state: 'ready', modelName: readyModel } : status;
}

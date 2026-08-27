import { createContext, useContext } from 'react';

export interface ActiveDownload {
  /** File being downloaded, when known (progress events can arrive first). */
  fileName: string | null;
  percent: number;
  transferred: number;
  total: number;
}

export interface DownloadTracker {
  /** Non-null while a model download is in flight, app-wide. */
  active: ActiveDownload | null;
  /** Runs `run` with the download tracked; state lives above the caller, so it
   *  survives the caller unmounting (e.g. the Settings dialog closing). */
  track: (fileName: string, run: () => Promise<void>) => Promise<void>;
  cancel: () => Promise<void>;
}

/**
 * Kept separate from `<DownloadProvider />` so each file exports one kind of
 * thing (React Fast Refresh requires component-only modules).
 */
export const DownloadContext = createContext<DownloadTracker | null>(null);

export function useDownloadTracker(): DownloadTracker {
  const value = useContext(DownloadContext);
  if (!value) throw new Error('useDownloadTracker must be used inside <DownloadProvider>');
  return value;
}

/** "Downloading X — 42% (120 MB of 4.3 GB)" for the banner and the gate. */
export function formatDownload(active: ActiveDownload): string {
  const name = active.fileName ?? 'model';
  const percent = active.total > 0 ? ` — ${active.percent}%` : '';
  const size =
    active.total > 0
      ? ` (${formatMb(active.transferred)} MB of ${formatMb(active.total)} MB)`
      : '';
  return `Downloading ${name}${percent}${size}`;
}

function formatMb(bytes: number): string {
  return (bytes / 1024 / 1024).toFixed(0);
}

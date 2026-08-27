import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react';
import { api } from './api';
import { DownloadContext, type ActiveDownload, type DownloadTracker } from './downloadState';

/**
 * Owns model-download state for the whole app.
 *
 * It deliberately lives above the components that start downloads: the model
 * list / Settings dialog unmount while a download is running (the app switches
 * screens as the model status changes), and a download indicator stored in
 * those components' local state disappears with them — which looked like
 * "nothing is happening" while the file was still downloading.
 */
export function DownloadProvider({ children }: { children: ReactNode }) {
  const [active, setActive] = useState<ActiveDownload | null>(null);

  // Subscribed once, for the lifetime of the app, so progress keeps arriving
  // regardless of who started the download.
  useEffect(() => {
    const bridge = api as Partial<typeof api> | undefined;
    if (typeof bridge?.onDownloadProgress !== 'function') return;
    return bridge.onDownloadProgress((p) => {
      setActive((prev) => (prev ? { ...prev, percent: p.percent, transferred: p.transferred, total: p.total } : { fileName: null, ...p }));
    });
  }, []);

  const track = useCallback(async (fileName: string, run: () => Promise<void>) => {
    setActive({ fileName, percent: 0, transferred: 0, total: 0 });
    try {
      await run();
    } finally {
      setActive(null);
    }
  }, []);

  const cancel = useCallback(async () => {
    try {
      await api.cancelDownload();
    } finally {
      // The pending `track()` call clears this too; clearing here makes the
      // button feel instant.
      setActive(null);
    }
  }, []);

  const value = useMemo<DownloadTracker>(() => ({ active, track, cancel }), [active, track, cancel]);

  return <DownloadContext.Provider value={value}>{children}</DownloadContext.Provider>;
}

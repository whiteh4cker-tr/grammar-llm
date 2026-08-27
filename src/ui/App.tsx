import { useEffect, useState } from 'react';
import Box from '@mui/material/Box';
import CircularProgress from '@mui/material/CircularProgress';
import Typography from '@mui/material/Typography';
import { api } from './api';
import { ModelGate } from './components/ModelGate';
import { effectiveModelStatus } from './modelStatus';
import GrammarApp from './components/GrammarApp';
import { Settings } from './components/Settings';
import type { ModelStatus } from '../electron/ipc-types';

export default function App() {
  const [status, setStatus] = useState<ModelStatus | null>(null);
  /**
   * Last model seen in state `ready`. While a *second* model downloads, the
   * manager reports `state: 'downloading'` and points `modelName` at the new
   * file, even though the loaded model is still usable — treating that as
   * "no model" would tear down the editor and any open dialog mid-download.
   */
  const [readyModel, setReadyModel] = useState<string | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [wordLevelEnabled, setWordLevelEnabled] = useState(true);

  useEffect(() => {
    let cancelled = false;
    async function poll() {
      const next = await api.modelStatus();
      if (cancelled) return;
      setStatus(next);
      if (next.state === 'ready' && next.modelName) setReadyModel(next.modelName);
      else if (next.state === 'missing') setReadyModel(null);
    }
    void poll();
    const timer = setInterval(() => void poll(), 2000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    api.getWordLevelCorrection()
      .then((r) => { if (!cancelled) setWordLevelEnabled(r.enabled); })
      .catch(() => { /* keep default (enabled) */ });
    return () => { cancelled = true; };
  }, []);

  async function handleWordLevelChange(enabled: boolean) {
    setWordLevelEnabled(enabled);
    try {
      await api.setWordLevelCorrection({ enabled });
    } catch {
      setWordLevelEnabled((prev) => !prev); // revert on failure
    }
  }

  if (!status) {
    return (
      <Box
        sx={{
          minHeight: '100vh',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 2,
          color: '#fff',
        }}
      >
        <CircularProgress sx={{ color: '#fff' }} />
        <Typography variant="body1">Loading…</Typography>
      </Box>
    );
  }

  // A download in flight is only blocking when nothing is loaded yet.
  const effectiveStatus = effectiveModelStatus(status, readyModel);
  const needsModel = effectiveStatus.state !== 'ready';

  if (needsModel) {
    return <ModelGate status={effectiveStatus} mode="required" />;
  }

  return (
    <>
      <GrammarApp onOpenSettings={() => setSettingsOpen(true)} wordLevelEnabled={wordLevelEnabled} />
      {settingsOpen && (
        <Settings
          status={effectiveStatus}
          wordLevelEnabled={wordLevelEnabled}
          onWordLevelChange={(enabled) => void handleWordLevelChange(enabled)}
          onClose={() => setSettingsOpen(false)}
        />
      )}
    </>
  );
}

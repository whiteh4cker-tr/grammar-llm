import { useEffect, useState } from 'react';
import { api } from './api';
import { ModelGate } from './components/ModelGate';
import GrammarApp from './components/GrammarApp';
import { Settings } from './components/Settings';
import type { ModelStatus } from '../electron/ipc-types';

export default function App() {
  const [status, setStatus] = useState<ModelStatus | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [wordLevelEnabled, setWordLevelEnabled] = useState(true);

  useEffect(() => {
    let cancelled = false;
    async function poll() {
      const next = await api.modelStatus();
      if (!cancelled) setStatus(next);
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

  if (!status) return <div className="app-loading">Loading…</div>;

  const needsModel = status.state === 'missing' || status.state === 'downloading' || status.state === 'error';

  if (needsModel) {
    return <ModelGate status={status} mode="required" />;
  }

  return (
    <>
      <GrammarApp onOpenSettings={() => setSettingsOpen(true)} wordLevelEnabled={wordLevelEnabled} />
      {settingsOpen && (
        <Settings
          status={status}
          wordLevelEnabled={wordLevelEnabled}
          onWordLevelChange={(enabled) => void handleWordLevelChange(enabled)}
          onClose={() => setSettingsOpen(false)}
        />
      )}
    </>
  );
}

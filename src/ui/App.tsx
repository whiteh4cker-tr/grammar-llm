import { useEffect, useState } from 'react';
import { api } from './api';
import { ModelGate } from './components/ModelGate';
import GrammarApp from './components/GrammarApp';
import type { ModelStatus } from '../electron/ipc-types';

export default function App() {
  const [status, setStatus] = useState<ModelStatus | null>(null);

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

  if (!status) return <div className="app-loading">Loading…</div>;
  if (status.state === 'missing' || status.state === 'downloading' || status.state === 'error') {
    return <ModelGate status={status} />;
  }
  return <GrammarApp />;
}

import { useEffect, useState } from 'react';
import { api } from '../api';
import type { DownloadProgress, ModelStatus } from '../../electron/ipc-types';
import './ModelGate.css';

const MODELS = [
  {
    fileName: 'GRMR-V3-G4B-Q4_K_M.gguf',
    url: 'https://huggingface.co/icecubetr/GRMR-V3-G4B-GGUF/resolve/main/GRMR-V3-G4B-Q4_K_M.gguf',
    label: 'Q4_K_M — Recommended',
    detail: 'Faster, smaller download',
  },
  {
    fileName: 'GRMR-V3-G4B-Q8_0.gguf',
    url: 'https://huggingface.co/icecubetr/GRMR-V3-G4B-GGUF/resolve/main/GRMR-V3-G4B-Q8_0.gguf',
    label: 'Q8_0 — Highest quality',
    detail: 'Slower, larger download',
  },
];

export function ModelGate({ status }: { status: ModelStatus }) {
  const [selected, setSelected] = useState(0);
  const [progress, setProgress] = useState<DownloadProgress | null>(null);
  const [downloading, setDownloading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!downloading) return;
    const unsubscribe = api.onDownloadProgress(setProgress);
    return unsubscribe;
  }, [downloading]);

  async function handleDownload() {
    const model = MODELS[selected];
    setDownloading(true);
    setError(null);
    try {
      await api.downloadModel({ url: model.url, fileName: model.fileName });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setDownloading(false);
      setProgress(null);
    }
  }

  async function handleCancel() {
    await api.cancelDownload();
    setDownloading(false);
    setProgress(null);
  }

  return (
    <div className="model-gate">
      <h1>GrammarLLM</h1>
      <p className="model-gate-subtitle">
        No model detected. Choose a model to download and get started.
      </p>

      <div className="model-options">
        {MODELS.map((model, index) => (
          <label key={model.fileName} className={`model-option ${selected === index ? 'selected' : ''}`}>
            <input
              type="radio"
              name="model"
              checked={selected === index}
              onChange={() => setSelected(index)}
              disabled={downloading}
            />
            <div>
              <strong>{model.label}</strong>
              <span>{model.detail}</span>
            </div>
          </label>
        ))}
      </div>

      {status.state === 'error' && <p className="model-gate-error">Model failed to load: {status.modelName}</p>}
      {error && <p className="model-gate-error">{error}</p>}

      {progress ? (
        <div className="download-area">
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: `${progress.percent}%` }} />
          </div>
          <p>
            {progress.percent}% — {(progress.transferred / 1024 / 1024).toFixed(0)} MB /{' '}
            {(progress.total / 1024 / 1024).toFixed(0)} MB
          </p>
          <button onClick={handleCancel}>Cancel</button>
        </div>
      ) : (
        <button className="download-btn" onClick={() => void handleDownload()} disabled={downloading}>
          {downloading ? 'Downloading…' : 'Download model'}
        </button>
      )}
    </div>
  );
}

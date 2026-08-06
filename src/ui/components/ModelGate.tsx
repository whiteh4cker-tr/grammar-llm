import { useEffect, useState } from 'react';
import { api } from '../api';
import type { DownloadProgress, ModelStatus } from '../../electron/ipc-types';
import './ModelGate.css';

const MODELS = [
  {
    fileName: 'GRMR-V3-G4B-Q4_K_M.gguf',
    url: 'https://huggingface.co/icecubetr/GRMR-V3-G4B-GGUF/resolve/main/GRMR-V3-G4B-Q4_K_M.gguf',
    label: 'GRMR-V3-G4B-Q4_K_M',
    detail: 'Recommended — faster, smaller download',
  },
  {
    fileName: 'GRMR-V3-G4B-Q8_0.gguf',
    url: 'https://huggingface.co/icecubetr/GRMR-V3-G4B-GGUF/resolve/main/GRMR-V3-G4B-Q8_0.gguf',
    label: 'GRMR-V3-G4B-Q8_0',
    detail: 'Highest quality — slower, larger download',
  },
];

function fileNameFromUrl(url: string): string | null {
  try {
    const parsed = new URL(url);
    const segment = parsed.pathname.split('/').filter(Boolean).pop();
    if (!segment) return null;
    return decodeURIComponent(segment);
  } catch {
    return null;
  }
}

interface Props {
  status: ModelStatus;
  mode?: 'required' | 'manage';
  onClose?: () => void;
}

export function ModelGate({ status, mode = 'required', onClose }: Props) {
  const [selected, setSelected] = useState(0);
  const [customUrl, setCustomUrl] = useState('');
  const [installed, setInstalled] = useState<string[]>([]);
  const [progress, setProgress] = useState<DownloadProgress | null>(null);
  const [downloading, setDownloading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [contextSize, setContextSize] = useState<number | null>(null);
  const [contextDraft, setContextDraft] = useState('');
  const [contextSaved, setContextSaved] = useState(false);

  useEffect(() => {
    let cancelled = false;
    api.listModels().then((models) => {
      if (!cancelled) setInstalled(models);
    }).catch(() => { /* ignore */ });
    api.getSettings().then((settings) => {
      if (cancelled) return;
      setContextSize(settings.contextSize);
      setContextDraft(String(settings.contextSize));
    }).catch(() => { /* ignore */ });
    return () => { cancelled = true; };
  }, [mode]);

  useEffect(() => {
    if (!downloading) return;
    const unsubscribe = api.onDownloadProgress(setProgress);
    return unsubscribe;
  }, [downloading]);

  const isCustom = selected === MODELS.length;
  const customFileName = isCustom ? fileNameFromUrl(customUrl) : null;
  const customValid = isCustom && customUrl.trim().length > 0 && customFileName !== null && customFileName.endsWith('.gguf');

  const chosen = isCustom
    ? { fileName: customFileName, url: customUrl.trim() }
    : MODELS[selected];

  const buttonLabel = isCustom
    ? (customValid ? `Download ${customFileName}` : 'Enter a valid .gguf URL')
    : (installed.includes(MODELS[selected].fileName)
        ? `Use ${MODELS[selected].label}`
        : `Download ${MODELS[selected].label}`);

  async function handlePrimaryAction() {
    if (!chosen.fileName) return;
    setError(null);

    // Already installed → just switch to it, no re-download.
    if (installed.includes(chosen.fileName)) {
      try {
        await api.selectModel({ fileName: chosen.fileName });
        setProgress(null);
        setDownloading(false);
        if (mode === 'manage' && onClose) onClose();
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
      return;
    }

    setDownloading(true);
    try {
      await api.downloadModel({ url: chosen.url, fileName: chosen.fileName });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setDownloading(false);
      setProgress(null);
    }
  }

  async function handleSelect(fileName: string) {
    setError(null);
    try {
      await api.selectModel({ fileName });
      if (onClose) onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function handleDelete(fileName: string) {
    setError(null);
    try {
      const next = await api.deleteModel({ fileName });
      setInstalled((prev) => prev.filter((name) => name !== fileName));
      if (next.state === 'missing' && mode === 'manage' && onClose) {
        onClose();
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function handleCancel() {
    await api.cancelDownload();
    setDownloading(false);
    setProgress(null);
  }

  async function handleSaveContextSize() {
    const parsed = Number(contextDraft);
    if (!Number.isInteger(parsed) || parsed < 256 || parsed > 131072) {
      setError('Context size must be an integer between 256 and 131072.');
      return;
    }
    setError(null);
    setContextSaved(false);
    try {
      const result = await api.setContextSize({ contextSize: parsed });
      setContextSize(result.contextSize);
      setContextDraft(String(result.contextSize));
      setContextSaved(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <div className="model-gate">
      <div className="model-gate-head">
        <h1>GrammarLLM</h1>
        {mode === 'manage' && onClose && (
          <button className="gate-close" onClick={onClose}>✕</button>
        )}
      </div>
      <p className="model-gate-subtitle">
        {mode === 'manage'
          ? 'Manage your models — switch, download, or delete.'
          : 'No model detected. Choose a model to download and get started.'}
      </p>

      {installed.length > 0 && (
        <div className="installed-models">
          <h2>Installed models</h2>
          {installed.map((name) => {
            const isActive = status.modelName === name && status.state === 'ready';
            return (
              <div key={name} className={`installed-model ${isActive ? 'active' : ''}`}>
                <span className="installed-name" title={name}>{name}</span>
                {isActive && <span className="installed-badge">in use</span>}
                <div className="installed-actions">
                  {!isActive && (
                    <button className="use-btn" onClick={() => void handleSelect(name)}>Use</button>
                  )}
                  <button className="delete-btn" onClick={() => void handleDelete(name)}>Delete</button>
                </div>
              </div>
            );
          })}
        </div>
      )}

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
              {installed.includes(model.fileName) && <span className="option-installed">already installed</span>}
            </div>
          </label>
        ))}

        <label className={`model-option ${isCustom ? 'selected' : ''}`}>
          <input
            type="radio"
            name="model"
            checked={isCustom}
            onChange={() => setSelected(MODELS.length)}
            disabled={downloading}
          />
          <div className="custom-option">
            <strong>Custom GGUF URL</strong>
            <span>Paste a direct download link to any .gguf model (e.g., from Hugging Face)</span>
            {isCustom && (
              <input
                className="custom-url-input"
                type="url"
                placeholder="https://huggingface.co/.../model.gguf"
                value={customUrl}
                onChange={(e) => setCustomUrl(e.target.value)}
                disabled={downloading}
              />
            )}
          </div>
        </label>
      </div>

      {status.state === 'error' && <p className="model-gate-error">Model failed to load: {status.modelName}</p>}
      {error && <p className="model-gate-error">{error}</p>}

      {mode === 'manage' && contextSize !== null && (
        <div className="context-size-row">
          <div className="context-size-label">
            <strong>Context size (tokens)</strong>
            <span>Applies after the model reloads. Larger context = more memory.</span>
          </div>
          <div className="context-size-controls">
            <input
              className="context-size-input"
              type="number"
              min={256}
              max={131072}
              step={256}
              value={contextDraft}
              onChange={(e) => {
                setContextDraft(e.target.value);
                setContextSaved(false);
              }}
              disabled={downloading}
            />
            <button className="context-save-btn" onClick={() => void handleSaveContextSize()} disabled={downloading}>
              Apply
            </button>
            {contextSaved && <span className="context-saved-note">Saved — reloading…</span>}
          </div>
        </div>
      )}

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
        <button
          className="download-btn"
          onClick={() => void handlePrimaryAction()}
          disabled={downloading || (isCustom && !customValid)}
        >
          {buttonLabel}
        </button>
      )}
    </div>
  );
}

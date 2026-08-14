import { useEffect, useState } from 'react';
import { api } from '../api';
import { ModelGate } from './ModelGate';
import { applyTheme, getStoredTheme, type Theme } from '../theme';
import type { ModelStatus } from '../../electron/ipc-types';

interface Props {
  status: ModelStatus;
  wordLevelEnabled: boolean;
  onWordLevelChange: (enabled: boolean) => void;
  onClose: () => void;
}

export function Settings({ status, wordLevelEnabled, onWordLevelChange, onClose }: Props) {
  const [theme, setTheme] = useState<Theme>(() => getStoredTheme());
  const [contextDraft, setContextDraft] = useState('');
  const [contextSaved, setContextSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getSettings().then((settings) => {
      setContextDraft(String(settings.contextSize));
    }).catch(() => { /* ignore */ });
  }, []);

  function handleThemeChange(next: Theme) {
    setTheme(next);
    applyTheme(next);
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
      setContextDraft(String(result.contextSize));
      setContextSaved(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <div className="settings-overlay" onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="settings-modal">
        <div className="settings-head">
          <h2>Settings</h2>
          <button className="settings-close" onClick={onClose} title="Close settings">✕</button>
        </div>

        {error && <p className="settings-error">{error}</p>}

        <section className="settings-section">
          <h3>General</h3>
          <div className="settings-row">
            <div className="settings-label">
              <strong>Theme</strong>
              <span>Applies to the whole app</span>
            </div>
            <div className="theme-segmented">
              <button className={theme === 'light' ? 'active' : ''} onClick={() => handleThemeChange('light')}>☀️ Light</button>
              <button className={theme === 'dark' ? 'active' : ''} onClick={() => handleThemeChange('dark')}>🌙 Dark</button>
            </div>
          </div>
          <div className="settings-row">
            <div className="settings-label">
              <strong>Word-level corrections</strong>
              <span>Highlight misspelled words in the editor and offer one-click fixes</span>
            </div>
            <button
              className={`toggle ${wordLevelEnabled ? 'on' : ''}`}
              onClick={() => onWordLevelChange(!wordLevelEnabled)}
              role="switch"
              aria-checked={wordLevelEnabled}
              title={wordLevelEnabled ? 'Enabled' : 'Disabled'}
            >
              <span className="toggle-knob" />
            </button>
          </div>
        </section>

        <section className="settings-section">
          <h3>LLM</h3>
          <div className="settings-row">
            <div className="settings-label">
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
              />
              <button className="context-save-btn" onClick={() => void handleSaveContextSize()}>Apply</button>
              {contextSaved && <span className="context-saved-note">Saved — reloading…</span>}
            </div>
          </div>
          <ModelGate status={status} mode="manage" onClose={onClose} embedded />
        </section>
      </div>
    </div>
  );
}

import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../api';
import type { CorrectionResponse, Suggestion } from '../../electron/core/types';
import { SuggestionsList } from './SuggestionsList';
import { ScoreBadge } from './ScoreBadge';
import { ReportButton } from './ReportButton';
import { ThemeToggle } from './ThemeToggle';

export default function GrammarApp() {
  const [text, setText] = useState('');
  const [corrections, setCorrections] = useState<CorrectionResponse | null>(null);
  const [applied, setApplied] = useState<Set<number>>(new Set());
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<{ message: string; isError?: boolean } | null>(null);
  const [score, setScore] = useState<number | null>(null);
  const [lastCheckedText, setLastCheckedText] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const lastCaretRef = useRef(0);

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        event.preventDefault();
        void handleCheck();
      }
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  });

  async function handleCheck() {
    const input = text.trim();
    if (!input) {
      setToast({ message: 'Please enter some text to check grammar.', isError: true });
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const result = await api.correct(input);
      setCorrections(result);
      setApplied(new Set());
      setLastCheckedText(input);
      setScore(computeScore(input, result.suggestions));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setScore(null);
    } finally {
      setLoading(false);
    }
  }

  function handleTextChange(next: string) {
    setText(next);
    // Smart-edit invalidation (port of checkForTextChanges): a small length
    // change is likely an applied suggestion; anything else clears suggestions.
    const isSuggestionApply =
      lastCheckedText !== '' &&
      next !== lastCheckedText &&
      Math.abs(next.length - lastCheckedText.length) < 20;
    if (!isSuggestionApply && corrections && next !== lastCheckedText) {
      setCorrections(null);
      setApplied(new Set());
      setScore(null);
    }
  }

  async function handleApply(index: number) {
    if (!corrections?.suggestions[index]) return;
    try {
      const result = await api.applySuggestion({
        originalText: text,
        suggestionIndex: index,
        suggestions: corrections.suggestions,
      });
      setText(result.correctedText);
      setApplied((prev) => new Set(prev).add(index));
      setToast({ message: `Applied correction for ${corrections.suggestions[index].sentence}` });
    } catch (e) {
      setToast({ message: e instanceof Error ? e.message : 'Failed to apply', isError: true });
    }
  }

  const handleSuggestionHover = useCallback((suggestion: Suggestion) => {
    const el = textareaRef.current;
    if (!el) return;
    const bestSpan = findBestOccurrence(el.value, suggestion.original, suggestion.start_index);
    if (bestSpan) {
      el.focus();
      el.setSelectionRange(bestSpan[0], bestSpan[1]);
    }
  }, []);

  const handleSuggestionLeave = useCallback(() => {
    const el = textareaRef.current;
    if (el) {
      const caret = Math.min(lastCaretRef.current, el.value.length);
      el.setSelectionRange(caret, caret);
    }
  }, []);

  return (
    <div className="app-shell">
      <ThemeToggle />

      <header className="app-header">
        <h1>GrammarLLM</h1>
        <p>Automated grammar correction and writing quality assessment</p>
      </header>

      <main className="app-main">
        <section className="editor-section">
          <div className="editor-header">
            <h2>Your Text</h2>
            <div className="editor-actions">
              <button className="check-btn" onClick={() => void handleCheck()} disabled={loading}>
                {loading ? 'Checking…' : 'Check Grammar'}
              </button>
              <button
                className="clear-btn"
                onClick={() => {
                  setText('');
                  setCorrections(null);
                  setApplied(new Set());
                  setScore(null);
                  setError(null);
                }}
              >
                Clear
              </button>
            </div>
          </div>
          <textarea
            ref={textareaRef}
            value={text}
            placeholder="Type or paste your text here, then press Ctrl+Enter or click Check Grammar"
            onChange={(e) => handleTextChange(e.target.value)}
            onKeyUp={(e) => { lastCaretRef.current = e.currentTarget.selectionStart; }}
            onClick={(e) => { lastCaretRef.current = e.currentTarget.selectionStart; }}
          />
        </section>

        <section className="suggestions-section">
          <div className="suggestions-header">
            <h2>Suggestions</h2>
            <div className="writing-quality-right">
              {score !== null && <ScoreBadge score={score} />}
              <ReportButton suggestions={corrections?.suggestions ?? []} score={score} />
            </div>
          </div>
          <SuggestionsList
            suggestions={corrections?.suggestions ?? []}
            applied={applied}
            loading={loading}
            error={error}
            onApply={handleApply}
            onHover={handleSuggestionHover}
            onLeave={handleSuggestionLeave}
          />
        </section>
      </main>

      {toast && <Toast toast={toast} onDone={() => setToast(null)} />}
    </div>
  );
}

export function computeScore(text: string, suggestions: Suggestion[]): number {
  const words = text.trim().split(/\s+/).filter((w) => w.length > 0);
  const totalWords = words.length;
  let errorCount = 0;
  for (const s of suggestions) {
    const matches = s.original_highlighted.match(/<span class="error-word">/g);
    if (matches) errorCount += matches.length;
  }
  if (totalWords === 0) return 100;
  return Math.max(0, Math.round(100 * (1 - errorCount / totalWords)));
}

export function findBestOccurrence(
  haystack: string,
  needle: string,
  approxIndex: number,
): [number, number] | null {
  if (!needle || !haystack) return null;
  const escaped = needle.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const re = new RegExp(escaped, 'g');
  const matches: Array<[number, number]> = [];
  let m: RegExpExecArray | null;
  while ((m = re.exec(haystack)) !== null) {
    matches.push([m.index, m.index + needle.length]);
    if (m.index === re.lastIndex) re.lastIndex++;
  }
  if (matches.length === 0) return null;
  return matches.reduce((best, occ) =>
    Math.abs(occ[0] - approxIndex) < Math.abs(best[0] - approxIndex) ? occ : best,
  );
}

function Toast({
  toast,
  onDone,
}: {
  toast: { message: string; isError?: boolean };
  onDone: () => void;
}) {
  // Key the timer on the toast object identity, NOT onDone: onDone is a fresh
  // closure every GrammarApp render, which would reset the timer forever.
  useEffect(() => {
    const timer = setTimeout(onDone, 3000);
    return () => clearTimeout(timer);
  }, [toast]); // eslint-disable-line react-hooks/exhaustive-deps -- onDone identity is unstable by design
  return <div className={`toast ${toast.isError ? 'toast-error' : ''}`}>{toast.message}</div>;
}

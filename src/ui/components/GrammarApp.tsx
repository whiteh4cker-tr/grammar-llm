import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { api } from '../api';
import type { CorrectionResponse, Suggestion, WordFix } from '../../electron/core/types';
import { SuggestionsList } from './SuggestionsList';
import { ScoreBadge } from './ScoreBadge';
import { ReportButton } from './ReportButton';
import { buildSegments, normalizeFixes } from '../wordOverlay';
import { applyWordFixToText, mergeSentenceRecheck, rebaseRecheckedSuggestions, shiftApplied } from '../recheck';

const RECHECK_DELAY_MS = 600;

export default function GrammarApp({
  onOpenSettings,
  wordLevelEnabled,
}: {
  onOpenSettings: () => void;
  wordLevelEnabled: boolean;
}) {
  const [text, setText] = useState('');
  const [corrections, setCorrections] = useState<CorrectionResponse | null>(null);
  const [applied, setApplied] = useState<Set<number>>(new Set());
  const [loading, setLoading] = useState(false);
  const [rechecking, setRechecking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<{ message: string; isError?: boolean } | null>(null);
  const [score, setScore] = useState<number | null>(null);
  const [lastCheckedText, setLastCheckedText] = useState('');
  const [hoverFix, setHoverFix] = useState<WordFix | null>(null);
  const [popupPos, setPopupPos] = useState<{ left: number; top: number } | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const mirrorRef = useRef<HTMLDivElement>(null);
  const editorWrapRef = useRef<HTMLDivElement>(null);
  const popupRef = useRef<HTMLDivElement>(null);
  const spanRectRef = useRef<DOMRect | null>(null);
  const lastCaretRef = useRef(0);
  const recheckTimerRef = useRef<number | null>(null);
  const requestIdRef = useRef(0);

  const wordFixes = useMemo(
    () => normalizeFixes(text, corrections?.suggestions.flatMap((s) => s.wordFixes) ?? []),
    [text, corrections],
  );
  const overlayActive = wordLevelEnabled && corrections !== null && text === lastCheckedText;

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

  // Sync mirror scroll position whenever a check lands.
  useEffect(() => {
    const el = textareaRef.current;
    const mirror = mirrorRef.current;
    if (el && mirror) {
      mirror.scrollTop = el.scrollTop;
      mirror.scrollLeft = el.scrollLeft;
    }
  }, [corrections]);

  // The textarea's vertical scrollbar (classic scrollbars) consumes layout
  // width, so the mirror must end where the textarea's content ends —
  // otherwise wrapped lines break at different words and highlights drift
  // from the real text (and selection looks misaligned).
  useLayoutEffect(() => {
    const el = textareaRef.current;
    const mirror = mirrorRef.current;
    if (!el || !mirror) return;
    const cs = getComputedStyle(el);
    const borders = parseFloat(cs.borderLeftWidth) + parseFloat(cs.borderRightWidth);
    mirror.style.right = `${Math.max(0, el.offsetWidth - el.clientWidth - borders)}px`;
  }, [text, corrections, overlayActive]);

  // Re-measure when the editor wrapper resizes (scrollbar may appear/leave).
  useEffect(() => {
    const wrap = editorWrapRef.current;
    const el = textareaRef.current;
    const mirror = mirrorRef.current;
    if (!wrap || !el || !mirror) return;
    const ro = new ResizeObserver(() => {
      const cs = getComputedStyle(el);
      const borders = parseFloat(cs.borderLeftWidth) + parseFloat(cs.borderRightWidth);
      mirror.style.right = `${Math.max(0, el.offsetWidth - el.clientWidth - borders)}px`;
    });
    ro.observe(wrap);
    return () => ro.disconnect();
  }, []);

  // Position the popup under the hovered fix's mirror span. The first pass
  // places it provisionally at the word; a second pass (below) clamps it
  // horizontally using the popup's measured width.
  useEffect(() => {
    if (!hoverFix) {
      setPopupPos(null);
      spanRectRef.current = null;
      return;
    }
    const mirror = mirrorRef.current;
    const container = editorWrapRef.current;
    if (!mirror || !container) return;
    const span = mirror.querySelector(`[data-fix-start="${hoverFix.start}"]`) as HTMLElement | null;
    if (!span) {
      setPopupPos(null);
      spanRectRef.current = null;
      return;
    }
    const spanRect = span.getBoundingClientRect();
    spanRectRef.current = spanRect;
    const containerRect = container.getBoundingClientRect();
    setPopupPos({
      left: spanRect.left - containerRect.left,
      top: spanRect.bottom - containerRect.top + 2,
    });
  }, [hoverFix]);

  // Clamp pass: center the popup under the word, then keep it inside the
  // editor by measuring its actual width (words near the right edge must not
  // push the popup away from the word). Settles in one extra render — the
  // state setter returns the previous object when the left is unchanged.
  useEffect(() => {
    if (!popupPos || !hoverFix) return;
    const popup = popupRef.current;
    const container = editorWrapRef.current;
    const spanRect = spanRectRef.current;
    if (!popup || !container || !spanRect) return;
    const popupWidth = popup.getBoundingClientRect().width;
    const containerRect = container.getBoundingClientRect();
    const wordCenter = spanRect.left - containerRect.left + spanRect.width / 2;
    let left = wordCenter - popupWidth / 2;
    left = Math.max(4, Math.min(left, containerRect.width - popupWidth - 4));
    setPopupPos((prev) => (prev && prev.left === left ? prev : prev ? { ...prev, left } : prev));
  }, [popupPos, hoverFix]);

  useEffect(() => {
    return () => {
      if (recheckTimerRef.current !== null) clearTimeout(recheckTimerRef.current);
    };
  }, []);

  // Re-check only the sentence that contained the applied word fix. The
  // engine corrects sentences independently, so untouched sentences would
  // produce identical results — re-checking them is wasted LLM work.
  async function runSentenceCheck(
    replacedText: string,
    sentenceStart: number,
    sentenceText: string,
    parentIndex: number,
    provisionalSuggestions: Suggestion[],
  ) {
    const id = ++requestIdRef.current;
    setRechecking(true);
    setError(null);
    try {
      const result = await api.correct(sentenceText);
      if (id !== requestIdRef.current) return;
      // Insert the fresh results at the fixed sentence's position; the
      // provisional list already removed it and shifted later suggestions.
      const rebased = rebaseRecheckedSuggestions(result, sentenceStart);
      const finalSuggestions = [
        ...provisionalSuggestions.slice(0, parentIndex),
        ...rebased,
        ...provisionalSuggestions.slice(parentIndex),
      ];
      setCorrections({ suggestions: finalSuggestions, correctedText: replacedText });
      setApplied((prev) => shiftApplied(prev, parentIndex, rebased.length));
      setLastCheckedText(replacedText);
      setScore(computeScore(replacedText.trim(), finalSuggestions));
    } catch (e) {
      if (id !== requestIdRef.current) return;
      setError(e instanceof Error ? e.message : String(e));
      setScore(null);
    } finally {
      if (id === requestIdRef.current) setRechecking(false);
    }
  }

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
      setLastCheckedText(text);
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
    setHoverFix(null);
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

  function handleWordFixApply(fix: WordFix) {
    const currentCorrections = corrections;
    if (!currentCorrections) return;
    const { text: replaced, parentIndex, sentenceStart, sentenceText, delta } = applyWordFixToText(
      text,
      fix,
      currentCorrections.suggestions,
    );
    setHoverFix(null);
    setPopupPos(null);
    setText(replaced);
    if (parentIndex === -1) {
      // No parent sentence — nothing to re-check; clear highlights.
      setCorrections(null);
      setApplied(new Set());
      setScore(null);
      return;
    }
    // Keep every other sentence's highlights immediately: drop the fixed
    // sentence from the suggestion list (the re-check result will replace it)
    // and shift later suggestions' offsets to match the edited text.
    const provisional = mergeSentenceRecheck(
      currentCorrections.suggestions,
      applied,
      parentIndex,
      sentenceStart,
      delta,
      { correctedText: replaced, suggestions: [] },
    );
    setCorrections({ suggestions: provisional.suggestions, correctedText: replaced });
    setApplied(provisional.applied);
    setLastCheckedText(replaced);
    setScore(computeScore(replaced.trim(), provisional.suggestions));
    if (recheckTimerRef.current !== null) clearTimeout(recheckTimerRef.current);
    recheckTimerRef.current = window.setTimeout(() => {
      void runSentenceCheck(replaced, sentenceStart, sentenceText, parentIndex, provisional.suggestions);
    }, RECHECK_DELAY_MS);
  }

  function handleMouseMove(e: React.MouseEvent<HTMLDivElement>) {
    if (!overlayActive) return;
    // Mouse button held (text drag/selection): never pop a fix popup over
    // the selection — it would interrupt dragging to the end of the text.
    if (e.buttons !== 0) return;
    // Pointer over the popup: keep it visible instead of recomputing.
    if (popupRef.current && popupRef.current.contains(e.target as Node)) return;
    const mirror = mirrorRef.current;
    if (!mirror) return;
    // Hit-test the mouse point against the highlight spans' rects. The mirror
    // shares the textarea's metrics, so this matches the visible words.
    let found: WordFix | null = null;
    for (const span of mirror.querySelectorAll<HTMLElement>('.wfix')) {
      const rect = span.getBoundingClientRect();
      if (e.clientX >= rect.left && e.clientX <= rect.right && e.clientY >= rect.top && e.clientY <= rect.bottom) {
        const start = Number(span.dataset.fixStart);
        found = wordFixes.find((f) => f.start === start) ?? null;
        break;
      }
    }
    setHoverFix(found);
  }

  function clearHover() {
    setHoverFix(null);
    setPopupPos(null);
  }

  function handleScroll(e: React.UIEvent<HTMLTextAreaElement>) {
    const mirror = mirrorRef.current;
    if (mirror) {
      mirror.scrollTop = e.currentTarget.scrollTop;
      mirror.scrollLeft = e.currentTarget.scrollLeft;
    }
    setHoverFix(null);
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
      <button className="settings-btn" onClick={onOpenSettings} title="Settings">⚙️</button>

      <header className="app-header">
        <h1>GrammarLLM</h1>
        <p>Automated grammar correction and writing quality assessment</p>
      </header>

      <main className="app-main">
        <section className="editor-section">
          <div className="editor-header">
            <h2>Your Text</h2>
            <div className="editor-actions">
              {rechecking && <span className="rechecking-note">Re-checking…</span>}
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
                  setHoverFix(null);
                  if (recheckTimerRef.current !== null) clearTimeout(recheckTimerRef.current);
                }}
              >
                Clear
              </button>
            </div>
          </div>
          <div
            className="editor-wrap"
            ref={editorWrapRef}
            onMouseMove={handleMouseMove}
            onMouseLeave={clearHover}
            onMouseDown={(e) => {
              // Pressing the popup's own button must not dismiss it before
              // the click event lands.
              if (popupRef.current && popupRef.current.contains(e.target as Node)) return;
              clearHover();
            }}
          >
            {overlayActive && (
              <div className="editor-mirror" ref={mirrorRef}>
                {buildSegments(text, wordFixes).map((segment, index) => {
                  if (!segment.isFix || !segment.fix) {
                    return <span key={index}>{segment.text}</span>;
                  }
                  if (segment.fix.original === '') {
                    // Insertion: show the text that would be inserted.
                    return (
                      <span key={index} className="wfix wfix-insert" data-fix-start={segment.fix.start}>
                        {segment.fix.corrected}
                      </span>
                    );
                  }
                  return (
                    <span key={index} className="wfix" data-fix-start={segment.fix.start}>
                      {segment.text}
                    </span>
                  );
                })}
              </div>
            )}
            <textarea
              ref={textareaRef}
              className={overlayActive ? 'editor-input overlay-mode' : 'editor-input'}
              value={text}
              placeholder="Type or paste your text here, then press Ctrl+Enter or click Check Grammar"
              onChange={(e) => handleTextChange(e.target.value)}
              onKeyUp={(e) => { lastCaretRef.current = e.currentTarget.selectionStart; }}
              onKeyDown={(e) => {
                if (e.key === 'Escape') {
                  setHoverFix(null);
                  setPopupPos(null);
                }
              }}
              onClick={(e) => { lastCaretRef.current = e.currentTarget.selectionStart; }}
              onScroll={handleScroll}
            />
            {hoverFix && popupPos && (
              <div
                ref={popupRef}
                className="fix-popup"
                style={{ left: popupPos.left, top: popupPos.top }}
              >
                <button className="fix-popup-btn" onClick={() => handleWordFixApply(hoverFix)}>
                  {hoverFix.original === ''
                    ? `+ ${hoverFix.corrected}`
                    : hoverFix.corrected === ''
                      ? 'Delete'
                      : hoverFix.corrected}
                </button>
              </div>
            )}
          </div>
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

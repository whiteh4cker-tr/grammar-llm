import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import Alert from '@mui/material/Alert';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import CircularProgress from '@mui/material/CircularProgress';
import IconButton from '@mui/material/IconButton';
import Paper from '@mui/material/Paper';
import Snackbar from '@mui/material/Snackbar';
import Typography from '@mui/material/Typography';
import ClearIcon from '@mui/icons-material/Clear';
import SettingsIcon from '@mui/icons-material/Settings';
import SpellcheckIcon from '@mui/icons-material/Spellcheck';
import { api } from '../api';
import type { CorrectionResponse, Suggestion, WordFix } from '../../electron/core/types';
import { SuggestionsList } from './SuggestionsList';
import { ScoreBadge } from './ScoreBadge';
import { ReportButton } from './ReportButton';
import { EditorMirror, EditorTextarea, FixHighlight, FixInsertHighlight } from './editorStyles';
import { SUCCESS_TEXT } from '../theme';
import { buildSegments, normalizeFixes } from '../wordOverlay';
import { applyWordFixToText, mergeSentenceRecheck, rebaseRecheckedSuggestions, shiftApplied } from '../recheck';

const RECHECK_DELAY_MS = 600;

/**
 * How long a fix popup survives after the pointer leaves the highlighted word
 * (or the popup itself). Long enough to cross the few pixels between the word
 * and the popup and click the fix button; short enough to feel responsive.
 */
const HOVER_GRACE_MS = 500;

/** How long a toast stays up. */
const TOAST_MS = 3000;

/**
 * Usable rect of a mirror fix span.
 *
 * An insertion marker is an empty inline box, so its own rect can be
 * zero-sized — which would leave it unhittable and the popup without an anchor.
 * A collapsed Range around the span reports the caret-sized rect at the same
 * spot, so fall back to that.
 */
function fixSpanRect(span: HTMLElement): DOMRect {
  const rect = span.getBoundingClientRect();
  if (rect.height > 0) return rect;
  const range = span.ownerDocument.createRange();
  range.setStartBefore(span);
  range.setEndAfter(span);
  // Layout-less hosts (jsdom) implement Element rects as zeros and no Range
  // metrics at all; there is nothing better to fall back to there.
  if (typeof range.getBoundingClientRect !== 'function') return rect;
  const rangeRect = range.getBoundingClientRect();
  return rangeRect.height > 0 ? rangeRect : rect;
}

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
  const hoverGraceRef = useRef<number | null>(null);
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
    const spanRect = fixSpanRect(span);
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
      if (hoverGraceRef.current !== null) clearTimeout(hoverGraceRef.current);
    };
  }, []);

  // Timed here rather than left to Snackbar's `autoHideDuration`, which arms
  // its own timer in the enter transition's `onEntered` — a transition that may
  // never run (reduced motion, hidden window), leaving the toast on screen.
  useEffect(() => {
    if (!toast) return;
    const id = setTimeout(() => setToast(null), TOAST_MS);
    return () => clearTimeout(id);
  }, [toast]);

  function cancelHoverGrace() {
    if (hoverGraceRef.current !== null) {
      clearTimeout(hoverGraceRef.current);
      hoverGraceRef.current = null;
    }
  }

  function clearHoverNow() {
    cancelHoverGrace();
    setHoverFix(null);
    setPopupPos(null);
  }

  /**
   * Hide the popup after a short grace period instead of the moment the
   * pointer leaves the highlighted word, so the user can travel down to the
   * popup and press its fix button. Re-entering a fix span or the popup itself
   * cancels it via `cancelHoverGrace()`.
   */
  function scheduleHoverClear() {
    if (hoverGraceRef.current !== null) return;
    hoverGraceRef.current = window.setTimeout(() => {
      hoverGraceRef.current = null;
      setHoverFix(null);
      setPopupPos(null);
    }, HOVER_GRACE_MS);
  }

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
    clearHoverNow();
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
    clearHoverNow();
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
    // Pointer over the popup: keep it visible instead of recomputing, and drop
    // any pending grace-period dismissal (the user is on their way to click).
    if (popupRef.current && popupRef.current.contains(e.target as Node)) {
      cancelHoverGrace();
      return;
    }
    const mirror = mirrorRef.current;
    if (!mirror) return;
    // Hit-test the mouse point against the highlight spans' rects. The mirror
    // shares the textarea's metrics, so this matches the visible words.
    let found: WordFix | null = null;
    for (const span of mirror.querySelectorAll<HTMLElement>('.wfix')) {
      const rect = fixSpanRect(span);
      // An insertion point is a hairline, so give it a few pixels of target.
      const padX = rect.width < 2 ? 4 : 0;
      if (
        e.clientX >= rect.left - padX &&
        e.clientX <= rect.right + padX &&
        e.clientY >= rect.top &&
        e.clientY <= rect.bottom
      ) {
        const start = Number(span.dataset.fixStart);
        found = wordFixes.find((f) => f.start === start) ?? null;
        break;
      }
    }
    if (found) {
      cancelHoverGrace();
      setHoverFix(found);
      return;
    }
    if (hoverFix) scheduleHoverClear();
  }

  function handleScroll(e: React.UIEvent<HTMLTextAreaElement>) {
    const mirror = mirrorRef.current;
    if (mirror) {
      mirror.scrollTop = e.currentTarget.scrollTop;
      mirror.scrollLeft = e.currentTarget.scrollLeft;
    }
    // Positions go stale as soon as the text moves — no grace period here.
    clearHoverNow();
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

  function handleClearAll() {
    setText('');
    setCorrections(null);
    setApplied(new Set());
    setScore(null);
    setError(null);
    clearHoverNow();
    if (recheckTimerRef.current !== null) clearTimeout(recheckTimerRef.current);
  }

  return (
    <Box sx={{ p: { xs: 0.75, md: 1.25 } }}>
      <IconButton
        onClick={onOpenSettings}
        title="Settings"
        sx={{
          position: 'fixed',
          top: 1.25,
          right: 1.25,
          zIndex: 1200,
          color: '#fff',
          bgcolor: 'rgba(255, 255, 255, 0.12)',
          border: '1px solid rgba(255, 255, 255, 0.25)',
          backdropFilter: 'blur(10px)',
          '&:hover': { bgcolor: 'rgba(255, 255, 255, 0.25)' },
        }}
      >
        <SettingsIcon />
      </IconButton>

      <Box sx={{ textAlign: 'center', mb: 2.5, px: 1, color: '#fff' }}>
        <Typography variant="h4" sx={{ textShadow: '2px 2px 4px rgba(0, 0, 0, 0.3)' }}>
          GrammarLLM
        </Typography>
        <Typography variant="body1" sx={{ opacity: 0.9 }}>
          Automated grammar correction and writing quality assessment
        </Typography>
      </Box>

      <Box
        sx={{
          display: 'grid',
          gap: 1.875,
          gridTemplateColumns: { xs: '1fr', md: '2fr 1fr' },
          gridTemplateRows: { xs: 'minmax(0, 2fr) minmax(0, 1fr)', md: 'minmax(0, 1fr)' },
          height: { xs: 'calc(100vh - 100px)', md: 'calc(100vh - 120px)' },
          maxHeight: '80vh',
        }}
      >
        <Paper elevation={6} sx={{ p: { xs: 1.875, md: 2.5 }, display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0, minWidth: 0, borderRadius: 1.25 }}>
          <Box
            sx={{
              display: 'flex',
              flexWrap: 'wrap',
              justifyContent: 'space-between',
              alignItems: 'center',
              gap: 1.25,
              mb: 1.875,
              flexShrink: 0,
            }}
          >
            <Typography variant="h6">Your Text</Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.25 }}>
              {rechecking && (
                <Typography
                  variant="caption"
                  sx={{ display: 'inline-flex', alignItems: 'center', gap: 0.75, fontStyle: 'italic', color: 'text.secondary' }}
                >
                  <CircularProgress size={12} />
                  Re-checking…
                </Typography>
              )}
              <Button
                variant="contained"
                startIcon={loading ? <CircularProgress size={16} color="inherit" /> : <SpellcheckIcon />}
                onClick={() => void handleCheck()}
                disabled={loading}
              >
                {loading ? 'Checking…' : 'Check Grammar'}
              </Button>
              <Button
                variant="outlined"
                startIcon={<ClearIcon />}
                onClick={handleClearAll}
                sx={{
                  borderColor: 'divider',
                  color: 'text.primary',
                  '&:hover': { borderColor: 'primary.main', bgcolor: 'action.hover' },
                }}
              >
                Clear
              </Button>
            </Box>
          </Box>

          <Box
            sx={{ position: 'relative', flex: 1, display: 'flex', minHeight: 0 }}
            ref={editorWrapRef}
            onMouseMove={handleMouseMove}
            onMouseLeave={scheduleHoverClear}
            onMouseDown={(e) => {
              // Pressing the popup's own button must not dismiss it before
              // the click event lands.
              if (popupRef.current && popupRef.current.contains(e.target as Node)) return;
              clearHoverNow();
            }}
          >
            {overlayActive && (
              <EditorMirror ref={mirrorRef}>
                {buildSegments(text, wordFixes).map((segment, index) => {
                  if (!segment.isFix || !segment.fix) {
                    return <span key={index}>{segment.text}</span>;
                  }
                  if (segment.fix.original === '') {
                    // Insertion: an empty marker only. Rendering the inserted
                    // word here would push the rest of the line away from the
                    // caret, which the textarea positions on the real text.
                    return (
                      <FixInsertHighlight key={index} className="wfix wfix-insert" data-fix-start={segment.fix.start} />
                    );
                  }
                  return (
                    <FixHighlight key={index} className="wfix" data-fix-start={segment.fix.start}>
                      {segment.text}
                    </FixHighlight>
                  );
                })}
                {/* A trailing newline is a painted line in a textarea but is
                    folded away by `pre-wrap`; without it the mirror is a line
                    shorter than the text and the caret sits on the wrong line
                    at the very end. */}
                {text.endsWith('\n') ? '\n' : ''}
              </EditorMirror>
            )}
            <EditorTextarea
              ref={textareaRef}
              $overlay={overlayActive}
              value={text}
              placeholder="Type or paste your text here, then press Ctrl+Enter or click Check Grammar"
              onChange={(e) => handleTextChange(e.target.value)}
              onKeyUp={(e) => { lastCaretRef.current = e.currentTarget.selectionStart; }}
              onKeyDown={(e) => {
                if (e.key === 'Escape') clearHoverNow();
              }}
              onClick={(e) => { lastCaretRef.current = e.currentTarget.selectionStart; }}
              onScroll={handleScroll}
            />
            {overlayActive && hoverFix && popupPos && (
              <Paper
                ref={popupRef}
                elevation={8}
                onMouseEnter={cancelHoverGrace}
                onMouseLeave={scheduleHoverClear}
                sx={{
                  position: 'absolute',
                  zIndex: 20,
                  left: popupPos.left,
                  top: popupPos.top,
                  p: 0.5,
                  display: 'flex',
                }}
              >
                {/* Text button: MUI would paint `success.main`, a fill colour
                    that is too dark to read as small text on the popup. */}
                <Button
                  size="small"
                  color="success"
                  sx={{ color: ({ palette }) => SUCCESS_TEXT[palette.mode] }}
                  onClick={() => handleWordFixApply(hoverFix)}
                >
                  {hoverFix.original === ''
                    ? `+ ${hoverFix.corrected}`
                    : hoverFix.corrected === ''
                      ? 'Delete'
                      : hoverFix.corrected}
                </Button>
              </Paper>
            )}
          </Box>
        </Paper>

        <Paper elevation={6} sx={{ p: { xs: 1.875, md: 2.5 }, display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0, minWidth: 0, borderRadius: 1.25 }}>
          <Box
            sx={{
              display: 'flex',
              flexWrap: 'wrap',
              justifyContent: 'space-between',
              alignItems: 'center',
              gap: 1.25,
              mb: 1.875,
              flexShrink: 0,
            }}
          >
            <Typography variant="h6">Suggestions</Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.25 }}>
              {score !== null && <ScoreBadge score={score} />}
              <ReportButton suggestions={corrections?.suggestions ?? []} score={score} />
            </Box>
          </Box>
          <SuggestionsList
            suggestions={corrections?.suggestions ?? []}
            applied={applied}
            loading={loading}
            error={error}
            onApply={handleApply}
            onHover={handleSuggestionHover}
            onLeave={handleSuggestionLeave}
          />
        </Paper>
      </Box>

      <Snackbar
        key={toast ? toast.message : 'toast'}
        open={toast !== null}
        autoHideDuration={TOAST_MS}
        onClose={() => setToast(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert
          severity={toast?.isError ? 'error' : 'success'}
          variant="filled"
          onClose={() => setToast(null)}
          sx={{ alignItems: 'center' }}
        >
          {toast?.message}
        </Alert>
      </Snackbar>
    </Box>
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

import { alpha, styled } from '@mui/material/styles';

/**
 * Metrics shared by the editor textarea and the overlay mirror.
 *
 * `GrammarApp` paints word-level fix highlights in a mirror element layered
 * behind a transparent textarea. The highlights only line up with the real
 * words while BOTH elements share every box metric (padding, border, font,
 * line-height), so the numbers are declared once here instead of being
 * duplicated across two style sheets.
 */
export const EDITOR_METRICS = {
  padding: 15,
  borderWidth: 2,
  fontSize: 16,
  lineHeight: 1.5,
  borderRadius: 8,
} as const;

const editorBox = {
  boxSizing: 'border-box' as const,
  padding: `${EDITOR_METRICS.padding}px`,
  borderRadius: `${EDITOR_METRICS.borderRadius}px`,
  fontSize: `${EDITOR_METRICS.fontSize}px`,
  lineHeight: EDITOR_METRICS.lineHeight,
  fontWeight: 400,
  whiteSpace: 'pre-wrap' as const,
  wordWrap: 'break-word' as const,
  overflowWrap: 'break-word' as const,
};

/** The real input. Text is hidden while `$overlay` so the mirror shows through. */
export const EditorTextarea = styled('textarea')<{ $overlay?: boolean }>(({ theme, $overlay }) => ({
  ...editorBox,
  border: `${EDITOR_METRICS.borderWidth}px solid ${
    theme.palette.mode === 'dark' ? 'rgba(226, 232, 240, 0.23)' : '#e2e8f0'
  }`,
  position: 'relative',
  zIndex: 1,
  width: '100%',
  flex: 1,
  minHeight: 0,
  resize: 'none',
  outline: 'none',
  fontFamily: theme.typography.fontFamily,
  color: $overlay ? 'transparent' : theme.palette.text.primary,
  backgroundColor: $overlay ? 'transparent' : theme.palette.background.paper,
  caretColor: theme.palette.text.primary,
  transition: 'border-color 0.3s ease, background-color 0.3s ease, color 0.3s ease',
  '&:focus': {
    borderColor: theme.palette.primary.main,
  },
}));

/** Highlight layer. Must stay metric-identical to `EditorTextarea`. */
export const EditorMirror = styled('div')(({ theme }) => ({
  ...editorBox,
  // Same box edge as the textarea, but invisible: the JS scrollbar
  // compensation measures `borderWidth` off the textarea and mirrors it here.
  border: `${EDITOR_METRICS.borderWidth}px solid transparent`,
  position: 'absolute',
  inset: 0,
  zIndex: 0,
  overflow: 'hidden',
  pointerEvents: 'none',
  fontFamily: theme.typography.fontFamily,
  backgroundColor: theme.palette.background.paper,
  color: theme.palette.text.primary,
}));

/** Misspelled / corrected word inside the mirror. */
export const FixHighlight = styled('span')(({ theme }) => ({
  borderRadius: 3,
  color: theme.palette.mode === 'dark' ? theme.palette.error.light : theme.palette.error.dark,
  backgroundColor: alpha(theme.palette.error.main, theme.palette.mode === 'dark' ? 0.3 : 0.12),
}));

/**
 * Marker for a word a fix would insert.
 *
 * This box must not change the mirror's layout — not by a character, not by a
 * pixel of line height. The caret is drawn by the *textarea* at the real
 * character offset while the user reads the mirror, so every glyph added here
 * slides all following words away from the caret (it then looks as if the caret
 * stopped short of the end of the text). The indicator is therefore an empty
 * inline box with an out-of-flow mark, and the word it stands for is shown in
 * the hover popup (`+ corrected`) instead of inline.
 */
export const FixInsertHighlight = styled('span')(({ theme }) => ({
  position: 'relative',
  '&::after': {
    content: "''",
    position: 'absolute',
    left: -1,
    top: '0.2em',
    width: 2,
    height: '0.95em',
    borderRadius: 1,
    backgroundColor: alpha(theme.palette.success.main, theme.palette.mode === 'dark' ? 0.95 : 0.85),
  },
}));

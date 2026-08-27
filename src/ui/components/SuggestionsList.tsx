import type { ReactNode } from 'react';
import Alert from '@mui/material/Alert';
import AlertTitle from '@mui/material/AlertTitle';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Card from '@mui/material/Card';
import CircularProgress from '@mui/material/CircularProgress';
import Typography from '@mui/material/Typography';
import { alpha, styled } from '@mui/material/styles';
import type { Suggestion } from '../../electron/core/types';
import { escapeHtml } from '../escapeHtml';
import { SUCCESS_TEXT } from '../theme';

interface Props {
  suggestions: Suggestion[];
  applied: Set<number>;
  loading: boolean;
  error: string | null;
  onApply: (index: number) => void;
  onHover: (suggestion: Suggestion) => void;
  onLeave: () => void;
}

const SCROLLABLE = {
  flex: 1,
  minHeight: 0,
  overflowY: 'auto' as const,
  '&::-webkit-scrollbar': { width: 6 },
  '&::-webkit-scrollbar-track': { bgcolor: 'action.hover', borderRadius: 0.5 },
  '&::-webkit-scrollbar-thumb': { bgcolor: 'action.disabled', borderRadius: 0.5 },
};

/** Original sentence; `.error-word` spans come from the correction engine. */
const OriginalText = styled('div')(({ theme }) => ({
  color: theme.palette.mode === 'dark' ? theme.palette.error.light : theme.palette.error.dark,
  backgroundColor: alpha(theme.palette.error.main, theme.palette.mode === 'dark' ? 0.18 : 0.08),
  borderLeft: `3px solid ${theme.palette.error.main}`,
  borderRadius: 4,
  padding: '8px 10px',
  marginBottom: 8,
  fontSize: '0.9rem',
  wordWrap: 'break-word',
  '& .error-word': {
    fontWeight: 700,
    color: theme.palette.mode === 'dark' ? theme.palette.error.light : theme.palette.error.dark,
    textDecoration: 'underline',
    textDecorationColor: theme.palette.error.main,
    textDecorationThickness: 2,
    textUnderlineOffset: 2,
  },
}));

/** Suggested sentence; `.corrected-word` spans come from the correction engine. */
const CorrectedText = styled('div')(({ theme }) => ({
  color: SUCCESS_TEXT[theme.palette.mode],
  backgroundColor: alpha(theme.palette.success.main, theme.palette.mode === 'dark' ? 0.16 : 0.08),
  borderLeft: `3px solid ${theme.palette.success.main}`,
  borderRadius: 4,
  padding: '8px 10px',
  fontSize: '0.9rem',
  wordWrap: 'break-word',
  '& .corrected-word': {
    fontWeight: 700,
    color: SUCCESS_TEXT[theme.palette.mode],
    backgroundColor: alpha(theme.palette.success.main, theme.palette.mode === 'dark' ? 0.24 : 0.18),
    borderRadius: 3,
    padding: '0 2px',
    textDecoration: 'underline',
    textDecorationColor: theme.palette.success.main,
    textDecorationThickness: 2,
    textUnderlineOffset: 2,
  },
}));

function PanelState({
  title,
  detail,
  icon,
  severity,
}: {
  title: string;
  detail?: string;
  icon?: ReactNode;
  severity?: 'info' | 'error' | 'success';
}) {
  if (severity) {
    return (
      <Box sx={{ ...SCROLLABLE, p: 0.5 }}>
        <Alert severity={severity} variant="outlined">
          <AlertTitle>{title}</AlertTitle>
          {detail}
        </Alert>
      </Box>
    );
  }
  return (
    <Box
      sx={{
        ...SCROLLABLE,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 1,
        textAlign: 'center',
        color: 'text.secondary',
      }}
    >
      {icon}
      <Typography variant="subtitle1">{title}</Typography>
      {detail && <Typography variant="body2">{detail}</Typography>}
    </Box>
  );
}

export function SuggestionsList({ suggestions, applied, loading, error, onApply, onHover, onLeave }: Props) {
  if (loading) {
    return <PanelState title="Checking grammar…" icon={<CircularProgress size={24} />} />;
  }
  if (error) {
    return <PanelState severity="error" title="Error checking grammar" detail={error} />;
  }
  if (suggestions.length === 0) {
    return <PanelState title="No grammar issues found" detail="Your text looks great!" />;
  }

  const unappliedCount = suggestions.filter((_, index) => !applied.has(index)).length;
  if (unappliedCount === 0) {
    return <PanelState title="All suggestions applied!" detail="Your text looks great" />;
  }

  return (
    <Box sx={{ ...SCROLLABLE, display: 'flex', flexDirection: 'column', gap: 1.5 }}>
      {suggestions.map((suggestion, index) => {
        if (applied.has(index)) return null;
        return (
          <Card
            key={index}
            variant="outlined"
            tabIndex={0}
            onMouseEnter={() => onHover(suggestion)}
            onMouseLeave={onLeave}
            onFocus={() => onHover(suggestion)}
            onBlur={onLeave}
            sx={{
              flexShrink: 0,
              p: 1.875,
              borderColor: 'divider',
              transition: 'border-color 0.3s ease, box-shadow 0.3s ease',
              '&:hover': { borderColor: 'primary.main', boxShadow: 2 },
              '&:focus-visible': { borderColor: 'primary.main', outline: 'none' },
            }}
          >
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 1, mb: 1.25 }}>
              <Typography variant="subtitle2" sx={{ minWidth: 0, wordBreak: 'break-word' }}>
                {suggestion.sentence}
              </Typography>
              {/* Text button: `success.main` is a fill colour, too weak as small
                  text on the suggestions card. */}
              <Button
                size="small"
                color="success"
                sx={{ color: ({ palette }) => SUCCESS_TEXT[palette.mode] }}
                onClick={() => onApply(index)}
              >
                Apply
              </Button>
            </Box>
            <OriginalText>
              <strong>Original:</strong>{' '}
              <span dangerouslySetInnerHTML={{ __html: suggestion.original_highlighted || escapeHtml(suggestion.original) }} />
            </OriginalText>
            <CorrectedText>
              <strong>Suggested:</strong>{' '}
              <span dangerouslySetInnerHTML={{ __html: suggestion.corrected_highlighted || escapeHtml(suggestion.corrected) }} />
            </CorrectedText>
          </Card>
        );
      })}
    </Box>
  );
}

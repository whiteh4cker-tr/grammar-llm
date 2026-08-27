import Chip from '@mui/material/Chip';
import Box from '@mui/material/Box';
import { useTheme } from '@mui/material/styles';
import { SCORE_INK } from '../theme';

export function ScoreBadge({ score }: { score: number }) {
  // Token-driven colours rather than `color="success"` alone: the chip paints
  // `success.main` as its text, which is a fill colour (chosen for white labels
  // on green) and fails contrast as text on the panel behind it.
  const theme = useTheme();
  const ink = SCORE_INK[theme.palette.mode];

  return (
    <Chip
      size="small"
      variant="outlined"
      color="success"
      title="Writing quality score"
      sx={{ color: ink.value, borderColor: ink.border }}
      label={
        <Box component="span" sx={{ display: 'inline-flex', alignItems: 'baseline', gap: 0.25 }}>
          {score}
          {/* Solid colour, not `opacity`: blending the number into the panel is
              what dragged the suffix below the AA threshold. */}
          <Box component="span" sx={{ fontSize: '0.72em', color: ink.suffix }}>
            / 100
          </Box>
        </Box>
      }
    />
  );
}

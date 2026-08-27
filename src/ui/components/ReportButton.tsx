import Button from '@mui/material/Button';
import PictureAsPdfIcon from '@mui/icons-material/PictureAsPdf';
import { generatePdfReport } from './pdf';
import type { Suggestion } from '../../electron/core/types';

export function ReportButton({ suggestions, score }: { suggestions: Suggestion[]; score: number | null }) {
  return (
    <Button
      size="small"
      variant="contained"
      color="secondary"
      startIcon={<PictureAsPdfIcon />}
      onClick={() => generatePdfReport(suggestions, score)}
    >
      Download Report
    </Button>
  );
}

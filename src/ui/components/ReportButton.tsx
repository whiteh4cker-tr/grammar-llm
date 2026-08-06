import { generatePdfReport } from './pdf';
import type { Suggestion } from '../../electron/core/types';

export function ReportButton({ suggestions, score }: { suggestions: Suggestion[]; score: number | null }) {
  return (
    <button className="report-btn" onClick={() => generatePdfReport(suggestions, score)}>
      Download Report
    </button>
  );
}

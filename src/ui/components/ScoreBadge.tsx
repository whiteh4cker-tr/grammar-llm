export function ScoreBadge({ score }: { score: number }) {
  return (
    <div className="score-badge" title="Writing quality score">
      <span className="score-value">{score}</span>
      <span className="score-max">/ 100</span>
    </div>
  );
}

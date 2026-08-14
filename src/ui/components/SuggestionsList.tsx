import type { Suggestion } from '../../electron/core/types';
import { escapeHtml } from '../escapeHtml';

interface Props {
  suggestions: Suggestion[];
  applied: Set<number>;
  loading: boolean;
  error: string | null;
  onApply: (index: number) => void;
  onHover: (suggestion: Suggestion) => void;
  onLeave: () => void;
}

export function SuggestionsList({ suggestions, applied, loading, error, onApply, onHover, onLeave }: Props) {
  if (loading) {
    return <div className="empty-state"><p>Checking grammar…</p></div>;
  }
  if (error) {
    return <div className="empty-state error-state"><p>Error checking grammar</p><small>{error}</small></div>;
  }

  if (suggestions.length === 0) {
    return <div className="empty-state"><p>No grammar issues found</p><small>Your text looks great!</small></div>;
  }

  const unappliedCount = suggestions.filter((_, index) => !applied.has(index)).length;
  if (unappliedCount === 0) {
    return <div className="empty-state"><p>All suggestions applied!</p><small>Your text looks great</small></div>;
  }

  return (
    <div className="suggestions-list">
      {suggestions.map((suggestion, index) => {
        if (applied.has(index)) return null;
        return (
          <div
            key={index}
            className="suggestion-item"
            onMouseEnter={() => onHover(suggestion)}
            onMouseLeave={onLeave}
            onFocus={() => onHover(suggestion)}
            onBlur={onLeave}
          >
            <div className="suggestion-header">
              <span className="suggestion-sentence">{suggestion.sentence}</span>
              <button className="apply-btn" onClick={() => onApply(index)}>Apply</button>
            </div>
            <div className="original-text">
              <strong>Original:</strong>{' '}
              <span dangerouslySetInnerHTML={{ __html: suggestion.original_highlighted || escapeHtml(suggestion.original) }} />
            </div>
            <div className="corrected-text-suggestion">
              <strong>Suggested:</strong>{' '}
              <span dangerouslySetInnerHTML={{ __html: suggestion.corrected_highlighted || escapeHtml(suggestion.corrected) }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}


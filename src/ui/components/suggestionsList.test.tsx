// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { Suggestion } from '../../electron/core/types';
import { SuggestionsList } from './SuggestionsList';

const suggestion: Suggestion = {
  original: 'helo world',
  corrected: 'hello world',
  sentence: 'helo world',
  start_index: 0,
  end_index: 10,
  original_highlighted: '<span class="error-word">helo</span> world',
  corrected_highlighted: '<span class="corrected-word">hello</span> world',
  wordFixes: [{ original: 'helo', corrected: 'hello', start: 0, end: 4 }],
};

function renderList(props: Partial<Parameters<typeof SuggestionsList>[0]> = {}) {
  const onApply = vi.fn();
  const onHover = vi.fn();
  const onLeave = vi.fn();
  const view = render(
    <SuggestionsList
      suggestions={[suggestion]}
      applied={new Set()}
      loading={false}
      error={null}
      onApply={onApply}
      onHover={onHover}
      onLeave={onLeave}
      {...props}
    />,
  );
  return { ...view, onApply, onHover, onLeave };
}

afterEach(() => {
  cleanup();
});

describe('SuggestionsList', () => {
  it('shows the sentence and keeps the engine highlight markup', () => {
    const { container } = renderList();
    expect(screen.getByText('helo world')).toBeTruthy();
    const errorWord = container.querySelector('.error-word');
    expect(errorWord?.textContent).toBe('helo');
    expect(container.querySelector('.corrected-word')?.textContent).toBe('hello');
  });

  it('reports which suggestion was applied and hides it', () => {
    const { onApply } = renderList();
    fireEvent.click(screen.getByRole('button', { name: 'Apply' }));
    expect(onApply).toHaveBeenCalledWith(0);

    cleanup();
    renderList({ applied: new Set([0]) });
    expect(screen.queryByRole('button', { name: 'Apply' })).toBeNull();
    expect(screen.getByText('All suggestions applied!')).toBeTruthy();
  });

  it('focuses the sentence in the editor on hover', () => {
    const { onHover, onLeave } = renderList();
    const card = screen.getByRole('button', { name: 'Apply' }).closest('li, div');
    fireEvent.mouseEnter(card as Element);
    expect(onHover).toHaveBeenCalledWith(suggestion);
    fireEvent.mouseLeave(card as Element);
    expect(onLeave).toHaveBeenCalled();
  });

  it('renders loading, error and empty states', () => {
    const { unmount } = renderList({ loading: true });
    expect(screen.getByText('Checking grammar…')).toBeTruthy();
    unmount();

    const second = renderList({ error: 'llama exploded', suggestions: [] });
    expect(screen.getByText('llama exploded')).toBeTruthy();
    second.unmount();

    renderList({ suggestions: [] });
    expect(screen.getByText('No grammar issues found')).toBeTruthy();
  });
});

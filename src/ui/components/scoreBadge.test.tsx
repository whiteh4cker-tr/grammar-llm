// @vitest-environment jsdom
import { cleanup, render } from '@testing-library/react';
import { ThemeProvider } from '@mui/material/styles';
import { afterEach, describe, expect, it } from 'vitest';
import { ScoreBadge } from './ScoreBadge';
import { SCORE_INK, darkTheme, lightTheme } from '../theme';
import { hexToRgb } from '../contrast';

const rgb = (hex: string) => {
  const [r, g, b] = hexToRgb(hex);
  return `rgb(${r}, ${g}, ${b})`;
};

function paint(mode: 'light' | 'dark') {
  const { container } = render(
    <ThemeProvider theme={mode === 'dark' ? darkTheme : lightTheme}>
      <ScoreBadge score={100} />
    </ThemeProvider>,
  );
  const chip = container.querySelector('.MuiChip-root') as HTMLElement;
  // The chip's own label span contains "100/ 100"; only the inner Box is "/ 100".
  const suffix = [...chip.querySelectorAll('span')].find((s) => s.textContent === '/ 100') as HTMLElement;
  return { chip, suffix };
}

afterEach(cleanup);

describe('ScoreBadge', () => {
  for (const mode of ['light', 'dark'] as const) {
    it(`${mode}: paints the accessible score colours rather than success.main`, () => {
      const { chip, suffix } = paint(mode);
      const ink = SCORE_INK[mode];
      expect(getComputedStyle(chip).color).toBe(rgb(ink.value));
      expect(getComputedStyle(chip).borderTopColor).toBe(rgb(ink.border));
      expect(getComputedStyle(suffix).color).toBe(rgb(ink.suffix));
    });

    // `opacity` fades the colour towards the panel, and the ratio is defined for
    // the painted pixel — a solid colour is what can be verified.
    it(`${mode}: does not fade the "/ 100" suffix with opacity`, () => {
      const { suffix } = paint(mode);
      expect(getComputedStyle(suffix).opacity).toBe('1');
    });
  }

  it('renders the score and its maximum', () => {
    const { container } = render(
      <ThemeProvider theme={darkTheme}>
        <ScoreBadge score={100} />
      </ThemeProvider>,
    );
    expect(container.textContent).toBe('100/ 100');
  });
});

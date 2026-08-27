import { describe, expect, it } from 'vitest';
import { contrastOver, contrastRatio, composite, relativeLuminance } from './contrast';
import { DIALOG_SURFACE, SCORE_INK, SUCCESS_TEXT, darkTheme, lightTheme } from './theme';

const themes = { light: lightTheme, dark: darkTheme };

describe('contrast helpers', () => {
  it('matches the WCAG reference values', () => {
    expect(contrastRatio('#000000', '#ffffff')).toBeCloseTo(21, 1);
    expect(contrastRatio('#ffffff', '#ffffff')).toBe(1);
    // A known borderline pair: just under the 4.5 normal-text threshold.
    expect(contrastRatio('#777777', '#ffffff')).toBeLessThan(4.5);
    expect(contrastRatio('#767676', '#ffffff')).toBeGreaterThanOrEqual(4.5);
  });

  it('measures translucent text as the colour it paints', () => {
    expect(composite('#ffffff', '#000000', 1)).toBe('#ffffff');
    expect(composite('#ffffff', '#000000', 0)).toBe('#000000');
    // 75% white on black paints #bfbfbf, whose ratio to black is well under 21.
    expect(contrastOver('#ffffff', '#000000', 0.75)).toBeCloseTo(contrastRatio('#bfbfbf', '#000000'), 5);
  });
});

describe('score readout is WCAG 2.0 AA readable on its panel', () => {
  for (const [mode, theme] of Object.entries(themes)) {
    const paper = theme.palette.background.paper;

    it(`${mode}: the number and the "/ 100" suffix reach 4.5:1 on the panel`, () => {
      const ink = SCORE_INK[mode as 'light' | 'dark'];
      expect(contrastRatio(ink.value, paper), `${mode} score value on ${paper}`).toBeGreaterThanOrEqual(4.5);
      expect(contrastRatio(ink.suffix, paper), `${mode} "/ 100" on ${paper}`).toBeGreaterThanOrEqual(4.5);
    });

    it(`${mode}: the chip outline reaches 3:1 (WCAG 1.4.11)`, () => {
      const ink = SCORE_INK[mode as 'light' | 'dark'];
      expect(contrastRatio(ink.border, paper)).toBeGreaterThanOrEqual(3);
    });
  }
});

describe('settings dialog surface is WCAG 2.0 AA readable', () => {
  for (const [mode, theme] of Object.entries(themes)) {
    const surface = DIALOG_SURFACE[mode as 'light' | 'dark'];

    it(`${mode}: body and secondary text reach 4.5:1 on ${surface}`, () => {
      expect(contrastRatio(theme.palette.text.primary, surface)).toBeGreaterThanOrEqual(4.5);
      expect(contrastRatio(theme.palette.text.secondary, surface)).toBeGreaterThanOrEqual(4.5);
    });

    it(`${mode}: green captions reach 4.5:1 on ${surface}`, () => {
      expect(contrastRatio(SUCCESS_TEXT[mode as 'light' | 'dark'], surface)).toBeGreaterThanOrEqual(4.5);
    });
  }

  it('dark mode lifts the dialog off the lighter panels it opens over', () => {
    expect(relativeLuminance(DIALOG_SURFACE.dark)).toBeLessThan(relativeLuminance(darkTheme.palette.background.paper));
  });
});

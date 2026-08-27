/**
 * WCAG 2.x colour-contrast helpers.
 *
 * The app states its colours in plain hex (two non-CSS-variable MUI themes — see
 * `theme.ts`), so contrast can be checked at build/test time instead of being
 * eyeballed. These are the formulas from WCAG 2.x "Relative Luminance" and
 * "Contrast Ratio", nothing app-specific.
 */

export type Rgb = readonly [number, number, number];

/** AA for normal text (< 18pt, or < 14pt bold). */
export const WCAG_AA_NORMAL = 4.5;
/** AA for large text (>= 18pt, or >= 14pt bold) and UI component boundaries. */
export const WCAG_AA_LARGE = 3;

export function hexToRgb(hex: string): Rgb {
  const digits = hex.trim().replace('#', '');
  const full = digits.length === 3 ? digits.split('').map((c) => c + c).join('') : digits;
  if (!/^[0-9a-fA-F]{6}$/.test(full)) throw new Error(`not a hex colour: ${hex}`);
  return [0, 2, 4].map((i) => parseInt(full.slice(i, i + 2), 16)) as unknown as Rgb;
}

export function rgbToHex([r, g, b]: Rgb): string {
  return '#' + [r, g, b].map((v) => Math.round(Math.min(255, Math.max(0, v))).toString(16).padStart(2, '0')).join('');
}

/** WCAG 2.x relative luminance (0 = black, 1 = white). */
export function relativeLuminance(hex: string): number {
  const channel = hexToRgb(hex).map((v) => {
    const c = v / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * channel[0] + 0.7152 * channel[1] + 0.0722 * channel[2];
}

/** Contrast ratio of two opaque colours: 1 (identical) to 21 (black on white). */
export function contrastRatio(foreground: string, background: string): number {
  const [hi, lo] = [relativeLuminance(foreground), relativeLuminance(background)].sort((a, b) => b - a);
  return (hi + 0.05) / (lo + 0.05);
}

/**
 * Blend a translucent foreground over an opaque background, returning the hex
 * the browser actually paints. Required for `opacity`/`alpha()` text colours:
 * the ratio is defined for painted pixels, so a 70% green on a dark panel has to
 * be measured as the blended colour, not the raw one.
 */
export function composite(foreground: string, background: string, alpha: number): string {
  const fg = hexToRgb(foreground);
  const bg = hexToRgb(background);
  return rgbToHex(fg.map((v, i) => v * alpha + bg[i] * (1 - alpha)) as unknown as Rgb);
}

/** Contrast of a translucent foreground over `background`. */
export function contrastOver(foreground: string, background: string, alpha: number): number {
  return contrastRatio(composite(foreground, background, alpha), background);
}

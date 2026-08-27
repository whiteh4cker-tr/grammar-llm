import { describe, expect, it } from 'vitest';
import { darkTheme, getTheme, lightTheme } from './theme';

describe('theme', () => {
  it('does not uppercase button labels', () => {
    // Material's default is 'uppercase'; the word-fix popup puts real words in
    // buttons, so uppercasing them would misrepresent the correction.
    expect(lightTheme.typography.button.textTransform).toBe('none');
    expect(darkTheme.typography.button.textTransform).toBe('none');
  });

  it('keeps the brand palette across both color schemes', () => {
    expect(getTheme('light')).toBe(lightTheme);
    expect(getTheme('dark')).toBe(darkTheme);
    expect(lightTheme.palette.mode).toBe('light');
    expect(darkTheme.palette.mode).toBe('dark');
    expect(lightTheme.palette.primary.main).toBe('#667eea');
    expect(darkTheme.palette.primary.main).toBe('#667eea');
    expect(lightTheme.palette.secondary.main).toBe('#764ba2');
  });
});

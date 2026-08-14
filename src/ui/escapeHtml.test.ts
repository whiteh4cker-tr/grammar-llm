import { describe, it, expect } from 'vitest';
import { escapeHtml } from './escapeHtml';

describe('escapeHtml', () => {
  it('escapes html special characters', () => {
    expect(escapeHtml('<b>&"\'')).toBe('&lt;b&gt;&amp;&quot;&#039;');
  });

  it('leaves plain text untouched', () => {
    expect(escapeHtml('plain text 123')).toBe('plain text 123');
  });
});

import { describe, expect, it } from 'vitest';
import { effectiveModelStatus } from './modelStatus';

describe('effectiveModelStatus', () => {
  it('keeps a loaded model usable while a second one downloads', () => {
    // This is what the main process reports mid-download: `modelName` has been
    // repointed at the file being fetched, but the loaded model still works.
    expect(effectiveModelStatus({ state: 'downloading', modelName: 'b.gguf' }, 'a.gguf')).toEqual({
      state: 'ready',
      modelName: 'a.gguf',
    });
  });

  it('is blocking when nothing has been loaded yet', () => {
    expect(effectiveModelStatus({ state: 'downloading', modelName: 'b.gguf' }, null)).toEqual({
      state: 'downloading',
      modelName: 'b.gguf',
    });
  });

  it('passes every other state through untouched', () => {
    expect(effectiveModelStatus({ state: 'missing' }, 'a.gguf').state).toBe('missing');
    expect(effectiveModelStatus({ state: 'error', modelName: 'a.gguf' }, 'a.gguf').state).toBe('error');
    expect(effectiveModelStatus({ state: 'ready', modelName: 'b.gguf' }, 'a.gguf')).toEqual({
      state: 'ready',
      modelName: 'b.gguf',
    });
  });
});

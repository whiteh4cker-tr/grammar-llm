import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts'],
  },
  // Vite's default extension order puts .js BEFORE .ts, so relative imports
  // would resolve to stale compiled dist-electron/*.js over fresh .ts source.
  resolve: {
    extensions: ['.ts', '.tsx', '.js', '.jsx', '.json'],
  },
});

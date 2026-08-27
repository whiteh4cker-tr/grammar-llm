// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const apiStub = {
  listModels: vi.fn(async () => ['installed.gguf']),
  selectModel: vi.fn(async () => ({ state: 'ready' as const, modelName: 'installed.gguf' })),
  deleteModel: vi.fn(async () => ({ state: 'missing' as const })),
  downloadModel: vi.fn(async () => {}),
  cancelDownload: vi.fn(async () => {}),
  onDownloadProgress: vi.fn(() => () => {}),
};
(window as unknown as { api: typeof apiStub }).api = apiStub;

async function renderGate() {
  const { ModelGate } = await import('./ModelGate');
  const { DownloadProvider } = await import('../DownloadProvider');
  const view = render(
    <DownloadProvider>
      <ModelGate status={{ state: 'missing' }} />
    </DownloadProvider>,
  );
  await waitFor(() => expect(screen.getByText('installed.gguf')).toBeTruthy());
  return view;
}

beforeEach(() => {
  apiStub.listModels.mockClear();
  apiStub.selectModel.mockClear();
  apiStub.deleteModel.mockClear();
  apiStub.downloadModel.mockClear();
});

afterEach(() => {
  cleanup();
});

describe('ModelGate', () => {
  it('offers both bundled models, a custom URL and the installed model', async () => {
    await renderGate();
    expect(screen.getByRole('radio', { name: /GRMR-V3-G4B-Q4_K_M/ })).toBeTruthy();
    expect(screen.getByRole('radio', { name: /GRMR-V3-G4B-Q8_0/ })).toBeTruthy();
    expect(screen.getByRole('radio', { name: /Custom GGUF URL/ })).toBeTruthy();
    expect(screen.getByRole('button', { name: /Delete/ })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Use' })).toBeTruthy();
  });

  it('routes the selected option card to downloadModel', async () => {
    await renderGate();
    fireEvent.click(screen.getByRole('radio', { name: /GRMR-V3-G4B-Q8_0/ }));
    fireEvent.click(screen.getByRole('button', { name: 'Download GRMR-V3-G4B-Q8_0' }));
    await waitFor(() =>
      expect(apiStub.downloadModel).toHaveBeenCalledWith({
        url: 'https://huggingface.co/icecubetr/GRMR-V3-G4B-GGUF/resolve/main/GRMR-V3-G4B-Q8_0.gguf',
        fileName: 'GRMR-V3-G4B-Q8_0.gguf',
      }),
    );
  });

  it('reveals the URL field only for the custom option and validates it', async () => {
    await renderGate();
    expect(screen.queryByPlaceholderText(/model\.gguf/)).toBeNull();

    fireEvent.click(screen.getByRole('radio', { name: /Custom GGUF URL/ }));
    const field = screen.getByPlaceholderText('https://huggingface.co/.../model.gguf');

    fireEvent.change(field, { target: { value: 'https://example.com/nope.txt' } });
    const disabled = screen.getByRole('button', { name: 'Enter a valid .gguf URL' });
    expect(disabled.hasAttribute('disabled')).toBe(true);

    fireEvent.change(field, { target: { value: 'https://example.com/my.gguf' } });
    const enabled = screen.getByRole('button', { name: 'Download my.gguf' });
    expect(enabled.hasAttribute('disabled')).toBe(false);
    fireEvent.click(enabled);
    await waitFor(() => expect(apiStub.downloadModel).toHaveBeenCalledWith({
      url: 'https://example.com/my.gguf',
      fileName: 'my.gguf',
    }));
  });

  it('switches to an already installed model without downloading', async () => {
    await renderGate();
    fireEvent.click(screen.getByRole('button', { name: 'Use' }));
    await waitFor(() => expect(apiStub.selectModel).toHaveBeenCalledWith({ fileName: 'installed.gguf' }));
    expect(apiStub.downloadModel).not.toHaveBeenCalled();
  });
});

import type { CorrectionResponse, Suggestion } from './core/types.js';

export interface ModelDownloadRequest {
  url: string;
  fileName: string;
}

export interface DownloadProgress {
  percent: number;
  transferred: number;
  total: number;
}

export type ModelState = 'ready' | 'missing' | 'downloading' | 'error';

export interface ModelStatus {
  state: ModelState;
  modelName?: string;
}

export interface ApplySuggestionRequest {
  originalText: string;
  suggestionIndex: number;
  suggestions: Suggestion[];
}

export interface ApplyManyRequest {
  originalText: string;
  suggestions: Suggestion[];
}

export interface IpcApi {
  correct(text: string): Promise<CorrectionResponse>;
  applySuggestion(args: ApplySuggestionRequest): Promise<{ correctedText: string }>;
  applyMany(args: ApplyManyRequest): Promise<{ correctedText: string }>;
  modelStatus(): Promise<ModelStatus>;
  listModels(): Promise<string[]>;
  selectModel(args: { fileName: string }): Promise<ModelStatus>;
  deleteModel(args: { fileName: string }): Promise<ModelStatus>;
  getSettings(): Promise<{ contextSize: number }>;
  setContextSize(args: { contextSize: number }): Promise<{ contextSize: number }>;
  getWordLevelCorrection(): Promise<{ enabled: boolean }>;
  setWordLevelCorrection(args: { enabled: boolean }): Promise<{ enabled: boolean }>;
  downloadModel(args: ModelDownloadRequest): Promise<void>;
  cancelDownload(): Promise<void>;
  onDownloadProgress(cb: (progress: DownloadProgress) => void): () => void;
}

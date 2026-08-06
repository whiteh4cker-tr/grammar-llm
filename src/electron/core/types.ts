export interface SentenceData {
  text: string;
  start: number;
  end: number;
  spanEnd: number;
  gapBeforeStart: number;
}

export interface Suggestion {
  original: string;
  corrected: string;
  sentence: string;
  start_index: number;
  end_index: number;
  original_highlighted: string;
  corrected_highlighted: string;
}

export interface CorrectionResponse {
  suggestions: Suggestion[];
  correctedText: string;
}

export interface SentenceCorrector {
  correct(sentence: string): Promise<string>;
}

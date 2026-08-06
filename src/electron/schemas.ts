import { z } from 'zod';

export const suggestionSchema = z.object({
  original: z.string(),
  corrected: z.string(),
  sentence: z.string(),
  start_index: z.number(),
  end_index: z.number(),
  original_highlighted: z.string(),
  corrected_highlighted: z.string(),
});

export const correctRequestSchema = z.object({ text: z.string() });

export const applyRequestSchema = z.object({
  originalText: z.string(),
  suggestionIndex: z.number().int().nonnegative(),
  suggestions: z.array(suggestionSchema),
});

export const applyManySchema = z.object({
  originalText: z.string(),
  suggestions: z.array(suggestionSchema),
});

export const downloadRequestSchema = z.object({
  url: z.string().url(),
  fileName: z.string(),
});

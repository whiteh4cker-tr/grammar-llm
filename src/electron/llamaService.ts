import { getLlama, LlamaChatSession } from 'node-llama-cpp';
import type { ModelStatus } from './ipc-types';
import type { ModelManager } from './modelManager';
import type { SentenceCorrector } from './core/types';

const SYSTEM_PROMPT =
  'You are a grammar correction assistant. Correct the grammar, spelling, and punctuation of the given text. Return only the corrected text without any additional explanations or prefixes.';

const STOP_SEQUENCES = ['<|endoftext|>', '<|corrected_end|>', '\n\n', 'Corrected:', 'Here is'];

// Sampling params verified against installed node-llama-cpp v3.19.1 types.
// Python mapping: temperature=0.7, top_p=0.95, top_k=40, min_p=0.01,
// frequency_penalty=0.0/presence_penalty=0.0 -> repeatPenalty=1.0, max_tokens=len+20.
// Stop sequences are passed per-prompt via `customStopTriggers` (no constructor option).
export class LlamaCorrectionService implements SentenceCorrector {
  private session: LlamaChatSession | null = null;
  private loadError: unknown = null;
  private loading: Promise<void> | null = null;

  constructor(private readonly modelManager: ModelManager) {}

  async ensureLoaded(): Promise<void> {
    if (this.session) return;
    if (this.loading) return this.loading;

    this.loading = (async () => {
      const modelPath = this.modelManager.getModelPath();
      if (!modelPath) throw new Error('No model found');
      const llama = await getLlama(); // auto-detects CUDA / Metal / Vulkan / CPU
      const model = await llama.loadModel({ modelPath, contextSize: 4096 });
      const context = await model.createContext();
      this.session = new LlamaChatSession({
        contextSequence: context.createContextSequence(),
        systemPrompt: SYSTEM_PROMPT,
        temperature: 0.7,
        topK: 40,
        topP: 0.95,
        minP: 0.01,
        repeatPenalty: 1.0,
      });
    })();

    try {
      await this.loading;
      this.loadError = null;
    } catch (error) {
      this.loadError = error;
      this.loading = null;
      throw error;
    }
  }

  async correct(sentence: string): Promise<string> {
    try {
      await this.ensureLoaded();
      const response = await this.session!.prompt(sentence, {
        maxTokens: sentence.length + 20,
        customStopTriggers: STOP_SEQUENCES,
      });
      return response.trim();
    } catch (error) {
      console.error(`Error correcting sentence '${sentence}':`, error);
      return sentence;
    }
  }

  getStatus(): ModelStatus {
    if (this.loadError) return { state: 'error', modelName: this.modelManager.getModelPath() ?? undefined };
    if (this.session) return { state: 'ready', modelName: this.modelManager.getModelPath() ?? undefined };
    return { state: 'missing' };
  }
}

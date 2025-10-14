from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from llama_cpp import Llama
import re
from typing import List, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Grammar Correction API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class CorrectionRequest(BaseModel):
    text: str

class Suggestion(BaseModel):
    original: str
    corrected: str
    sentence: str
    start_index: int
    end_index: int

class CorrectionResponse(BaseModel):
    suggestions: List[Suggestion]
    corrected_text: str

class ApplySuggestionRequest(BaseModel):
    original_text: str
    suggestion_index: int
    suggestions: List[Suggestion]

class ApplySuggestionsRequest(BaseModel):
    original_text: str
    suggestions: List[Suggestion]

# Initialize the model
llm = None

def initialize_model():
    global llm
    try:
        logger.info("Loading GRMR model...")
        llm = Llama.from_pretrained(
            repo_id="qingy2024/GRMR-V3-G4B-GGUF",
            filename="GRMR-V3-G4B-Q8_0.gguf",
            n_ctx=4096,
            verbose=False
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise e

def split_into_sentences(text: str) -> List[Dict]:
    """Split text into sentences and preserve trailing whitespace for reconstruction.

    We ensure that:
    - Leading whitespace is NOT included in the sentence span (so replacements don't eat spaces before a sentence)
    - Trailing whitespace AFTER the sentence-ending punctuation IS tracked separately (so we can re-append it later)
    This prevents concatenation like "them.I" and avoids overlaps that cause duplicated fragments.
    """
    sentences: List[Dict] = []

    # Match any run of non-ending-punct followed by a sentence-ending punct
    # We will then extend the end to include any FOLLOWING whitespace, but not preceding
    pattern = r"[^.!?]*[.!?]"

    last_span_end = 0
    for match in re.finditer(pattern, text):
        raw_start, raw_end = match.start(), match.end()  # raw_end is right after the terminal punctuation

        # Skip any leading whitespace so start points at the first non-space char of the sentence
        start_no_ws = raw_start
        while start_no_ws < raw_end and text[start_no_ws].isspace():
            start_no_ws += 1

        # Extend end to include any trailing whitespace AFTER punctuation
        span_end = raw_end
        while span_end < len(text) and text[span_end].isspace():
            span_end += 1

        # Sentence (without leading/trailing spaces) that will be sent to the model
        content = text[start_no_ws:raw_end].strip()
        if not content:
            last_span_end = span_end
            continue

        sentences.append({
            'text': content,
            # Content bounds (do not include leading ws or trailing ws)
            'start': start_no_ws,
            'end': raw_end,
            # Full span end including trailing whitespace to be re-appended during reconstruction
            'span_end': span_end,
            # For reference, the gap before this sentence starts (used by reconstruction)
            'gap_before_start': last_span_end
        })

        last_span_end = span_end

    # If no sentences found (or text doesn't end with punctuation), treat the whole text as one sentence
    if not sentences and text.strip():
        # Trim leading spaces from start, keep trailing spaces tracked
        start_no_ws = 0
        while start_no_ws < len(text) and text[start_no_ws].isspace():
            start_no_ws += 1
        raw_end = len(text)
        span_end = len(text)
        content = text[start_no_ws:raw_end].strip()
        if content:
            sentences.append({
                'text': content,
                'start': start_no_ws,
                'end': raw_end,
                'span_end': span_end,
                'gap_before_start': 0
            })

    return sentences

def clean_corrected_text(corrected: str, original: str) -> str:
    """Clean the corrected text to remove common model artifacts"""
    if not corrected:
        return original
    
    # Remove any template tags
    corrected = re.sub(r'<\|.*?\|>', '', corrected).strip()
    
    # Remove the instruction prefix if it was included in response
    instruction_prefixes = [
        "correct the grammar and spelling of this sentence:",
        "here is the corrected sentence:",
        "corrected sentence:",
        "the corrected version is:",
        "grammar correction:",
        "corrected:"
    ]
    
    for prefix in instruction_prefixes:
        if corrected.lower().startswith(prefix):
            corrected = corrected[len(prefix):].strip()
            # Remove any leading colon or punctuation
            corrected = re.sub(r'^[:]\s*', '', corrected)
    
    # Remove any repeated phrases (common issue with this model)
    # Look for repeated segments and keep only one instance
    words = corrected.split()
    if len(words) > 10:  # Only check longer texts for repetition
        # Check for repeating patterns
        for i in range(len(words) - 5):
            segment = ' '.join(words[i:i+5])
            if segment in ' '.join(words[i+5:]):
                # Found repetition, remove duplicates
                corrected = ' '.join(words[:i+5])
                break
    
    # Ensure the corrected text maintains proper capitalization
    if original and original[0].isupper() and corrected and corrected[0].islower():
        corrected = corrected[0].upper() + corrected[1:]
    
    # Ensure the sentence ends with proper punctuation if the original did
    if original and original[-1] in '.!?' and corrected and corrected[-1] not in '.!?':
        corrected += original[-1]
    
    return corrected.strip()

def correct_sentence(sentence: str) -> str:
    """Correct a single sentence using the GRMR model with proper formatting"""
    if not sentence.strip() or len(sentence.strip()) < 2:
        return sentence
    
    try:
        # Clean the sentence first
        clean_sentence = sentence.strip()
        
        # Use a system prompt to guide the model behavior
        messages = [
            {"role": "system", "content": "You are a grammar correction assistant. Correct the grammar, spelling, and punctuation of the given text. Return only the corrected text without any additional explanations or prefixes."},
            {"role": "user", "content": clean_sentence}
        ]
        
        response = llm.create_chat_completion(
            messages=messages,
            temperature=0.1,  # Very low temperature for consistent results
            top_p=0.7,
            top_k=20,
            min_p=0.01,
            max_tokens=len(clean_sentence) + 20,  # Limit to prevent excessive output
            stop=["<|endoftext|>", "<|corrected_end|>", "\n\n", "Corrected:", "Here is"]
        )
        
        corrected = response['choices'][0]['message']['content'].strip()
        
        # Clean the corrected text
        corrected = clean_corrected_text(corrected, clean_sentence)
        
        # VALIDATION: Skip correction if the only changes are quote-related
        if is_only_quote_change(clean_sentence, corrected):
            logger.info(f"Skipping correction - only quote changes detected: '{clean_sentence}' -> '{corrected}'")
            return clean_sentence
        
        # Validate the correction - if it's too different from original or contains repetition, use original
        if len(corrected) > len(clean_sentence) * 2:
            logger.warning(f"Correction rejected due to repetition or excessive length. Original: '{clean_sentence}', Corrected: '{corrected}'")
            return clean_sentence
        
        # If cleaning resulted in empty text or no meaningful change, return original
        if not corrected or corrected == clean_sentence:
            return clean_sentence
            
        return corrected
        
    except Exception as e:
        logger.error(f"Error correcting sentence '{sentence}': {e}")
        return sentence

def is_only_quote_change(original: str, corrected: str) -> bool:
    """Check if the only differences between original and corrected are quote characters"""
    if original == corrected:
        return False

    # Create versions with normalized quotes for comparison
    normalized_original = original.replace('’', "'").replace('‘', "'").replace('“', '"').replace('”', '"')
    normalized_corrected = corrected.replace('’', "'").replace('‘', "'").replace('“', '"').replace('”', '"')
    
    # If normalized versions are identical, then only quotes changed
    if normalized_original == normalized_corrected:
        return True
    
    # Also check if the changes are only in whitespace or very minor
    original_stripped = original.strip()
    corrected_stripped = corrected.strip()
    
    if original_stripped.replace('’', "'").replace('‘', "'") == corrected_stripped.replace('’', "'").replace('‘', "'"):
        return True
        
    return False

def reconstruct_text_from_sentences(original_text: str, sentence_data: List[Dict], corrected_sentences: List[str]) -> str:
    """Reconstruct text from corrected sentences while preserving original spacing.

    We re-append:
    - The gap before each sentence (from the end of the previous sentence's full span)
    - The original trailing whitespace after each sentence's ending punctuation
    """
    if len(sentence_data) != len(corrected_sentences):
        return original_text

    result_parts: List[str] = []
    last_span_end = 0

    for sent_data, corrected in zip(sentence_data, corrected_sentences):
        start = sent_data['start']
        end = sent_data['end']
        span_end = sent_data.get('span_end', end)
        gap_before_start = sent_data.get('gap_before_start', last_span_end)

        # Append any gap that existed before the sentence (leading spaces, newlines, etc.)
        if gap_before_start > last_span_end:
            result_parts.append(original_text[last_span_end:gap_before_start])
        elif start > last_span_end:
            # Fallback to preserve any text between the last span and this sentence start
            result_parts.append(original_text[last_span_end:start])

        # Append corrected content
        result_parts.append(corrected)

        # Append original trailing whitespace that followed the sentence-ending punctuation
        if span_end > end:
            result_parts.append(original_text[end:span_end])

        last_span_end = span_end

    # Append any remaining text after the final sentence
    if last_span_end < len(original_text):
        result_parts.append(original_text[last_span_end:])

    return ''.join(result_parts)

@app.on_event("startup")
async def startup_event():
    initialize_model()

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

@app.post("/correct", response_model=CorrectionResponse)
async def correct_text(request: CorrectionRequest):
    try:
        text = request.text.strip()
        if not text:
            return CorrectionResponse(suggestions=[], corrected_text="")
        
        # Split into sentences with positions
        sentence_data = split_into_sentences(text)
        suggestions = []
        corrected_sentences = []
        
        logger.info(f"Processing {len(sentence_data)} sentences")
        
        # Process each sentence
        for i, sent_data in enumerate(sentence_data):
            sentence = sent_data['text']
            logger.info(f"Sentence {i+1}: '{sentence}'")
            
            if len(sentence) < 2:  # Skip very short sentences
                corrected_sentences.append(sentence)
                continue
                
            corrected = correct_sentence(sentence)
            logger.info(f"Corrected {i+1}: '{corrected}'")
            
            corrected_sentences.append(corrected)
            
            # Only add to suggestions if there's a meaningful difference
            if (corrected.lower().strip() != sentence.lower().strip() and 
                corrected.strip() != sentence.strip() and
                len(corrected) <= len(sentence) * 1.5 and # Don't suggest if correction is too long
                not is_only_quote_change(sentence, corrected)):  # Don't suggest if only quotes changed)
                suggestions.append(Suggestion(
                    original=sentence,
                    corrected=corrected,
                    sentence=f"Sentence {i+1}",
                    start_index=sent_data['start'],
                    end_index=sent_data['end']
                ))
        
        # Reconstruct the corrected text while preserving original structure
        corrected_text = reconstruct_text_from_sentences(text, sentence_data, corrected_sentences)
        
        logger.info(f"Original: '{text}'")
        logger.info(f"Corrected: '{corrected_text}'")
        
        return CorrectionResponse(
            suggestions=suggestions,
            corrected_text=corrected_text
        )
        
    except Exception as e:
        logger.error(f"Error processing correction request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/apply-suggestion")
async def apply_suggestion(request: ApplySuggestionRequest):
    try:
        text = request.original_text
        suggestion_index = request.suggestion_index
        suggestions = request.suggestions
        
        if suggestion_index < 0 or suggestion_index >= len(suggestions):
            raise HTTPException(status_code=400, detail="Invalid suggestion index")
        
        suggestion = suggestions[suggestion_index]
        
        # Apply the single suggestion robustly: verify span, otherwise find closest match
        start, end = suggestion.start_index, suggestion.end_index
        applied_text = text
        if 0 <= start <= end <= len(text) and text[start:end] == suggestion.original:
            applied_text = text[:start] + suggestion.corrected + text[end:]
        else:
            # Find all exact matches and choose the closest occurrence to the expected index
            occurrences = [m.span() for m in re.finditer(re.escape(suggestion.original), text)]
            if occurrences:
                # Pick the span with minimal distance to expected start
                target_span = min(occurrences, key=lambda sp: abs(sp[0] - start))
                t_start, t_end = target_span
                applied_text = text[:t_start] + suggestion.corrected + text[t_end:]
            else:
                # If not found, leave text unchanged
                applied_text = text
        
        return {"corrected_text": applied_text}
        
    except Exception as e:
        logger.error(f"Error applying suggestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def apply_suggestions_bulk(text: str, suggestions: List[Suggestion]) -> str:
    """Apply multiple suggestions safely in reverse order to avoid index drift.

    Overlapping suggestions are resolved by keeping the later (rightmost) replacement
    and skipping earlier ones that would overlap its span in the original text.
    """
    if not suggestions:
        return text

    # Sort by start index descending so earlier text indices remain valid while applying
    sorted_suggestions = sorted(suggestions, key=lambda s: s.start_index, reverse=True)

    # Track covered ranges (list of non-overlapping intervals in ORIGINAL indices)
    applied_intervals: List[tuple] = []

    result_text = text
    for s in sorted_suggestions:
        start, end = s.start_index, s.end_index
        if start is None or end is None or start < 0 or end < 0 or start > end:
            continue

        # Determine candidate span in the CURRENT result_text
        candidate_span = None
        if end <= len(result_text) and result_text[start:end] == s.original:
            candidate_span = (start, end)
        else:
            # Find exact matches of original sentence in the current text
            occurrences = [m.span() for m in re.finditer(re.escape(s.original), result_text)]
            if occurrences:
                candidate_span = min(occurrences, key=lambda sp: abs(sp[0] - start))

        if candidate_span is None:
            continue

        c_start, c_end = candidate_span

        # Skip if this suggestion overlaps any region we already replaced (in current coordinates)
        overlaps = False
        for a_start, a_end in applied_intervals:
            if not (c_end <= a_start or c_start >= a_end):
                overlaps = True
                break
        if overlaps:
            continue

        result_text = result_text[:c_start] + s.corrected + result_text[c_end:]

        # Record this interval relative to the CURRENT text region we just replaced
        applied_intervals.append((c_start, c_end))

    return result_text

@app.post("/apply-suggestions")
async def apply_suggestions_endpoint(request: ApplySuggestionsRequest):
    try:
        text = request.original_text
        suggestions = request.suggestions
        corrected = apply_suggestions_bulk(text, suggestions)
        return {"corrected_text": corrected}
    except Exception as e:
        logger.error(f"Error applying suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": llm is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

let currentCorrections = {
    suggestions: [],
    correctedText: ''
};

let appliedSuggestions = new Set();
let originalTextForCorrection = '';

async function correctText() {
    const inputText = document.getElementById('inputText').value.trim();
    const suggestionsDiv = document.getElementById('suggestionsList');
    const loading = document.getElementById('loading');
    const countBadge = document.getElementById('suggestionsCount');

    if (!inputText) {
        alert('Please enter some text to check grammar.');
        return;
    }

    // Store original text for this correction session
    originalTextForCorrection = inputText;

    // Show loading
    loading.style.display = 'flex';
    suggestionsDiv.innerHTML = '<div class="empty-state"><p>Checking grammar...</p></div>';

    try {
        const response = await fetch('/correct', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: inputText })
        });

        if (!response.ok) {
            throw new Error('Server error: ' + response.status);
        }

        const data = await response.json();
        
        // Store current corrections for apply functionality
        currentCorrections = data;
        appliedSuggestions.clear();

        // Display suggestions
        displaySuggestions(data.suggestions);

    } catch (error) {
        console.error('Error:', error);
        suggestionsDiv.innerHTML = '<div class="empty-state" style="color: #e53e3e;"><p>Error checking grammar</p><small>Please try again</small></div>';
        countBadge.textContent = '0';
    } finally {
        loading.style.display = 'none';
    }
}

function displaySuggestions(suggestions) {
    const suggestionsDiv = document.getElementById('suggestionsList');
    const countBadge = document.getElementById('suggestionsCount');

    if (suggestions && suggestions.length > 0) {
        const unappliedSuggestions = suggestions.filter((_, index) => !appliedSuggestions.has(index));
        
        if (unappliedSuggestions.length > 0) {
            suggestionsDiv.innerHTML = unappliedSuggestions.map((suggestion, originalIndex) => {
                const globalIndex = suggestions.indexOf(suggestion);
                return `
                    <div class="suggestion-item" data-index="${globalIndex}" data-start="${suggestion.start_index}" data-end="${suggestion.end_index}">
                        <div class="suggestion-header">
                            <span class="suggestion-sentence">${suggestion.sentence}</span>
                            <button class="apply-btn" onclick="applySingleSuggestion(${globalIndex})">Apply</button>
                        </div>
                        <div class="original-text">
                            <strong>Original:</strong> ${escapeHtml(suggestion.original)}
                        </div>
                        <div class="corrected-text-suggestion">
                            <strong>Suggested:</strong> ${escapeHtml(suggestion.corrected)}
                        </div>
                    </div>
                `;
            }).join('');

            // Attach hover listeners to highlight corresponding text in the textarea
            attachSuggestionHoverHandlers();
        } else {
            suggestionsDiv.innerHTML = '<div class="empty-state"><p>All suggestions applied!</p><small>Your text looks great</small></div>';
        }
        
        countBadge.textContent = unappliedSuggestions.length.toString();
    } else {
        suggestionsDiv.innerHTML = '<div class="empty-state"><p>No grammar issues found</p><small>Your text looks great!</small></div>';
        countBadge.textContent = '0';
    }
}

let lastCaretPosition = 0;
const inputEl = document.getElementById('inputText');
inputEl.addEventListener('keyup', () => { lastCaretPosition = inputEl.selectionStart; });
inputEl.addEventListener('click', () => { lastCaretPosition = inputEl.selectionStart; });

function attachSuggestionHoverHandlers() {
    const items = document.querySelectorAll('.suggestion-item');
    items.forEach((el) => {
        el.addEventListener('mouseenter', onSuggestionHover);
        el.addEventListener('mouseleave', clearTextHighlight);
        el.addEventListener('focusin', onSuggestionHover);
        el.addEventListener('focusout', clearTextHighlight);
    });
}

function onSuggestionHover(e) {
    const el = e.currentTarget;
    const startAttr = el.getAttribute('data-start');
    const endAttr = el.getAttribute('data-end');
    const index = parseInt(el.getAttribute('data-index'), 10);
    if (!currentCorrections.suggestions || isNaN(index)) return;
    const sug = currentCorrections.suggestions[index];
    if (!sug) return;

    const currentText = inputEl.value;
    const approxStart = startAttr ? parseInt(startAttr, 10) : 0;
    const bestSpan = findBestOccurrence(currentText, sug.original, isFinite(approxStart) ? approxStart : 0);
    if (bestSpan) {
        highlightSentence(bestSpan[0], bestSpan[1]);
    } else if (startAttr && endAttr) {
        // Fallback to provided indices clamped to current text
        const s = Math.max(0, Math.min(currentText.length, parseInt(startAttr, 10)));
        const eIdx = Math.max(0, Math.min(currentText.length, parseInt(endAttr, 10)));
        if (eIdx > s) highlightSentence(s, eIdx);
    }
}

function highlightSentence(start, end) {
    try {
        inputEl.focus();
        inputEl.setSelectionRange(start, end);
    } catch (_) {
        // Ignore selection errors
    }
}

function clearTextHighlight() {
    try {
        inputEl.setSelectionRange(lastCaretPosition, lastCaretPosition);
    } catch (_) {
        // Ignore
    }
}

function findBestOccurrence(haystack, needle, approxIndex) {
    if (!needle || !haystack) return null;
    const escaped = needle.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const re = new RegExp(escaped, 'g');
    const matches = [];
    let m;
    while ((m = re.exec(haystack)) !== null) {
        matches.push([m.index, m.index + needle.length]);
        // Prevent infinite loop on zero-length
        if (m.index === re.lastIndex) re.lastIndex++;
    }
    if (matches.length === 0) return null;
    // Choose span whose start is closest to approxIndex
    let best = matches[0];
    let bestDist = Math.abs(best[0] - approxIndex);
    for (let i = 1; i < matches.length; i++) {
        const dist = Math.abs(matches[i][0] - approxIndex);
        if (dist < bestDist) {
            best = matches[i];
            bestDist = dist;
        }
    }
    return best;
}

async function applySingleSuggestion(suggestionIndex) {
    if (!currentCorrections.suggestions || !currentCorrections.suggestions[suggestionIndex]) {
        return;
    }

    const textarea = document.getElementById('inputText');
    const currentText = textarea.value;

    try {
        const response = await fetch('/apply-suggestion', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                original_text: currentText,
                suggestion_index: suggestionIndex,
                suggestions: currentCorrections.suggestions
            })
        });

        if (!response.ok) {
            throw new Error('Failed to apply suggestion');
        }

        const result = await response.json();
        
        // Update the textarea with the partially corrected text
        textarea.value = result.corrected_text;
        
        // Mark this suggestion as applied
        appliedSuggestions.add(suggestionIndex);
        
        // Remove the suggestion from the UI with animation
        const suggestionElement = document.querySelector(`.suggestion-item[data-index="${suggestionIndex}"]`);
        if (suggestionElement) {
            suggestionElement.classList.add('suggestion-removing');
            setTimeout(() => {
                displaySuggestions(currentCorrections.suggestions);
            }, 300);
        }

        // Show confirmation
        showToast(`Applied correction for ${currentCorrections.suggestions[suggestionIndex].sentence}`);

    } catch (error) {
        console.error('Error applying suggestion:', error);
        showToast('Error applying suggestion. Please try again.', true);
    }
}

function clearText() {
    document.getElementById('inputText').value = '';
    document.getElementById('suggestionsList').innerHTML = '<div class="empty-state"><p>No grammar issues found yet</p><small>Start writing and click "Check Grammar" to see suggestions</small></div>';
    document.getElementById('suggestionsCount').textContent = '0';
    currentCorrections = { suggestions: [], correctedText: '' };
    appliedSuggestions.clear();
    originalTextForCorrection = '';
}

function escapeHtml(unsafe) {
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function showToast(message, isError = false) {
    // Create toast element
    const toast = document.createElement('div');
    toast.textContent = message;
    toast.style.cssText = `
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: ${isError ? '#e53e3e' : '#48bb78'};
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        z-index: 1000;
        animation: slideIn 0.3s ease;
    `;
    
    document.body.appendChild(toast);
    
    // Remove toast after 3 seconds
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => {
            if (toast.parentNode) {
                document.body.removeChild(toast);
            }
        }, 300);
    }, 3000);
}

// Smart text editing detection
let lastKnownText = '';
let isUserTyping = false;
let typingTimer = null;

function checkForTextChanges() {
    const currentText = document.getElementById('inputText').value;
    
    // If the text has changed significantly (not just the applied suggestion)
    if (currentText !== lastKnownText) {
        // Check if this change is likely a user edit (not an applied suggestion)
        const isLikelyUserEdit = !isApplyingSuggestion && 
                                currentText.length !== lastKnownText.length && 
                                !isSimpleReplacement(lastKnownText, currentText);
        
        if (isLikelyUserEdit) {
            // User has edited the text manually - clear suggestions to avoid confusion
            currentCorrections = { suggestions: [], correctedText: '' };
            appliedSuggestions.clear();
            document.getElementById('suggestionsList').innerHTML = '<div class="empty-state"><p>No grammar issues found yet</p><small>Start writing and click "Check Grammar" to see suggestions</small></div>';
            document.getElementById('suggestionsCount').textContent = '0';
        }
        
        lastKnownText = currentText;
    }
}

function isSimpleReplacement(oldText, newText) {
    // Check if the change is likely just an applied suggestion (small replacement)
    const diffLength = Math.abs(oldText.length - newText.length);
    return diffLength < 20; // If change is small, it's probably an applied suggestion
}

let isApplyingSuggestion = false;

// Add keyboard shortcut (Ctrl/Cmd + Enter to check grammar)
document.addEventListener('keydown', function(e) {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        correctText();
    }
});

// Initialize text tracking
const textarea = document.getElementById('inputText');
textarea.addEventListener('input', function() {
    isUserTyping = true;
    
    // Clear any existing timer
    if (typingTimer) {
        clearTimeout(typingTimer);
    }
    
    // Set a new timer to check for changes after user stops typing
    typingTimer = setTimeout(() => {
        checkForTextChanges();
        isUserTyping = false;
    }, 1000); // Wait 1 second after user stops typing
});

// Track when suggestions are being applied
const originalFetch = window.fetch;
window.fetch = function(...args) {
    if (args[0] === '/apply-suggestion') {
        isApplyingSuggestion = true;
        return originalFetch.apply(this, args).finally(() => {
            setTimeout(() => {
                isApplyingSuggestion = false;
            }, 100);
        });
    }
    return originalFetch.apply(this, args);
};

// Initialize last known text
lastKnownText = textarea.value;

// Add CSS for animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
`;
document.head.appendChild(style);
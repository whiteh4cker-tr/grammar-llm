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

        // Calculate and display writing quality score
        updateWritingQuality(inputText, data.suggestions);

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
                
                // Use highlighted versions if available, otherwise fall back to escaped text
                const originalDisplay = suggestion.original_highlighted || escapeHtml(suggestion.original);
                const correctedDisplay = suggestion.corrected_highlighted || escapeHtml(suggestion.corrected);
                
                return `
                    <div class="suggestion-item" data-index="${globalIndex}" data-start="${suggestion.start_index}" data-end="${suggestion.end_index}">
                        <div class="suggestion-header">
                            <span class="suggestion-sentence">${suggestion.sentence}</span>
                            <button class="apply-btn" onclick="applySingleSuggestion(${globalIndex})">Apply</button>
                        </div>
                        <div class="original-text">
                            <strong>Original:</strong> ${originalDisplay}
                        </div>
                        <div class="corrected-text-suggestion">
                            <strong>Suggested:</strong> ${correctedDisplay}
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
    // Hide writing quality header
    document.getElementById('writingQualityHeader').style.display = 'none';
}

function updateWritingQuality(text, suggestions) {
    const header = document.getElementById('writingQualityHeader');
    const scoreEl = document.getElementById('writingQualityScore');

    // Count total words
    const words = text.trim().split(/\s+/).filter(w => w.length > 0);
    const totalWords = words.length;

    // Count errors by counting error-word spans across all suggestions
    let errorCount = 0;
    if (suggestions && suggestions.length > 0) {
        suggestions.forEach(s => {
            const highlighted = s.original_highlighted || '';
            const matches = highlighted.match(/<span class="error-word">/g);
            if (matches) {
                errorCount += matches.length;
            }
        });
    }

    // Calculate score: 0–100
    let score;
    if (totalWords === 0) {
        score = 100;
    } else {
        score = Math.max(0, Math.round(100 * (1 - errorCount / totalWords)));
    }

    scoreEl.textContent = score;
    header.style.display = 'flex';
}

function downloadReport() {
    if (!window.jspdf || !window.jspdf.jsPDF) {
        alert('PDF library failed to load. Please check your internet connection and refresh the page.');
        return;
    }
    try {
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF({ unit: 'mm', format: 'a4' });
    const suggestions = currentCorrections.suggestions || [];
    const scoreEl = document.getElementById('writingQualityScore');
    const score = scoreEl ? scoreEl.textContent : '—';

    const pageW = doc.internal.pageSize.getWidth();
    const marginL = 20;
    const marginR = 20;
    const usableW = pageW - marginL - marginR;
    let y = 20;

    // WCAG 2.0 AA compliant colors on white background (contrast >= 4.5:1)
    const COLOR_TITLE    = [41, 41, 41];       // #292929 – near-black
    const COLOR_BODY     = [51, 51, 51];       // #333333
    const COLOR_ERROR    = [163, 28, 28];      // #A31C1C – dark red (7.5:1)
    const COLOR_CORRECT  = [21, 111, 56];      // #156F38 – dark green (5.2:1)
    const COLOR_LABEL    = [80, 80, 80];       // #505050
    const COLOR_LINE     = [180, 180, 180];    // #B4B4B4
    const COLOR_SCORE    = [21, 111, 56];      // same dark green for score
    const COLOR_BG_ORIG  = [254, 226, 226];    // #FEE2E2 – light red bg
    const COLOR_BG_CORR  = [220, 252, 231];    // #DCFCE7 – light green bg

    function checkPage(needed) {
        if (y + needed > doc.internal.pageSize.getHeight() - 15) {
            doc.addPage();
            y = 20;
        }
    }

    // ── Title ──
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(22);
    doc.setTextColor(...COLOR_TITLE);
    doc.text('Writing Quality Report', pageW / 2, y, { align: 'center' });
    y += 10;

    // ── Accent line ──
    doc.setDrawColor(102, 126, 234); // #667eea brand purple
    doc.setLineWidth(0.8);
    doc.line(marginL, y, pageW - marginR, y);
    y += 10;

    // ── Score ──
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(14);
    doc.setTextColor(...COLOR_BODY);
    doc.text('Writing Quality Score:', marginL, y);
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(20);
    doc.setTextColor(...COLOR_SCORE);
    const scoreText = score + ' / 100';
    doc.text(scoreText, marginL + 58, y);
    y += 12;

    // ── Divider ──
    doc.setDrawColor(...COLOR_LINE);
    doc.setLineWidth(0.3);
    doc.line(marginL, y, pageW - marginR, y);
    y += 8;

    // ── Suggestions heading ──
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(16);
    doc.setTextColor(...COLOR_TITLE);
    doc.text('Suggestions', marginL, y);
    y += 8;

    if (suggestions.length === 0) {
        checkPage(10);
        doc.setFont('helvetica', 'italic');
        doc.setFontSize(12);
        doc.setTextColor(...COLOR_BODY);
        doc.text('No grammar issues found. Great writing!', marginL, y);
        y += 10;
    } else {
        suggestions.forEach((s, i) => {
            checkPage(40);

            // Sentence label
            doc.setFont('helvetica', 'bold');
            doc.setFontSize(12);
            doc.setTextColor(...COLOR_TITLE);
            doc.text((i + 1) + '. ' + s.sentence, marginL, y);
            y += 7;

            // ── Original row ──
            const origTokens = parseHighlightedTokens(s.original_highlighted || s.original, 'error-word');
            y = renderHighlightedRow(doc, 'Original:', origTokens, marginL, y, usableW, COLOR_LABEL, COLOR_BODY, COLOR_ERROR, COLOR_BG_ORIG);
            y += 2;

            // ── Corrected row ──
            const corrTokens = parseHighlightedTokens(s.corrected_highlighted || s.corrected, 'corrected-word');
            y = renderHighlightedRow(doc, 'Suggested:', corrTokens, marginL, y, usableW, COLOR_LABEL, COLOR_BODY, COLOR_CORRECT, COLOR_BG_CORR);
            y += 6;

            // Light separator between suggestions
            if (i < suggestions.length - 1) {
                doc.setDrawColor(...COLOR_LINE);
                doc.setLineWidth(0.15);
                doc.line(marginL + 5, y, pageW - marginR - 5, y);
                y += 6;
            }
        });
    }

    // ── Footer ──
    checkPage(16);
    y += 4;
    doc.setDrawColor(102, 126, 234);
    doc.setLineWidth(0.5);
    doc.line(marginL, y, pageW - marginR, y);
    y += 6;
    doc.setFont('helvetica', 'italic');
    doc.setFontSize(9);
    doc.setTextColor(...COLOR_LABEL);
    doc.text('Generated by GrammarLLM', pageW / 2, y, { align: 'center' });

    doc.save('writing-quality-report.pdf');
    } catch (err) {
        console.error('PDF generation error:', err);
        alert('Failed to generate PDF report: ' + err.message);
    }
}

/**
 * Parse an HTML-highlighted string into tokens: [{text, highlighted: bool}, ...]
 * Works with spans like <span class="error-word">word</span>
 */
function parseHighlightedTokens(html, spanClass) {
    if (!html) return [{ text: '', highlighted: false }];
    const tokens = [];
    // Match <span class="...">...</span> or plain text between them
    const regex = new RegExp('<span class="' + spanClass.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '">([^<]*)<\/span>', 'g');
    let lastIndex = 0;
    let match;
    while ((match = regex.exec(html)) !== null) {
        if (match.index > lastIndex) {
            tokens.push({ text: html.substring(lastIndex, match.index), highlighted: false });
        }
        tokens.push({ text: match[1], highlighted: true });
        lastIndex = regex.lastIndex;
    }
    if (lastIndex < html.length) {
        tokens.push({ text: html.substring(lastIndex), highlighted: false });
    }
    // Strip any remaining HTML tags from plain segments
    tokens.forEach(t => { t.text = t.text.replace(/<[^>]*>/g, ''); });
    return tokens;
}

/**
 * Render a labeled row of highlighted tokens into the PDF with word-wrap.
 * Returns the new Y position.
 */
function renderHighlightedRow(doc, label, tokens, marginL, y, usableW, colorLabel, colorNormal, colorHighlight, bgColor) {
    const fontSize = 11;
    const lineH = 5.5;
    const labelW = 24;
    const contentX = marginL + labelW;
    const contentW = usableW - labelW;

    // Label
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(fontSize);
    doc.setTextColor(...colorLabel);
    doc.text(label, marginL + 2, y);

    // Build pieces with measured widths for word-wrap
    doc.setFontSize(fontSize);
    const pieces = [];
    tokens.forEach(tok => {
        // Split on space boundaries to allow wrapping
        const parts = tok.text.split(/( )/);
        parts.forEach(p => {
            if (p.length > 0) {
                // Set the correct font style BEFORE measuring width
                doc.setFont('helvetica', tok.highlighted ? 'bold' : 'normal');
                const width = doc.getTextWidth(p);
                pieces.push({ text: p, highlighted: tok.highlighted, width: width });
            }
        });
    });

    // Render pieces with word-wrap
    let curX = contentX;
    let lineY = y;
    let lineStartY = y;
    // Collect lines for background rectangles
    let lineRanges = [{ startY: lineY, maxY: lineY }];

    pieces.forEach(p => {
        if (curX + p.width > contentX + contentW && p.text.trim() !== '') {
            // Wrap to next line
            lineY += lineH;
            curX = contentX;
            // Check for page break
            if (lineY > doc.internal.pageSize.getHeight() - 15) {
                doc.addPage();
                lineY = 20;
            }
            lineRanges.push({ startY: lineY, maxY: lineY });
        }
        if (p.highlighted) {
            // Draw background highlight rect
            doc.setFillColor(...bgColor);
            doc.roundedRect(curX - 0.5, lineY - 3.5, p.width + 1, lineH, 0.8, 0.8, 'F');
            // Draw highlighted text (bold + colored)
            doc.setFont('helvetica', 'bold');
            doc.setTextColor(...colorHighlight);
            doc.text(p.text, curX, lineY);
            doc.setFont('helvetica', 'normal');
        } else {
            doc.setTextColor(...colorNormal);
            doc.text(p.text, curX, lineY);
        }
        curX += p.width;
    });

    return lineY + lineH;
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

// Theme toggle functionality
(function() {
    let themeToggleInitialized = false;
    
    function initTheme() {
        if (themeToggleInitialized) return;
        
        const themeToggle = document.getElementById('themeToggle');
        const body = document.body;
        
        if (!themeToggle || !body) {
            // Retry after a short delay if element not found
            setTimeout(initTheme, 50);
            return;
        }
        
        themeToggleInitialized = true;
        
        // Check localStorage or default to dark mode
        const savedTheme = localStorage.getItem('theme');
        // Default to dark mode (null or 'dark')
        const isDarkMode = savedTheme !== 'light';
        
        // Apply theme immediately
        if (isDarkMode) {
            body.classList.add('dark-mode');
        } else {
            body.classList.remove('dark-mode');
        }
        
        function updateThemeToggle(theme) {
            if (theme === 'dark') {
                themeToggle.textContent = '🌙';
                themeToggle.title = 'Dark Mode (Click to switch to Light Mode)';
            } else {
                themeToggle.textContent = '☀️';
                themeToggle.title = 'Light Mode (Click to switch to Dark Mode)';
            }
        }
        
        // Update toggle display based on initial theme
        updateThemeToggle(isDarkMode ? 'dark' : 'light');
        
        // Add click handler
        themeToggle.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            
            const isDark = body.classList.contains('dark-mode');
            if (isDark) {
                body.classList.remove('dark-mode');
                localStorage.setItem('theme', 'light');
                updateThemeToggle('light');
            } else {
                body.classList.add('dark-mode');
                localStorage.setItem('theme', 'dark');
                updateThemeToggle('dark');
            }
        });
    }
    
    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initTheme);
    } else {
        // DOM already ready
        initTheme();
    }
})();

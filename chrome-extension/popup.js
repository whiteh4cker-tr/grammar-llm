// Default API URL
const DEFAULT_API_URL = 'http://localhost:8000';

// Get elements
const selectedTextEl = document.getElementById('selectedText');
const hintTextEl = document.getElementById('hintText');
const checkGrammarBtn = document.getElementById('checkGrammarBtn');
const copyCorrectedBtn = document.getElementById('copyCorrectedBtn');
const refreshBtn = document.getElementById('refreshBtn');
const suggestionsListEl = document.getElementById('suggestionsList');
const suggestionsCountEl = document.getElementById('suggestionsCount');
const loadingEl = document.getElementById('loading');
const errorEl = document.getElementById('error');
const apiUrlInput = document.getElementById('apiUrl');
const saveSettingsBtn = document.getElementById('saveSettingsBtn');

let currentSelectedText = '';
let currentCorrections = {
    suggestions: [],
    correctedText: ''
};

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    await loadSettings();
    await getSelectedText();
    setupEventListeners();
});

function setupEventListeners() {
    checkGrammarBtn.addEventListener('click', checkGrammar);
    copyCorrectedBtn.addEventListener('click', copyCorrectedText);
    refreshBtn.addEventListener('click', getSelectedText);
    saveSettingsBtn.addEventListener('click', saveSettings);
}

async function loadSettings() {
    const result = await chrome.storage.local.get(['apiUrl']);
    const apiUrl = result.apiUrl || DEFAULT_API_URL;
    apiUrlInput.value = apiUrl;
}

async function saveSettings() {
    const apiUrl = apiUrlInput.value.trim() || DEFAULT_API_URL;
    await chrome.storage.local.set({ apiUrl });
    showError('Settings saved!', false);
    setTimeout(() => {
        errorEl.style.display = 'none';
    }, 2000);
}

async function getSelectedText() {
    try {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        
        // Inject content script to get selected text
        try {
            const results = await chrome.scripting.executeScript({
                target: { tabId: tab.id },
                func: getSelectedTextFromPage
            });
            
            const selectedText = results[0]?.result || '';
            updateSelectedText(selectedText);
        } catch (scriptError) {
            // Some pages (like chrome:// pages) cannot be accessed
            console.error('Script injection error:', scriptError);
            updateSelectedText('', 'Error: Cannot access page content. Try on a regular webpage (not chrome:// pages).');
        }
    } catch (error) {
        console.error('Error getting selected text:', error);
        updateSelectedText('', 'Error: Cannot access page content.');
    }
}

function getSelectedTextFromPage() {
    const selection = window.getSelection();
    return selection.toString().trim();
}

function updateSelectedText(text, hint = null) {
    currentSelectedText = text;
    
    if (text) {
        selectedTextEl.textContent = text;
        selectedTextEl.classList.remove('empty');
        checkGrammarBtn.disabled = false;
        if (hint) {
            hintTextEl.textContent = hint;
        } else {
            hintTextEl.textContent = `${text.length} characters selected`;
        }
    } else {
        selectedTextEl.textContent = 'No text selected';
        selectedTextEl.classList.add('empty');
        checkGrammarBtn.disabled = true;
        copyCorrectedBtn.disabled = true;
        hintTextEl.textContent = hint || 'Highlight text on the page and click refresh';
        currentCorrections = { suggestions: [], correctedText: '' };
        displaySuggestions([]);
    }
}

async function checkGrammar() {
    if (!currentSelectedText.trim()) {
        showError('Please select some text first.');
        return;
    }

    const apiUrl = apiUrlInput.value.trim() || DEFAULT_API_URL;
    
    // Show loading
    loadingEl.style.display = 'flex';
    errorEl.style.display = 'none';
    suggestionsListEl.innerHTML = '<div class="empty-state"><p>Checking grammar...</p></div>';
    suggestionsCountEl.textContent = '0';
    checkGrammarBtn.disabled = true;

    try {
        const response = await fetch(`${apiUrl}/correct`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: currentSelectedText })
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}. Make sure the API is running on ${apiUrl}`);
        }

        const data = await response.json();
        
        // Store corrections
        currentCorrections = data;
        copyCorrectedBtn.disabled = !data.corrected_text;

        // Display suggestions
        displaySuggestions(data.suggestions);

    } catch (error) {
        console.error('Error:', error);
        showError(error.message || 'Error checking grammar. Make sure the API is running.');
        suggestionsListEl.innerHTML = '<div class="empty-state" style="color: #e53e3e;"><p>Error checking grammar</p><small>Please try again</small></div>';
        suggestionsCountEl.textContent = '0';
        copyCorrectedBtn.disabled = true;
    } finally {
        loadingEl.style.display = 'none';
        checkGrammarBtn.disabled = false;
    }
}

function displaySuggestions(suggestions) {
    suggestionsCountEl.textContent = suggestions.length.toString();

    if (suggestions.length === 0) {
        suggestionsListEl.innerHTML = '<div class="empty-state"><p>No grammar issues found!</p><small>Your text looks good</small></div>';
        return;
    }

    suggestionsListEl.innerHTML = suggestions.map((suggestion, index) => `
        <div class="suggestion-item">
            <div class="suggestion-header">
                <span class="suggestion-sentence">${suggestion.sentence}</span>
            </div>
            <div class="original-text">
                <strong>Original:</strong> ${escapeHtml(suggestion.original)}
            </div>
            <div class="corrected-text-suggestion">
                <strong>Corrected:</strong> ${escapeHtml(suggestion.corrected)}
            </div>
        </div>
    `).join('');
}

async function copyCorrectedText() {
    if (!currentCorrections.corrected_text) {
        return;
    }

    try {
        await navigator.clipboard.writeText(currentCorrections.corrected_text);
        showError('Corrected text copied to clipboard!', false);
        setTimeout(() => {
            errorEl.style.display = 'none';
        }, 2000);
    } catch (error) {
        console.error('Error copying text:', error);
        showError('Failed to copy text');
    }
}

function showError(message, isError = true) {
    errorEl.textContent = message;
    errorEl.style.display = 'block';
    if (!isError) {
        errorEl.style.background = '#c6f6d5';
        errorEl.style.color = '#22543d';
        errorEl.style.borderLeftColor = '#48bb78';
    } else {
        errorEl.style.background = '#fed7d7';
        errorEl.style.color = '#e53e3e';
        errorEl.style.borderLeftColor = '#e53e3e';
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}



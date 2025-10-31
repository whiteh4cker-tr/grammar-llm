// Content script to handle text selection
// This script runs on all pages to capture selected text

// Listen for messages from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'getSelectedText') {
        const selection = window.getSelection();
        const selectedText = selection.toString().trim();
        sendResponse({ selectedText: selectedText });
    }
    return true;
});

// Store selection in sessionStorage when user selects text
document.addEventListener('mouseup', () => {
    const selection = window.getSelection();
    if (selection && selection.toString().trim()) {
        sessionStorage.setItem('grammarChecker_selectedText', selection.toString().trim());
    }
});


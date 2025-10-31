# Grammar Checker Chrome Extension

A Chrome extension that integrates with the GrammarLLM application to check grammar for selected text on any webpage.

## Features

- **Select text** on any webpage
- **Click the extension icon** to open the popup
- **Check grammar** with AI-powered corrections
- **View suggestions** with original and corrected text
- **Apply suggestions** individually
- **Copy corrected text** to clipboard
- **Customizable API URL** for your local server

## Setup Instructions

### 1. Prepare Icons

The extension requires three icon files:
- `icon16.png` (16x16 pixels)
- `icon48.png` (48x48 pixels)
- `icon128.png` (128x128 pixels)

You can:
- Use the existing logo from `../static/img/grammar-llm.png` and resize it
- Create your own icons
- Use a placeholder icon generator

**Quick solution:** You can use any image editing tool or online icon generator to create these icons from the existing logo.

### 2. Start the Grammar-LLM API

Make sure your grammar checking API is running:

```bash
# From the project root
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Or using Docker:
```bash
docker-compose up -d
```

The extension is configured to connect to `http://localhost:8000` by default.

### 3. Load the Extension in Chrome

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable **Developer mode** (toggle in the top right)
3. Click **Load unpacked**
4. Select the `chrome-extension` folder
5. The extension should now appear in your extensions list

### 4. Configure API URL (Optional)

If your API is running on a different URL or port:

1. Click the extension icon
2. Scroll down to the Settings section
3. Enter your API URL (e.g., `http://localhost:8000`)
4. Click **Save**

## Usage

1. **Navigate to any webpage**
2. **Highlight/select the text** you want to check
3. **Click the extension icon** in your Chrome toolbar
4. The selected text will appear in the popup
5. Click **"Check Grammar"** to analyze the text
6. Review suggestions and:
   - Click **"Apply"** on individual suggestions to update the text in the popup
   - Click **"Copy Corrected"** to copy the fully corrected text to clipboard
7. Use the **refresh button** to get newly selected text from the page

## Troubleshooting

### Extension shows "No text selected"
- Make sure you've highlighted text on the webpage before clicking the extension
- Click the refresh button after selecting text
- Some pages (like Chrome's internal pages) may not allow text selection

### "Server error" or "Cannot access page content"
- Make sure the Grammar-LLM API is running on `http://localhost:8000`
- Check that the API URL in settings is correct
- Verify CORS is enabled in the API (it should be by default)

### Suggestions not appearing
- The text might already be grammatically correct
- Check the browser console for any errors
- Verify the API is responding correctly by visiting `http://localhost:8000/health` in your browser

## Development

The extension uses:
- **Manifest V3** (latest Chrome extension standard)
- **Content Scripts** to capture selected text
- **Chrome Storage API** for settings persistence
- **Chrome Scripting API** to inject scripts and read page content

## Files Structure

```
chrome-extension/
├── manifest.json       # Extension configuration
├── popup.html         # Popup UI
├── popup.css          # Popup styling
├── popup.js           # Main extension logic
├── content.js         # Content script for page interaction
├── icon16.png         # Extension icon (16x16)
├── icon48.png         # Extension icon (48x48)
├── icon128.png        # Extension icon (128x128)
└── README.md          # This file
```


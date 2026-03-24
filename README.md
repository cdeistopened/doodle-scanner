# SCANDOC 9000

A document scanner + OCR tool with a retro Xerox copier aesthetic. Scan physical documents with your phone camera or upload PDFs. Auto-classifies and files everything.

**Two modes, one app:**
- **Scan mode** — Camera or PDF upload → auto-classify → file into organized folders
- **Book OCR** — Upload long PDFs → chunked markdown extraction with boundary smoothing

Powered by Google Gemini. Bring your own API key.

## Quick Start

```bash
# Clone
git clone https://github.com/cdeistopened/doodle-scanner.git
cd doodle-scanner

# Install
pip install -r requirements.txt

# Set your Gemini API key
export GEMINI_API_KEY=your_key_here

# Run
python web_app.py
```

Open **http://localhost:5001** in your browser.

## Get a Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Click "Create API Key"
3. Copy the key and set it as `GEMINI_API_KEY`

Free tier includes generous usage. Scan mode costs ~$0.001 per document.

## How It Works

### Scan Mode (default)
1. Open the app — camera viewfinder is ready
2. Point at a document, tap **SCAN PAGE**
3. For multi-page docs, keep scanning pages
4. Tap **NEXT DOC** to mark a document boundary — previous doc starts processing in the background
5. Keep scanning while classification happens
6. When done, tap **FINISH** — choose "Classify & File" or "Just Save PDFs"

**Motion detection**: Toggle it on for hands-free scanning. Flip a page, the app detects the motion, waits for the page to settle, and auto-captures.

**Book presets**: Frame guides for Open Book, Paperback, Letter, or Custom dimensions.

### Upload Mode
1. Drop a PDF on the upload zone (or click to choose file)
2. Choose **Classify & File** (quick, ~$0.001/doc) or **Book OCR** (full markdown, ~$0.005/page)
3. Results appear in the output column

### Output
Documents are filed to `~/scandoc-output/` by category:
```
~/scandoc-output/
├── auto/
│   ├── oil-change-receipt-jiffy-lube_2026-03-22.pdf
│   └── oil-change-receipt-jiffy-lube_2026-03-22.txt
├── medical/
├── financial/
├── books/
│   └── my-book_ocr.md
└── _index.json
```

12 auto-detected categories: auto, medical, financial, insurance, legal, education, home, personal, receipt, government, business, misc.

`_index.json` is a machine-readable manifest of everything scanned.

## Configuration

| Setting | Default | Notes |
|---------|---------|-------|
| `GEMINI_API_KEY` | (required) | Google AI API key |
| `--output` / `-o` | `~/scandoc-output` | Output directory |
| `--port` / `-p` | `5001` | Server port |

In-app settings (Settings tab):
- **Motion sensitivity** — How much movement triggers page-turn detection
- **Capture delay** — How long the page must be still before auto-capture
- **Book OCR options** — Include page numbers, page breaks

## Models

| Mode | Model | Cost |
|------|-------|------|
| Scan & Classify | `gemini-3.1-flash-lite-preview` | ~$0.001/doc |
| Book OCR | `gemini-3-flash-preview` | ~$0.005/page |

## Tech Stack

- **Backend**: Flask (Python)
- **OCR**: Google Gemini API
- **Camera**: Browser `getUserMedia` + custom motion detection
- **Image processing**: Pillow, PyMuPDF
- **Frontend**: Tailwind CSS, IBM Plex fonts, Material Symbols
- **Design**: Retro Xerox copier aesthetic (Solarized warm palette)

## Files

| File | Purpose |
|------|---------|
| `web_app.py` | Flask app — routes, scan pipeline, Book OCR manager |
| `templates/index.html` | SPA frontend — scanner, library, settings views |
| `pdf_pipeline.py` | Chunked OCR for long PDFs (Book OCR mode) |
| `ocr_gemini.py` | Image processing — downsample, combine to PDF |
| `docx_export.py` | Markdown → Word export |
| `DESIGN.md` | Design system reference |

## License

MIT

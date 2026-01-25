# Doodle Scanner

Unified OCR service for Doodle Reader. Two routes:

1. **Upload PDF** → Doodle OCR pipeline with intelligent chunking
2. **Camera Scan** → PageSnap with motion detection

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the web app
python web_app.py
```

Open http://localhost:5001 in your browser.

## Architecture

```
pagesnap/
├── web_app.py        # Flask app with landing, upload, and camera routes
├── pdf_pipeline.py   # PDF OCR with Gemini (chunking, analysis)
├── ocr_gemini.py     # Image OCR for camera captures
├── pagesnap.py       # CLI version of PageSnap
├── uploads/          # Uploaded PDFs (auto-created)
├── output/           # OCR output files (auto-created)
└── sessions/         # Camera capture sessions (auto-created)
```

## Environment

Requires `GEMINI_API_KEY` in environment or `.env` file.

## Routes

| Path | Description |
|------|-------------|
| `/` | Landing page with Upload/Camera choice |
| `/upload` | PDF upload and job management |
| `/camera` | Camera scanning with motion detection |
| `/api/upload` | POST PDF file |
| `/api/jobs` | List processing jobs |
| `/api/jobs/<id>` | Get job status |
| `/api/jobs/<id>/start` | Start processing |
| `/api/jobs/<id>/download` | Download OCR output |

## Deployment

Configured for Railway deployment. See `railway.json` and `Procfile`.

#!/usr/bin/env python3
"""
Scandoc 9000 — Unified document scanner + OCR.

One screen. Two inputs (camera + PDF). One output stream.
Scan mode: classify + file to ~/scandoc-output/{category}/
Book OCR: chunked markdown extraction via pdf_pipeline.py
"""
from __future__ import annotations

import os
import io
import json
import time
import uuid
import base64
import shutil
import threading
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Tuple

from flask import Flask, render_template, jsonify, request, send_file, redirect
from werkzeug.utils import secure_filename

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

# =============================================================================
# Config
# =============================================================================

DEFAULT_OUTPUT_DIR = os.path.expanduser("~/scandoc-output")
SCAN_MODEL = "gemini-3.1-flash-lite-preview"
BOOK_OCR_MODEL = "gemini-3-flash-preview"
MAX_IMAGE_DIMENSION = 1500
JPEG_QUALITY = 85

# Cost estimates (per 1M tokens)
SCAN_COST_IN = 0.25
SCAN_COST_OUT = 1.50
BOOK_COST_IN = 0.50
BOOK_COST_OUT = 3.00
TOKENS_PER_PAGE_IMG = 1800
TOKENS_PER_PAGE_TXT = 400

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024


# =============================================================================
# Shared Utilities
# =============================================================================

def downsample_image(image_bytes: bytes, rotation: int = 0) -> bytes:
    """Downsample to 1500px long edge + optional rotation."""
    if Image is None:
        return image_bytes
    img = Image.open(io.BytesIO(image_bytes))
    if rotation and rotation % 90 == 0:
        img = img.rotate(-rotation, expand=True)
    w, h = img.size
    long_edge = max(w, h)
    if long_edge > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / long_edge
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=JPEG_QUALITY)
    return buf.getvalue()


def images_to_pdf(image_paths: list, output_path: str) -> str:
    """Compile images into a PDF."""
    if fitz is None:
        raise RuntimeError("PyMuPDF not installed")
    doc = fitz.open()
    for img_path in image_paths:
        img = fitz.open(img_path)
        rect = img[0].rect
        pdf_page = doc.new_page(width=rect.width, height=rect.height)
        pdf_page.insert_image(rect, filename=img_path)
        img.close()
    doc.save(output_path)
    doc.close()
    return output_path


def get_gemini_client():
    """Get Gemini API client."""
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("VITE_GEMINI_API_KEY")
    if not api_key:
        for env_path in [Path(__file__).parent / ".env", Path.home() / ".env"]:
            if env_path.exists():
                for line in open(env_path):
                    line = line.strip()
                    if line.startswith("GEMINI_API_KEY="):
                        api_key = line.split("=", 1)[1].strip()
                        break
            if api_key:
                break
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found")
    return genai.Client(api_key=api_key)


def est_scan_cost(pages: int) -> float:
    return round((pages * TOKENS_PER_PAGE_IMG / 1e6) * SCAN_COST_IN + (pages * TOKENS_PER_PAGE_TXT / 1e6) * SCAN_COST_OUT, 4)

def est_book_cost(pages: int) -> float:
    return round((pages * TOKENS_PER_PAGE_IMG / 1e6) * BOOK_COST_IN + (pages * TOKENS_PER_PAGE_TXT * 3 / 1e6) * BOOK_COST_OUT, 4)


# =============================================================================
# Scan Pipeline (from scan_my_life experiment)
# =============================================================================

CLASSIFY_PROMPT = """You are a document scanner assistant. Analyze this scanned document and return a JSON response with:

1. **category**: One of: "auto", "medical", "financial", "insurance", "legal", "education", "home", "personal", "receipt", "government", "business", "misc"
2. **title**: Short descriptive title (e.g., "Oil Change Receipt - Jiffy Lube")
3. **slug**: Filesystem-safe slug, lowercase with hyphens
4. **date**: Date on document in YYYY-MM-DD format, or null
5. **text**: Complete extracted text, plain text with line breaks
6. **summary**: One sentence description

Return ONLY valid JSON, no other text."""


@dataclass
class ScanConfig:
    """Camera/motion detection settings."""
    motion_threshold: float = 3.0
    stability_threshold: float = 1.0
    stability_delay: float = 1.0
    cooldown_period: float = 2.0


class ScanSession:
    """Manages multi-doc scanning with eager background processing."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.work_dir = os.path.join(output_dir, ".work", self.session_id)
        os.makedirs(self.work_dir, exist_ok=True)
        self.current_doc_id: Optional[str] = None
        self.current_doc_pages: list = []
        self.documents: list = []
        self.process_log: list = []
        self.session_cost: float = 0.0
        self.config = ScanConfig()
        self.lock = threading.Lock()
        self._new_document()

    def _new_document(self):
        doc_id = f"doc_{len(self.documents) + 1:03d}"
        os.makedirs(os.path.join(self.work_dir, doc_id), exist_ok=True)
        self.current_doc_id = doc_id
        self.current_doc_pages = []

    def capture_page(self, image_bytes: bytes, rotation: int = 0) -> dict:
        if not self.current_doc_id:
            self._new_document()
        downsampled = downsample_image(image_bytes, rotation=rotation)
        page_num = len(self.current_doc_pages) + 1
        filepath = os.path.join(self.work_dir, self.current_doc_id, f"page_{page_num:03d}.jpg")
        with open(filepath, 'wb') as f:
            f.write(downsampled)
        self.current_doc_pages.append(filepath)
        return {'doc_id': self.current_doc_id, 'page_num': page_num, 'total_pages': len(self.current_doc_pages)}

    def next_document(self) -> dict:
        completed = None
        if self.current_doc_pages:
            doc = {'id': self.current_doc_id, 'page_count': len(self.current_doc_pages),
                   'images': list(self.current_doc_pages), 'status': 'queued'}
            self.documents.append(doc)
            completed = doc
        self._new_document()
        if completed:
            threading.Thread(target=self._process_doc, args=(completed,), daemon=True).start()
        return {'completed_docs': len(self.documents), 'new_doc_id': self.current_doc_id}

    def finalize(self):
        if self.current_doc_pages:
            doc = {'id': self.current_doc_id, 'page_count': len(self.current_doc_pages),
                   'images': list(self.current_doc_pages), 'status': 'captured'}
            self.documents.append(doc)
            self.current_doc_pages = []
            self._new_document()

    def _log(self, msg):
        print(msg)
        self.process_log.append(msg)

    def _process_doc(self, doc: dict):
        """Background: compile PDF → classify → file."""
        doc_id = doc['id']
        pdf_path = os.path.join(self.work_dir, doc_id, f"{doc_id}.pdf")

        # Compile PDF
        try:
            images_to_pdf(doc['images'], pdf_path)
            doc['pdf_path'] = pdf_path
            doc['status'] = 'processing'
            self._log(f"[{doc_id}] PDF compiled ({doc['page_count']}p)")
        except Exception as e:
            self._log(f"[{doc_id}] ERROR: {e}")
            doc['status'] = 'error'
            return

        # Classify + extract
        self._classify_doc(doc, pdf_path)

    def _classify_doc(self, doc: dict, pdf_path: str):
        """Run Gemini classification on a document."""
        doc_id = doc['id']
        try:
            client = get_gemini_client()
            with open(pdf_path, 'rb') as f:
                pdf_b64 = base64.b64encode(f.read()).decode()
            t0 = time.time()
            response = client.models.generate_content(
                model=SCAN_MODEL,
                contents=[{"role": "user", "parts": [
                    {"inline_data": {"mime_type": "application/pdf", "data": pdf_b64}},
                    {"text": CLASSIFY_PROMPT},
                ]}],
                config={"max_output_tokens": 64000, "temperature": 0.1},
            )
            elapsed = time.time() - t0
            cost = est_scan_cost(doc['page_count'])
            self.session_cost += cost

            raw = response.text.strip()
            for prefix in ["```json", "```"]:
                if raw.startswith(prefix):
                    raw = raw[len(prefix):]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

            try:
                result = json.loads(raw)
            except json.JSONDecodeError:
                result = {"category": "misc", "title": f"Document {doc_id}", "slug": doc_id,
                          "date": None, "text": raw, "summary": ""}

            doc.update({
                'category': result.get('category', 'misc'),
                'title': result.get('title', f'Document {doc_id}'),
                'slug': result.get('slug', doc_id),
                'date': result.get('date'),
                'text': result.get('text', ''),
                'summary': result.get('summary', ''),
                'processing_time': round(elapsed, 2),
                'cost': cost, 'status': 'classified',
            })
            self._log(f"[{doc_id}] {doc['category']} / {doc['title']} ({elapsed:.1f}s)")
        except Exception as e:
            self._log(f"[{doc_id}] ERROR classifying: {e}")
            doc.update({'status': 'pdf_only', 'category': 'misc', 'slug': doc_id, 'cost': 0})

        # File it
        self._file_doc(doc)

    def _file_doc(self, doc: dict):
        """File document into category folder + update index."""
        doc_id = doc['id']
        pdf_path = doc.get('pdf_path')
        if not pdf_path:
            return

        try:
            cat = doc.get('category', 'misc')
            slug = doc.get('slug', doc_id)
            date_str = doc.get('date') or datetime.now().strftime("%Y-%m-%d")
            cat_dir = os.path.join(self.output_dir, cat)
            os.makedirs(cat_dir, exist_ok=True)

            final_name = f"{slug}_{date_str}"
            final_pdf = os.path.join(cat_dir, f"{final_name}.pdf")
            c = 1
            while os.path.exists(final_pdf):
                final_pdf = os.path.join(cat_dir, f"{final_name}_{c}.pdf")
                c += 1

            shutil.copy2(pdf_path, final_pdf)
            doc['final_pdf'] = final_pdf

            text = doc.get('text', '')
            if text:
                final_txt = final_pdf.replace('.pdf', '.txt')
                with open(final_txt, 'w') as f:
                    f.write(text)
                doc['final_txt'] = final_txt

            # Update index
            idx_path = os.path.join(self.output_dir, "_index.json")
            idx = []
            if os.path.exists(idx_path):
                try:
                    with open(idx_path) as f:
                        idx = json.load(f)
                except (json.JSONDecodeError, IOError):
                    pass
            idx.append({
                'doc_id': doc_id, 'category': cat, 'title': doc.get('title', ''),
                'slug': slug, 'date': date_str, 'summary': doc.get('summary', ''),
                'page_count': doc['page_count'], 'model': SCAN_MODEL,
                'processing_time': doc.get('processing_time', 0), 'cost': doc.get('cost', 0),
                'pdf': os.path.relpath(final_pdf, self.output_dir),
                'txt': os.path.relpath(doc['final_txt'], self.output_dir) if doc.get('final_txt') else None,
                'scanned_at': datetime.now().isoformat(),
            })
            with open(idx_path, 'w') as f:
                json.dump(idx, f, indent=2)

            doc['status'] = 'filed'
            self._log(f"[{doc_id}] Filed → {cat}/{os.path.basename(final_pdf)}")
        except Exception as e:
            self._log(f"[{doc_id}] ERROR filing: {e}")

    def get_state(self):
        return {
            'session_id': self.session_id,
            'current_doc': self.current_doc_id,
            'current_pages': len(self.current_doc_pages),
            'completed_docs': len(self.documents),
            'session_cost': round(self.session_cost, 4),
            'documents': [{
                'id': d['id'], 'pages': d['page_count'], 'status': d['status'],
                'category': d.get('category'), 'title': d.get('title'),
                'summary': d.get('summary'), 'processing_time': d.get('processing_time'),
                'cost': d.get('cost'),
            } for d in self.documents],
            'process_log': self.process_log[-30:],
        }


# =============================================================================
# Book OCR Manager
# =============================================================================

class BookOCRManager:
    def __init__(self):
        self.jobs = {}
        self.lock = threading.Lock()
        self.upload_dir = os.path.join(os.path.dirname(__file__), "uploads")
        os.makedirs(self.upload_dir, exist_ok=True)

    def create_job(self, filename: str, file_path: str, page_count: int) -> str:
        job_id = str(uuid.uuid4())[:8]
        with self.lock:
            self.jobs[job_id] = {
                'id': job_id, 'filename': filename, 'file_path': file_path,
                'page_count': page_count, 'status': 'pending',
                'progress': 0, 'current_chunk': None, 'output_path': None, 'error': None,
                'estimated_cost': est_book_cost(page_count),
                'created_at': datetime.now().isoformat(),
            }
        return job_id

    def get_job(self, job_id):
        with self.lock:
            return self.jobs[job_id].copy() if job_id in self.jobs else None

    def list_jobs(self):
        with self.lock:
            return sorted([j.copy() for j in self.jobs.values()], key=lambda x: x['created_at'], reverse=True)

    def start(self, job_id):
        job = self.get_job(job_id)
        if not job or job['status'] not in ('pending', 'error'):
            return False, "Not ready"

        def worker():
            try:
                from pdf_pipeline import process_pdf
                out_dir = os.path.join(app.config.get('OUTPUT_DIR', DEFAULT_OUTPUT_DIR), "books")
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{Path(job['filename']).stem}_ocr.md")

                def progress_cb(done, total, desc):
                    with self.lock:
                        if job_id in self.jobs:
                            self.jobs[job_id].update(progress=int(done/total*100) if total else 0, current_chunk=desc)

                with self.lock:
                    self.jobs[job_id]['status'] = 'processing'

                result = process_pdf(job['file_path'], output_path=out_path, progress_callback=progress_cb, model=BOOK_OCR_MODEL)

                with self.lock:
                    if job_id in self.jobs:
                        self.jobs[job_id].update(status='complete', output_path=result['output_path'],
                                                  progress=100, current_chunk=None, total_time=result.get('total_time'))
            except Exception as e:
                with self.lock:
                    if job_id in self.jobs:
                        self.jobs[job_id].update(status='error', error=str(e))

        threading.Thread(target=worker, daemon=True).start()
        return True, None


# =============================================================================
# Library
# =============================================================================

def get_library(output_dir: str) -> dict:
    scans = []
    idx_path = os.path.join(output_dir, "_index.json")
    if os.path.exists(idx_path):
        try:
            with open(idx_path) as f:
                scans = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    categories = {}
    for s in scans:
        c = s.get('category', 'misc')
        categories[c] = categories.get(c, 0) + 1
    return {
        'scans': scans, 'books': book_ocr.list_jobs(), 'categories': categories,
        'total_docs': len(scans), 'total_pages': sum(s.get('page_count', 0) for s in scans),
        'total_cost': round(sum(s.get('cost', 0) for s in scans), 4),
    }


# =============================================================================
# Global State
# =============================================================================

scan_session: Optional[ScanSession] = None
book_ocr = BookOCRManager()


# =============================================================================
# Routes
# =============================================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/health')
def api_health():
    """Check if API key is configured."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("VITE_GEMINI_API_KEY")
    has_key = bool(api_key and len(api_key) > 10)
    return jsonify({'has_api_key': has_key, 'scan_model': SCAN_MODEL, 'book_model': BOOK_OCR_MODEL})


@app.route('/api/state')
def api_state():
    if not scan_session:
        return jsonify({'error': 'No session'}), 400
    return jsonify(scan_session.get_state())


@app.route('/api/capture', methods=['POST'])
def api_capture():
    if not scan_session:
        return jsonify({'error': 'No session'}), 400
    data = request.json
    if not data or 'image' not in data:
        return jsonify({'error': 'No image'}), 400
    img_data = data['image']
    if ',' in img_data:
        img_data = img_data.split(',', 1)[1]
    try:
        img_bytes = base64.b64decode(img_data)
    except Exception:
        return jsonify({'error': 'Invalid base64'}), 400
    with scan_session.lock:
        result = scan_session.capture_page(img_bytes, rotation=data.get('rotation', 0))
    return jsonify(result)


@app.route('/api/next-doc', methods=['POST'])
def api_next_doc():
    if not scan_session:
        return jsonify({'error': 'No session'}), 400
    with scan_session.lock:
        result = scan_session.next_document()
    return jsonify(result)


@app.route('/api/finish-scanning', methods=['POST'])
def api_finish_scanning():
    if not scan_session:
        return jsonify({'error': 'No session'}), 400
    with scan_session.lock:
        scan_session.finalize()
    total_pages = sum(d['page_count'] for d in scan_session.documents)
    return jsonify({
        'ok': True,
        'doc_count': len(scan_session.documents),
        'page_count': total_pages,
        'estimated_cost': est_scan_cost(total_pages),
    })


@app.route('/api/classify-session', methods=['POST'])
def api_classify_session():
    if not scan_session:
        return jsonify({'error': 'No session'}), 400
    unprocessed = [d for d in scan_session.documents if d['status'] in ('captured', 'queued')]
    for doc in unprocessed:
        threading.Thread(target=scan_session._process_doc, args=(doc,), daemon=True).start()
    return jsonify({'ok': True, 'processing': len(unprocessed)})


@app.route('/api/new-session', methods=['POST'])
def api_new_session():
    global scan_session
    scan_session = ScanSession(output_dir=app.config.get('OUTPUT_DIR', DEFAULT_OUTPUT_DIR))
    return jsonify({'ok': True, 'session_id': scan_session.session_id})


@app.route('/api/upload-pdf', methods=['POST'])
def api_upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'PDF only'}), 400
    filename = secure_filename(file.filename)
    file_path = os.path.join(book_ocr.upload_dir, f"{uuid.uuid4().hex[:8]}_{filename}")
    file.save(file_path)
    try:
        from pdf_pipeline import get_pdf_info
        page_count = get_pdf_info(file_path)['page_count']
    except Exception:
        page_count = 1
    # Create a Book OCR job (can be used for either classify or book OCR)
    job_id = book_ocr.create_job(filename, file_path, page_count)
    return jsonify({
        'ok': True, 'job_id': job_id, 'filename': filename, 'page_count': page_count,
        'classify_cost': est_scan_cost(page_count), 'book_cost': est_book_cost(page_count),
    })


@app.route('/api/start-classify', methods=['POST'])
def api_start_classify():
    """Classify an uploaded PDF (scan pipeline)."""
    data = request.json or {}
    job_id = data.get('job_id')
    job = book_ocr.get_job(job_id) if job_id else None
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    if not scan_session:
        return jsonify({'error': 'No session'}), 400

    # Create a doc entry and process through scan pipeline
    doc_id = f"doc_{len(scan_session.documents) + 1:03d}"
    doc = {
        'id': doc_id, 'page_count': job['page_count'], 'images': [],
        'pdf_path': job['file_path'], 'status': 'processing',
    }
    scan_session.documents.append(doc)
    scan_session._new_document()

    # Classify directly (PDF already exists)
    threading.Thread(target=scan_session._classify_doc, args=(doc, job['file_path']), daemon=True).start()

    # Remove from book OCR jobs since it's being classified instead
    with book_ocr.lock:
        book_ocr.jobs.pop(job_id, None)

    return jsonify({'ok': True, 'doc_id': doc_id})


@app.route('/api/start-book-ocr', methods=['POST'])
def api_start_book_ocr():
    data = request.json or {}
    job_id = data.get('job_id')
    if not job_id:
        return jsonify({'error': 'No job_id'}), 400
    ok, err = book_ocr.start(job_id)
    if not ok:
        return jsonify({'error': err}), 400
    return jsonify({'ok': True})


@app.route('/api/book-status/<job_id>')
def api_book_status(job_id):
    job = book_ocr.get_job(job_id)
    if not job:
        return jsonify({'error': 'Not found'}), 404
    return jsonify(job)


@app.route('/api/book-download/<job_id>')
def api_book_download(job_id):
    job = book_ocr.get_job(job_id)
    if not job or not job.get('output_path'):
        return jsonify({'error': 'Not found'}), 404
    return send_file(job['output_path'], as_attachment=True, download_name=os.path.basename(job['output_path']))


@app.route('/api/library')
def api_library():
    return jsonify(get_library(app.config.get('OUTPUT_DIR', DEFAULT_OUTPUT_DIR)))


@app.route('/api/file/<path:filepath>')
def api_file(filepath):
    output_dir = app.config.get('OUTPUT_DIR', DEFAULT_OUTPUT_DIR)
    full_path = os.path.join(output_dir, filepath)
    if not os.path.exists(full_path):
        return jsonify({'error': 'Not found'}), 404
    if not os.path.realpath(full_path).startswith(os.path.realpath(output_dir)):
        return jsonify({'error': 'Forbidden'}), 403
    return send_file(full_path)


@app.route('/api/update-setting', methods=['POST'])
def api_update_setting():
    if not scan_session:
        return jsonify({'error': 'No session'}), 400
    data = request.json or {}
    name = data.get('name')
    value = data.get('value')
    if name and hasattr(scan_session.config, name):
        setattr(scan_session.config, name, float(value))
        return jsonify({'ok': True})
    return jsonify({'error': 'Invalid setting'}), 400


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Scandoc 9000')
    parser.add_argument('-o', '--output', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('-p', '--port', type=int, default=int(os.environ.get('PORT', 5001)))
    args = parser.parse_args()

    output_dir = os.path.expanduser(args.output)
    os.makedirs(output_dir, exist_ok=True)
    app.config['OUTPUT_DIR'] = output_dir

    scan_session = ScanSession(output_dir=output_dir)

    print(f"\n  SCANDOC 9000")
    print(f"  ============")
    print(f"  Output:  {output_dir}")
    print(f"  Scan:    {SCAN_MODEL}")
    print(f"  Book:    {BOOK_OCR_MODEL}")
    print(f"  Port:    {args.port}")
    print(f"  Open:    http://localhost:{args.port}")
    print()

    app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)

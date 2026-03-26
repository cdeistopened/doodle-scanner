#!/usr/bin/env python3
"""
Image-based OCR pipeline for Scandoc 9000.

Extracts PDF pages as JPEGs, sends to Gemini as images in batches.
Bypasses RECITATION filter that blocks PDF-based OCR on published books.

Tested: 5-page batch → 14.2s, 12K chars, 2.8s/page, zero RECITATION blocks.
"""

import os
import json
import time
import base64
import math
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable, Dict, Any

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from google import genai
except ImportError:
    genai = None


# Config
DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"
DEFAULT_BATCH_SIZE = 5
DEFAULT_DPI = 150
MAX_OUTPUT_TOKENS = 32000
MAX_RETRIES = 3
RETRY_BACKOFF = 2
RATE_LIMIT_DELAY = 1.0

OCR_PROMPT = """You are an OCR assistant helping a user who owns this book convert their scanned pages to accessible text format.

Read all visible text from these scanned book pages IN ORDER and output as clean markdown.

Rules:
- Preserve paragraph breaks, headings, and document structure
- Use ## for chapter headings, ### for section headings
- Preserve italics and bold where visible
- Convert footnote markers to [^1], [^2] format with definitions at the end
- Skip page numbers and running headers/footers (repeated text at top/bottom of pages)
- If you see two columns, read left column first top-to-bottom, then right column
- Separate each page's content with a blank line
- Output clean, readable text — no commentary or explanations"""


def get_client(api_key: str = None):
    """Get Gemini client."""
    if genai is None:
        raise RuntimeError("google-genai not installed")
    if not api_key:
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


def extract_page_as_jpeg(doc, page_num: int, dpi: int = DEFAULT_DPI) -> bytes:
    """Extract a single page as JPEG bytes."""
    page = doc[page_num]
    pix = page.get_pixmap(dpi=dpi)
    return pix.tobytes('jpeg')


def ocr_batch(client, images: list, model: str = DEFAULT_MODEL, retries: int = MAX_RETRIES) -> str:
    """Send a batch of JPEG images to Gemini for OCR."""
    parts = []
    for img_bytes in images:
        parts.append({
            'inline_data': {
                'mime_type': 'image/jpeg',
                'data': base64.b64encode(img_bytes).decode()
            }
        })
    parts.append({'text': OCR_PROMPT})

    last_error = None
    for attempt in range(retries):
        try:
            if attempt > 0:
                backoff = RETRY_BACKOFF ** attempt
                print(f"    [Retry {attempt + 1}/{retries} after {backoff}s]")
                time.sleep(backoff)

            response = client.models.generate_content(
                model=model,
                contents=[{'role': 'user', 'parts': parts}],
                config={'max_output_tokens': MAX_OUTPUT_TOKENS, 'temperature': 0.1},
            )

            if response.text is None:
                raise RuntimeError("Empty response from Gemini")

            return response.text.strip()

        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            if 'rate' in error_str or 'quota' in error_str or '429' in error_str:
                wait = RETRY_BACKOFF ** (attempt + 2)
                print(f"    [Rate limited, waiting {wait}s]")
                time.sleep(wait)
                continue
            if '500' in error_str or '503' in error_str or 'timeout' in error_str:
                continue
            # Non-retryable
            raise

    raise RuntimeError(f"OCR failed after {retries} attempts: {last_error}")


def process_pdf_as_images(
    pdf_path: str,
    output_path: str = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    dpi: int = DEFAULT_DPI,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """
    Process a PDF by extracting pages as images and OCR'ing in batches.

    Args:
        pdf_path: Path to PDF file
        output_path: Where to save the final markdown
        batch_size: Pages per Gemini API call (default: 5)
        dpi: Resolution for page extraction (default: 150)
        progress_callback: Called with (completed_pages, total_pages, description)
        model: Gemini model to use

    Returns:
        Dict with output_path, total_time, total_chars, etc.
    """
    if fitz is None:
        raise RuntimeError("PyMuPDF not installed")

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    stem = Path(pdf_path).stem

    # Output setup
    if not output_path:
        output_dir = os.path.join(os.path.dirname(pdf_path), "output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{stem}_ocr.md")

    chunks_dir = Path(output_path).parent / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Calculate batches
    num_batches = math.ceil(total_pages / batch_size)
    client = get_client()

    all_content = []
    total_chars = 0
    start_time = time.time()

    print(f"\n  Image OCR Pipeline")
    print(f"  ==================")
    print(f"  File:    {Path(pdf_path).name}")
    print(f"  Pages:   {total_pages}")
    print(f"  Batches: {num_batches} ({batch_size} pages each)")
    print(f"  DPI:     {dpi}")
    print(f"  Model:   {model}")
    print()

    for batch_num in range(num_batches):
        batch_start = batch_num * batch_size
        batch_end = min(batch_start + batch_size, total_pages)
        page_range = f"Pages {batch_start + 1}-{batch_end}"
        chunk_num = batch_num + 1

        # Resume: check for existing chunk
        chunk_file = chunks_dir / f"{stem}_chunk{chunk_num:03d}.md"
        if chunk_file.exists() and chunk_file.stat().st_size > 100:
            print(f"  [{chunk_num}/{num_batches}] {page_range} — cached ({chunk_file.stat().st_size:,} bytes)")
            with open(chunk_file) as f:
                content = f.read()
            # Strip metadata comments
            lines = content.split('\n')
            while lines and lines[0].startswith('<!--'):
                lines.pop(0)
            content = '\n'.join(lines).strip()
            all_content.append(content)
            total_chars += len(content)
            if progress_callback:
                progress_callback(batch_end, total_pages, f"Chunk {chunk_num}/{num_batches}: {page_range} (cached)")
            continue

        if progress_callback:
            progress_callback(batch_start, total_pages, f"Chunk {chunk_num}/{num_batches}: {page_range}")

        # Extract pages as images
        images = []
        for pg in range(batch_start, batch_end):
            img_bytes = extract_page_as_jpeg(doc, pg, dpi=dpi)
            images.append(img_bytes)

        # OCR the batch
        try:
            batch_time_start = time.time()
            result = ocr_batch(client, images, model=model)
            batch_elapsed = time.time() - batch_time_start

            all_content.append(result)
            total_chars += len(result)

            # Save chunk
            with open(chunk_file, 'w') as f:
                f.write(f"<!-- Chunk {chunk_num}/{num_batches}: {page_range} -->\n")
                f.write(f"<!-- {batch_elapsed:.1f}s, {len(result):,} chars -->\n\n")
                f.write(result)

            # Save progress
            progress_data = {
                "total_chunks": num_batches,
                "completed": chunk_num,
                "total_pages": total_pages,
                "pages_done": batch_end,
                "total_chars": total_chars,
                "elapsed": round(time.time() - start_time, 1),
            }
            with open(chunks_dir / "progress.json", 'w') as pf:
                json.dump(progress_data, pf)

            print(f"  [{chunk_num}/{num_batches}] {page_range} — {len(result):,} chars in {batch_elapsed:.1f}s")

        except Exception as e:
            print(f"  [{chunk_num}/{num_batches}] {page_range} — ERROR: {e}")
            all_content.append(f"<!-- CHUNK {chunk_num} FAILED: {e} -->")

        # Rate limiting
        if batch_num < num_batches - 1:
            time.sleep(RATE_LIMIT_DELAY)

    doc.close()

    # Assemble final document
    total_time = time.time() - start_time
    header = f"# {stem}\n\n"
    header += f"*{total_pages} pages — OCR processed {datetime.now().strftime('%Y-%m-%d')} via image pipeline*\n\n"
    header += "---\n\n"

    final = header + "\n\n".join(c for c in all_content if c.strip())

    with open(output_path, 'w') as f:
        f.write(final)

    print(f"\n  Done! {total_chars:,} chars in {total_time:.1f}s")
    print(f"  Output: {output_path}")

    if progress_callback:
        progress_callback(total_pages, total_pages, "Complete")

    return {
        'output_path': output_path,
        'total_time': round(total_time, 1),
        'total_chars': total_chars,
        'total_pages': total_pages,
        'chunks_total': num_batches,
        'chunks_failed': sum(1 for c in all_content if 'FAILED' in c),
    }


# CLI usage
if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python image_ocr_pipeline.py <pdf_path> [output_path]")
        sys.exit(1)

    pdf = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None

    def progress(done, total, desc):
        pct = int(done / total * 100) if total else 0
        print(f"  [{pct}%] {desc}")

    result = process_pdf_as_images(pdf, output_path=out, progress_callback=progress)
    print(f"\n  Result: {json.dumps(result, indent=2)}")

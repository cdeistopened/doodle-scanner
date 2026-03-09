#!/usr/bin/env python3
"""
OCR for PageSnap sessions — combines captured images into a single PDF,
then runs the chunked pdf_pipeline for continuous, cross-page processing.

Replaces the old one-image-at-a-time approach with pdf-vision-style
chunked OCR: preflight analysis, good prompts, boundary smoothing,
retry logic.
"""

import os
import sys
import glob
import io
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


# Max dimension for captured images (long edge in pixels).
# 1500px preserves text sharpness while cutting file size ~5x vs 4K iPhone.
MAX_IMAGE_DIMENSION = 1500
JPEG_QUALITY = 85


def downsample_image(image_bytes: bytes) -> bytes:
    """Downsample an image so its long edge <= MAX_IMAGE_DIMENSION.

    Returns JPEG bytes. Skips resize if already small enough.
    """
    if Image is None:
        raise RuntimeError("Pillow not installed. Run: pip install Pillow")
    img = Image.open(io.BytesIO(image_bytes))
    w, h = img.size
    long_edge = max(w, h)

    if long_edge > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / long_edge
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    # Ensure RGB (no alpha channel for JPEG)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=JPEG_QUALITY)
    return buf.getvalue()


def images_to_pdf(image_paths: list, output_pdf_path: str) -> str:
    """Combine a list of JPEG images into a single PDF.

    Downsamples each image before embedding. Returns the PDF path.
    """
    if fitz is None:
        raise RuntimeError("PyMuPDF not installed. Run: pip install pymupdf")
    if Image is None:
        raise RuntimeError("Pillow not installed. Run: pip install Pillow")
    doc = fitz.open()

    for img_path in image_paths:
        with open(img_path, 'rb') as f:
            raw_bytes = f.read()

        # Downsample
        jpg_bytes = downsample_image(raw_bytes)

        # Get dimensions
        img = Image.open(io.BytesIO(jpg_bytes))
        w, h = img.size

        # Create page sized to image (in points: 1 point = 1/72 inch)
        # Use 150 DPI as reference so pages aren't tiny
        page_w = w * 72.0 / 150.0
        page_h = h * 72.0 / 150.0
        page = doc.new_page(width=page_w, height=page_h)

        # Insert image
        rect = fitz.Rect(0, 0, page_w, page_h)
        page.insert_image(rect, stream=jpg_bytes)

    doc.save(output_pdf_path)
    doc.close()
    return output_pdf_path


def _handle_error(message: str, exit_on_error: bool):
    """Handle errors differently for CLI vs. library usage."""
    if exit_on_error:
        print(f"Error: {message}")
        sys.exit(1)
    raise RuntimeError(message)


def process_session(session_path: str, output_path: str = None,
                    progress_callback=None, exit_on_error: bool = True) -> str:
    """Process all images in a session directory.

    1. Combines JPEGs into a single PDF (with downsampling)
    2. Runs pdf_pipeline's chunked OCR with analysis, good prompts,
       retry logic, and boundary smoothing
    """
    session_dir = Path(session_path)
    if not session_dir.exists():
        _handle_error(f"Session directory not found: {session_path}", exit_on_error)

    images = sorted(glob.glob(str(session_dir / "*.jpg")))
    if not images:
        _handle_error(f"No JPG images found in {session_path}", exit_on_error)

    total_pages = len(images)
    print(f"Found {total_pages} images in session")

    # Step 1: Combine images into PDF
    if progress_callback:
        progress_callback(0, total_pages, "combine")
    print(f"  Combining {total_pages} images into PDF (downsampling to {MAX_IMAGE_DIMENSION}px)...")

    pdf_path = str(session_dir / f"{session_dir.name}.pdf")
    images_to_pdf(images, pdf_path)

    pdf_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
    print(f"  PDF created: {pdf_size_mb:.1f} MB")

    # Step 2: Run chunked pipeline
    from pdf_pipeline import process_pdf

    if output_path is None:
        output_path = str(session_dir / f"{session_dir.name}_ocr.md")

    def pipeline_progress(completed, total, description):
        """Map pdf_pipeline progress to our callback format."""
        if progress_callback:
            progress_callback(completed, total, description)

    try:
        result = process_pdf(
            pdf_path,
            output_path=output_path,
            progress_callback=pipeline_progress,
        )
        print(f"\nOCR complete! {result.get('total_chars', 0):,} chars")
        print(f"Output saved to: {output_path}")

        if progress_callback:
            progress_callback(total_pages, total_pages, None)

        return str(output_path)

    except Exception as e:
        _handle_error(f"Pipeline failed: {e}", exit_on_error)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="OCR PageSnap sessions — combines images into PDF, "
                    "then runs chunked Gemini OCR pipeline"
    )
    parser.add_argument("session", help="Path to session directory containing JPG images")
    parser.add_argument("-o", "--output", help="Output markdown file path")

    args = parser.parse_args()
    process_session(args.session, args.output)


if __name__ == "__main__":
    main()

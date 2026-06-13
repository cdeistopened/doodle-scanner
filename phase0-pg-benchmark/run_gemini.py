#!/usr/bin/env python3
"""Gemini leg of the PG-57 OCR benchmark. Usage: run_gemini.py <model> <image> [<image>...]"""
import base64, json, os, sys, time, urllib.request

KEY = os.environ["GEMINI_API_KEY"]
MODEL = sys.argv[1]
PROMPT = open(os.path.join(os.path.dirname(__file__), "PROMPT.md")).read()

for path in sys.argv[2:]:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    body = {
        "contents": [{"parts": [
            {"text": PROMPT},
            {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
        ]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 16384},
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={KEY}"
    t0 = time.time()
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        resp = json.load(urllib.request.urlopen(req, timeout=300))
    except Exception as e:
        print(f"{path}: ERROR {e}", file=sys.stderr)
        continue
    dt = time.time() - t0
    cand = resp["candidates"][0]
    text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []))
    usage = resp.get("usageMetadata", {})
    stem = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(os.path.dirname(__file__), "results", f"gemini_{MODEL.replace('.', '-')}_{stem}.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(f"<!-- model={MODEL} image={path} secs={dt:.1f} usage={json.dumps(usage)} finish={cand.get('finishReason')} -->\n\n")
        f.write(text)
    print(f"{stem}: {dt:.1f}s in={usage.get('promptTokenCount')} out={usage.get('candidatesTokenCount')} finish={cand.get('finishReason')}")

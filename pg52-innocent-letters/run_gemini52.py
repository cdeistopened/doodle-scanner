#!/usr/bin/env python3
import base64, json, os, sys, time, urllib.request
KEY = os.environ["GEMINI_API_KEY"]
MODEL = "gemini-2.5-flash"
PROMPT = """You are transcribing a scanned page from Migne's Patrologia Graeca vol. 52 (19th-century Paris printing). The page may contain: a running head with column numbers, a Latin editorial introduction (monitum), a letter title, TWO COLUMNS of body text (polytonic Greek OR Latin — this volume alternates), and a footnote apparatus.

Rules:
1. DIPLOMATIC TRANSCRIPTION: reproduce EXACTLY what is printed, including apparent typos, odd accents, and inconsistencies. NEVER correct, normalize, modernize, or translate anything. If the printed text looks wrong, transcribe it wrong.
2. Transcribe the LEFT column completely first, then the RIGHT column. NEVER interleave lines across columns.
3. Preserve all polytonic diacritics exactly: breathings, accents, iota subscripts.
4. Output structure, using only the blocks present:
   [HEADER] running head and column numbers
   [MONITUM] the Latin editorial block if present (full width, before the letter)
   [TITLE] letter title + salutation lines if present
   [COL 1] left column body
   [COL 2] right column body
   [APPARATUS] footnote block(s), preserving markers; note with (under col 1) etc. where placed
5. Keep inline section markers (α´, β´ / 1., 2.), edition page-markers like [515], and superscript note references in place.
6. If a character or word is genuinely illegible, write ⟦?⟧ rather than guessing. Do NOT invent text.
7. Output ONLY the transcription. No commentary."""
for path in sys.argv[1:]:
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    body = {"contents": [{"parts": [{"text": PROMPT}, {"inline_data": {"mime_type": "image/jpeg", "data": b64}}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 32768, "thinkingConfig": {"thinkingBudget": 0}}}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={KEY}"
    t0 = time.time()
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        resp = json.load(urllib.request.urlopen(req, timeout=300))
    except Exception as e:
        print(f"{path}: ERROR {e}", file=sys.stderr); continue
    cand = resp["candidates"][0]
    text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []))
    usage = resp.get("usageMetadata", {})
    stem = os.path.splitext(os.path.basename(path))[0]
    os.makedirs("results", exist_ok=True)
    with open(f"results/gemini_{stem}.md", "w") as f:
        f.write(f"<!-- model={MODEL} image={path} secs={time.time()-t0:.1f} usage={json.dumps(usage)} finish={cand.get('finishReason')} -->\n\n{text}")
    print(f"{stem}: {time.time()-t0:.1f}s out={usage.get('candidatesTokenCount')} finish={cand.get('finishReason')}")

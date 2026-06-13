#!/usr/bin/env python3
"""A/B test: run a Gemini model on a page, score vs verified ground truth."""
import base64, json, os, sys, time, urllib.request, urllib.error

KEY = os.environ["GEMINI_API_KEY"]
PROMPT = """You are transcribing a scanned page from Migne's Patrologia Graeca vol. 52 (19th-century Paris printing). The page may contain: a running head with column numbers, a Latin editorial introduction (monitum), a letter title, TWO COLUMNS of body text (polytonic Greek OR Latin — this volume alternates), and a footnote apparatus.

Rules:
1. DIPLOMATIC TRANSCRIPTION: reproduce EXACTLY what is printed, including apparent typos, odd accents, and inconsistencies. NEVER correct, normalize, modernize, or translate anything. If the printed text looks wrong, transcribe it wrong.
2. Transcribe the LEFT column completely first, then the RIGHT column. NEVER interleave lines across columns.
3. Preserve all polytonic diacritics exactly: breathings, accents, iota subscripts.
4. Output structure, using only the blocks present:
   [HEADER] running head and column numbers
   [MONITUM] the Latin editorial block if present
   [TITLE] letter title + salutation lines if present
   [COL 1] left column body
   [COL 2] right column body
   [APPARATUS] footnote block(s), preserving markers
5. Keep inline section markers, edition page-markers like [517], and superscript note references in place.
6. If a character or word is genuinely illegible, write ⟦?⟧ rather than guessing. Do NOT invent text.
7. Output ONLY the transcription. No commentary."""

def call(model, path, thinking_off=True):
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    gen = {"temperature": 0, "maxOutputTokens": 32768}
    if thinking_off:
        gen["thinkingConfig"] = {"thinkingBudget": 0}
    body = {"contents": [{"parts": [{"text": PROMPT}, {"inline_data": {"mime_type": "image/jpeg", "data": b64}}]}],
            "generationConfig": gen}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={KEY}"
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        resp = json.load(urllib.request.urlopen(req, timeout=300))
    except urllib.error.HTTPError as e:
        if thinking_off:  # model may reject thinkingConfig — retry without
            return call(model, path, thinking_off=False)
        raise
    cand = resp["candidates"][0]
    text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []))
    return text, resp.get("usageMetadata", {}), time.time()-t0, cand.get("finishReason"), thinking_off

MODELS = sys.argv[1].split(",")
PAGES = sys.argv[2].split(",")
for model in MODELS:
    for p in PAGES:
        img = f"pages/idx{p}_full.jpg"
        try:
            text, usage, dt, fin, toff = call(model, img)
        except Exception as e:
            print(f"{model} idx{p}: ERROR {e}")
            continue
        out = f"results/ab_{model.replace('.','-')}_idx{p}.md"
        with open(out, "w") as f:
            f.write(f"<!-- model={model} image={img} secs={dt:.1f} thinkingOff={toff} usage={json.dumps(usage)} finish={fin} -->\n\n{text}")
        print(f"{model} idx{p}: {dt:.1f}s out={usage.get('candidatesTokenCount')} think={usage.get('thoughtsTokenCount',0)} finish={fin} thinkingOff={toff}")

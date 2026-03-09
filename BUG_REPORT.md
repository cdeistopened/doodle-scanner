# Doodle Scanner — Bug Report & Handoff

## Status: Fixed (all JS syntax errors resolved)

Last working commit: `61e6b4f`

---

## Bugs Found & Fixed This Session

### 1. Escaped apostrophe in Python template breaks JS string literal
- **Symptom:** `Uncaught SyntaxError: Unexpected identifier 're'` at camera:576
- **Root cause:** `You\'re on:` inside Python `'''` triple-quoted string. Python interprets `\'` as `'`, outputting unescaped `You're` into JS, which terminates the single-quoted JS string. The `re` after the apostrophe becomes an unexpected identifier.
- **Fix:** Changed text to `Currently on:` (avoid apostrophes entirely in JS strings inside Python templates)
- **Trap:** There were TWO instances in different code paths (HTTPS pre-check and catch block). First fix only caught one.

### 2. HTML comment `<!--` inside `<script>` tag breaks HTML parser
- **Symptom:** `Uncaught SyntaxError: Invalid regular expression: missing /` at camera:984
- **Root cause:** The regex `/<!-- page (\d+) -->/g` in `renderMarkdown()` contains `<!--`, which the HTML parser interprets as an HTML comment opener BEFORE the JS engine sees the script content. This breaks the regex literal.
- **Fix:** Replaced entire `renderMarkdown()` with plain text preview using `textContent` (no regexes in template at all).
- **If rendered markdown preview is needed later:** Use `new RegExp()` with hex escapes (`\x3c` for `<`, `\x3e` for `>`) instead of regex literals, OR load the markdown renderer from an external `.js` file (not inline in the template).

---

## Architecture Issue: Inline Templates

All HTML/CSS/JS is embedded as Python triple-quoted strings in `web_app.py` (~3800 lines). This creates a triple-escaping problem:

1. **Python** interprets `\'`, `\\`, `\n` etc. in the string
2. **Jinja2** (`render_template_string`) processes `{{ }}` and `{% %}`
3. **HTML parser** processes `<!-- -->`, `<script>`, entities
4. **JS engine** finally runs the code

Any character sequence that is meaningful to ANY of these layers can break. The safe path is:

- **No apostrophes** in JS strings (use template literals or different wording)
- **No HTML comments** in JS regex literals (use `new RegExp()` with hex escapes)
- **No Jinja syntax** in JS template literals (escape `{` as `{{ '{' }}`)
- **Prefer `textContent`** over `innerHTML` with string concatenation
- **Long-term:** Move JS to external `.js` files served as static assets

---

## Current Feature State

### Working
- Camera scanning with motion detection (browser getUserMedia)
- Photo upload fallback (file input with `capture="environment"`)
- PDF upload and OCR (chunked Gemini pipeline)
- Bounding box presets (Open Book, Paperback, Trade, Letter, Square)
- Image downsampling (1500px long edge, JPEG 85)
- Multi-model OCR comparison (gemini-2.5-flash-lite, 2.0-flash, 3-flash-preview, 2.5-pro)
- Library/history page
- Sensitivity slider with descriptive labels
- PDF compilation as intermediate step (Scan → PDF → OCR)
- OCR output download (.md)
- .docx export via pandoc

### Partially Working
- Markdown preview (currently plain text, was rendered but broke)
- Notion export (stub — needs NOTION_API_KEY)

### Not Yet Built
- Book/project model (convention carryover across chapters)
- Conventions editor (user-editable exceptions list)
- Doodle Reader integration (Convex backend, proper persistence)

---

## Key Files

| File | What |
|------|------|
| `web_app.py` | Flask app, all templates, all routes (~3800 lines) |
| `ocr_gemini.py` | Session processing: images → PDF → chunked OCR |
| `pdf_pipeline.py` | Core OCR pipeline: chunking, Gemini API, boundary smoothing |
| `railway.toml` | Railway config (nixpacks, pandoc apt package) |

## Config
- Default chunk size: 20 pages
- Default model: gemini-3-flash-preview
- Cooldown period: 1.0s
- Image downsampling: 1500px long edge, JPEG quality 85

# Spec — ScanDoc as a hosted SaaS (auth + billing), Gemini-powered

**Date:** 2026-06-19
**Status:** DRAFT — build spec, hand to a fresh session
**Owner:** Charlie
**Repo:** `cdeistopened/doodle-scanner` (`CIA/crux/scanner/`), live at https://scanner.doodlereader.com (Railway)

---

## One line

Turn the proven ScanDoc pipeline (phone-camera / PDF → Gemini Vision OCR → cleanup → markdown/docx) into a **full end-to-end product with accounts, auth (Clerk), and metered billing (Stripe)**, powered by **one Gemini API key** (the `google-genai` SDK, Gemini 3.5) with a thin orchestration loop you write yourself. The Google Antigravity SDK is a noted optional v2 upgrade, not a v1 dependency (see "The orchestrator").

This is **Tier 2 "Hosted Mode"** from the existing `VISION.md`, productized. Scope is the scanner only (not the reader, not the marketplace).

---

## What already exists (the proven assets — reuse, don't rebuild)

- **`web_app.py`** (~500 lines, Flask 3, 15 routes) — `ScanSession`, `BookOCRManager`, classify pipeline, `quick_scan`.
- **`image_ocr_pipeline.py`** (~300 lines) — extracts PDF pages → JPEGs → Gemini as images in batches of 5. **Bypasses the RECITATION filter.** Proven: 678-page *Graces of Interior Prayer*, 288K words, ~40 min, 132/136 chunks, ~$3–4.
- **`ocr_gemini.py`** — image downsample (1500px long edge = the OCR sweet spot), combine to PDF.
- **`pdf_pipeline.py`** (~900 lines) — older chunked OCR, kept as fallback.
- **`docx_export.py`** — markdown → Word.
- **`templates/index.html`** (~870 lines) — Scandoc 9000 SPA (camera viewfinder + motion-detection 4-state machine, PDF upload, job panel). Retro Xerox aesthetic, `DESIGN.md` + `mockups/`.
- **Model:** `gemini-3.1-flash-lite-preview` (~$0.001/doc classify). Two modes: **Classify & File** (12 categories) and **Book OCR** (long PDF → chunked markdown).

**What's NOT launch-ready today:**
- **No accounts, no auth.** BYOK (each user pastes their own Gemini key).
- **No billing.** No way to charge.
- **No persistent state.** Job history is browser `localStorage`; server job state is **in-memory and lost on every Railway redeploy.**
- **Outputs land on the server's local disk** (`~/scandoc-output/...`), not per-user cloud storage.

The gap is entirely the SaaS shell (identity, money, durable per-user storage, job queue) plus swapping the orchestrator to an agent. The OCR core is done and proven.

---

## The orchestrator — one Gemini key, direct (recommended for v1)

**You don't need an agent harness to ship this.** The pipeline is mostly deterministic (extract → batch → OCR → assemble); the only parts that want "agency" are failure recovery (retry the chunks that come back empty), classification, and a cleanup pass — and all three are a short retry/branch loop you write yourself around plain Gemini calls.

**v1 = the Gemini API directly.** `pip install google-genai`, **one `GEMINI_API_KEY`**, the `google.genai` client, **Gemini 3.5** (Flash for OCR + classify; Pro only if a job needs it). Multimodal vision is first-class — you pass page images straight in, which is exactly what OCR needs. The proven `image_ocr_pipeline.py` / `ocr_gemini.py` logic stays; you're calling the Gemini SDK under it and wrapping a retry/cleanup loop on top. One key, one provider, fewest moving parts, fastest to launch.

```python
from google import genai
client = genai.Client()  # reads GEMINI_API_KEY

# vision OCR — page images are first-class content
resp = client.models.generate_content(
    model="gemini-3.5-flash",          # confirm exact model id against the live API
    contents=[OCR_PROMPT, *page_images],
)
# orchestration = your own loop, not an agent:
#   batch pages -> ocr_batch() -> on empty/failed chunk, retry smaller / higher-DPI -> assemble -> optional cleanup pass
```

The parts that benefit from this thin loop are exactly the ones the scanner already wants: the **4-failed-chunk problem** (isolate failed pages, retry individually with smaller batch / different DPI, splice back in), **classification** into the 12 categories, and a **cleanup pass** over messy OCR. None of that needs a sandboxed agent — it needs a `for` loop and good prompts.

**Why not the Antigravity SDK for v1** (verified 2026-06-19): it's an autonomous-agent *harness* — a Python control plane driving a bundled **Go core over WebSockets** with sandboxed tool execution (`pip install google-antigravity`, part of Antigravity 2.0, I/O 2026). That's leverage when an agent must reason its own way through open-ended, multi-step work. OCR is a known pipeline with a couple of retry branches. The harness adds a Go process to host, a two-month-old dependency, and — per its current guide — **no documented way to pass images into the agent**, the one thing OCR needs. More parts, slower launch, no payoff yet.

**When Antigravity earns its place (v2, optional):** when you want the orchestrator to *reason* about novel failures — "this page is rotated / a dense table / too dark, change strategy" — and pick tools adaptively without you coding each branch. Real upgrade, not on the path to first dollar. Build v1 on the direct Gemini API; revisit the harness only if the loop starts needing judgment.

---

## Architecture — three approaches

**Approach A — Python monolith + Clerk + Stripe (minimal rewrite).**
Keep Flask (or port to FastAPI for async, which suits the OCR worker). Add Clerk via its backend SDK for session verification, Stripe Python SDK for billing, Postgres for state, a job queue + worker for OCR. UI stays the existing SPA.
*Effort: M · Risk: Med · Reuses: all current code. Cons: Clerk/Stripe DX is weaker outside Next.js; you hand-wire the auth UI.*

**Approach B — Next.js (Clerk + Stripe + UI) on Vercel + Python OCR worker on Railway (ideal).**
Identity, billing, and a rebuilt UI live in Next.js (Clerk's first-class home, cleanest Stripe wiring, where your other apps already live). The heavy OCR (direct Gemini API) runs as a **separate Python worker service** behind a job queue, with signed job tokens between the two and shared cloud storage (S3/R2) for inputs+outputs.
*Effort: L · Risk: Med · Reuses: the Python pipeline as the worker; UI is rebuilt. Cons: two services, a queue, cross-service auth.*

**Approach C — Pragmatic hosted MVP (fastest to first dollar).**
Existing Flask app + **Clerk** (hosted/embeddable components or JS, no UI rewrite) + **Stripe Checkout** (hosted page, simplest billing) + **Postgres** for users/jobs + a lightweight worker (RQ/Celery or asyncio) + the **direct-Gemini OCR loop**. Outputs to cloud storage. Ship, charge, learn.
*Effort: M · Risk: Low · Reuses: everything. Cons: not the prettiest long-term shape; Flask+Clerk is more manual than Next+Clerk.*

**RECOMMENDATION: start at C, with the data model and storage designed so a later move to B is a frontend swap, not a backend rewrite.** Charlie lives in Vercel/Next + has prior Clerk+Convex wiring (`crux/gui/` was scaffolded React 19 + Convex + Clerk), so B is the natural end state — but C gets a paying product out the door against a proven pipeline first. Don't rebuild the UI to take money.

---

## Auth — Clerk

**Tooling state (checked 2026-06-19):** Clerk **CLI not installed**; the old Clerk skills are **gone** (RLM CLAUDE.md flagged "NEED REINSTALL from clerk/skills GitHub"). Prior Clerk wiring exists to crib from in `crux/gui/` (React 19 + Convex + Clerk scaffold) and the doodle-reader work.

- **Next.js (B/C-with-Next):** `@clerk/nextjs`, middleware-based — the well-trodden path. Clerk's own CLI/quickstart scaffolds it.
- **Flask (A/C):** verify Clerk session JWTs server-side via Clerk's backend SDK / JWKS; gate every job route on a valid user. More manual but fine.
- **Install Clerk CLI** if going the Clerk-scaffolded route (`npm i -g @clerk/clerk-cli` or `npx`), and reinstall the Clerk skills for the build session.

---

## Billing — Stripe

**Tooling state:** Stripe **CLI installed** (v1.34.0). Good — local webhook testing (`stripe listen --forward-to ...`) works out of the box.

**The model matters because job cost is wildly variable:**
- Classify: ~$0.001 cost. Book OCR: ~$3–4 cost. A flat subscription gets eaten alive by heavy book-OCR users (the old VISION priced books at ~$5 = 20–40% margin, thin).
- **Recommended: credits / usage-based, not flat.** A subscription that includes a credit allowance + metered overage, with a **hard per-job cost ceiling** (quote the cost from `quick_scan`'s page count before the user commits — the app already does this estimate). Stripe usage-based billing or a prepaid-credits ledger.
- Stripe Checkout (hosted) for the first version; Billing/Customer Portal for plan management.

---

## Data model (replace in-memory + localStorage)

Minimum tables (Postgres):
- **users** (Clerk `user_id` as FK), **plan**, **credit_balance**
- **jobs** (`id`, `user_id`, `type` classify|book, `status`, `pages`, `cost_estimate`, `cost_actual`, `created_at`, durable so a redeploy doesn't lose them)
- **documents** (`job_id`, storage key, category, output formats)
- **usage_events** (for Stripe metering + the cost ceiling)

Inputs/outputs to **object storage** (S3/R2), not server local disk. Job state in Postgres, not memory.

---

## Honest flags

1. **One provider, one key.** v1 rides entirely on the Gemini API — simpler ops, simpler billing, first-class vision. The Antigravity SDK (two months old, agent image-input undocumented) is deliberately deferred to v2, so it cannot block launch.
2. **Unit economics on book OCR are thin.** Design billing as credits + per-job ceiling from day one, or heavy users lose you money.
3. **PII + privacy is real.** Users scan medical, financial, legal, government docs (the 12 categories). Hosting that means a privacy policy, encryption at rest, a retention policy, "we don't train on your documents," and clarity on Gemini API data handling. This is table stakes for charging money to OCR someone's medical records.
4. **Keep the orchestration thin.** The deterministic pipeline already did a 678-page book. The added value is failure recovery + classification + cleanup — a retry/branch loop around Gemini calls, not an agent framework. Ship the deterministic happy path; add judgment only where it pays.

---

## Build plan (phased)

1. **Shell:** Postgres + object storage + durable job model; move outputs off local disk. (Kills the redeploy-loses-jobs bug.)
2. **Auth:** Clerk in front of all job routes; per-user job history from the DB (replaces localStorage).
3. **Billing:** Stripe — credits + metered overage + per-job cost ceiling using the existing `quick_scan` estimate; Checkout + webhooks.
4. **Orchestrator:** call **Gemini 3.5 directly** (`google-genai`, one key) under the proven pipeline; a thin loop handles batching, the 4-failed-chunk retry, classification, and a cleanup pass. (Antigravity SDK is an optional v2 if this loop ever needs real autonomy.)
5. **Polish + launch:** mobile camera flow, output downloads (md/docx/pdf — docx export exists but isn't wired into the new UI), privacy policy, pricing page.

---

## Open questions

1. **Stack:** confirm C-first (Flask + Clerk + Stripe) vs jumping to B (Next.js front + Python worker). C is faster; B is where you'll end up.
2. **Pricing:** credits-per-book vs subscription-with-allowance vs both. Numbers?
3. **BYOK option retained?** Keep a free BYOK tier as a funnel (Tier 1) alongside the hosted paid tier?
4. **Domain/brand:** stays `scanner.doodlereader.com`, or its own domain for a standalone product?
5. **Data retention default** — delete after N days, or keep in the user's library?

---

## What I checked (2026-06-19)
- **Antigravity SDK** — real (Antigravity 2.0, I/O 2026; Python→Go-harness, auto tool-registration; agent image-input undocumented). **Decision: deferred to v2.** v1 uses the Gemini API directly (`google-genai`, Gemini 3.5, first-class image input, one key).
- **Stripe CLI** — installed, v1.34.0.
- **Clerk CLI** — not installed; Clerk skills absent (reinstall from `clerk/skills`); prior Clerk wiring in `crux/gui/`.
- **Scanner** — studied `crux/scanner/` (CLAUDE.md, VISION.md, web_app.py routes, image_ocr_pipeline.py, the 678-page proof).

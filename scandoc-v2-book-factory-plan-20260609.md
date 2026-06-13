# Scandoc v2 — The Book Factory

*Plan drafted 2026-06-09. Decisions: invite-only for real collaborators (Matt Horwitz is user #1) · Charlie's Gemini key for OCR + Claude account for harness work · full chain: scan → clean markdown → typeset → KDP-ready.*

## What v2 is

Scandoc v1 is a scanner with two modes. v2 is a **book factory with a scanner at the front**: a collaborator brings a rare book (camera, PDF, or Google Drive share) and leaves with a clean markdown edition, a typeset interior PDF, and a library entry that tracks the book through every stage. The UI is a playground — capture, progress, review. The harness (Claude + the existing skill chain) does the heavy lifting.

**Design correction from v1 — confirmed by the user, not just taste.** Matt's 2026-03-26 call feedback: he preferred the pre-Mar-24 UI ("simpler, more intuitive") over the Scandoc 9000 retro redesign — "something about the latest iteration made it harder to use" — and the corner brackets in the camera view confused him (unclear whether the page had to fit inside them). So: Terminal Cream design system (`crux/shared/tokens.css`), Editorial · Calm · Signal voice, no decorative chrome that creates questions. Motion detection "works well enough to be default" (his words) — keep it on by default, keep the status badge.

## User #1 spec (from the 2026-03-26 call + the Mar 2026 emails)

**The headline from the call: publisher use case beats librarian use case.** Matt wants scan → finalized interior manuscript, as fast as possible. His current workflow is scan → Markdown → Vellum formatting → near-finished interior. His near-term corpus: small Treasury of Christian Doctrine volumes (34–80 pages each) to scan and republish. He's not technical, mainly uses Gemini with light Claude — the tool has to carry him.

Call requirements:

| Feedback (2026-03-26 call) | v2 feature |
|---|---|
| **Page-by-page review: preview the actual page image against the OCR** | Core review screen (Phase 3): pager with scan image ↔ OCR markdown side-by-side, accept/fix per page. Not an open question — this is the centerpiece. |
| Scan → finalized interior ASAP | Full chain default; small-volume fast path (34–80pp books should go end-to-end in one sitting) |
| Exports: PDF, Word, Markdown + email delivery | Export trio per book + send-by-email. **The formatting step is our design-to-book skill, replacing Vellum** — Matt gets a finished interior PDF directly. The intermediates are first-class exports in their own right: clean markdown (canonical) and clean DOCX, both with real structure — footnotes as footnotes, proper header hierarchy, no page-number/running-header residue — so they survive import into Vellum or anything else if he prefers his own path. |
| Simpler UI than the retro redesign | Terminal Cream rebuild, above |
| Motion detection as default | Default on |

Email-derived requirements (corpus curation):

| What he does today (email + Drive) | v2 feature |
|---|---|
| Lists which titles aren't on Internet Archive / Google Books | **Provenance field** per book (own scan / seminary library / India digital library / WorldCat ref) |
| Flags "ODC" → "OCD" title-page typo | **Metadata review step**: title, author, post-nominals, edition — extracted by the harness, confirmed by a human |
| Explains Hull's *Studies* publication order | **Series grouping + ordering** in the library |
| Notes *Theosophy* is "missing at least the title page and verso" | **Completeness check**: harness flags missing title page, gaps in pagination |
| Shares scans via Google Drive links | **Drive-link ingestion** alongside camera + PDF upload |
| Tracks "one en route from another institution" | **Book status** before any scan exists: wanted → sourcing → scanning → … |

## Architecture

```
CAPTURE (playground UI)            HARNESS (skills, server-side)        OUTPUTS
camera + motion detect ─┐
PDF upload ─────────────┼─→ pages → OCR (Gemini Flash-Lite,      → clean markdown
Google Drive link ──────┘           Charlie's key, per-user cap)  → manifest.json (metadata,
                                  → cleanup + QA loop                provenance, completeness)
                                    (markitright rules, Claude)   → typeset interior PDF
                                  → metadata + completeness         (design-to-book template)
                                  → typeset (design-to-book)      → KDP handoff bundle
```

**Storage** (per the 2026-06-09 architecture discussion):
- **Bodies → R2**: `books/{slug}/pages/*.jpg`, `clean.md`, `manifest.json`, `interior.pdf`. Markdown stays markdown; export-as-folder is one click.
- **Ledger + index → Postgres** (Railway) — the Crux ledger schema ported: `discovered → captured → ocrd → cleaned → reviewed → typeset → published` as append-only events. Caches derived, never canonical.
- **No git as runtime store** (see 50 GB tmp_pack incident, same day as this plan).

**Auth/runtime:**
- Invite-only: magic-link per collaborator. No public signup in v2.
- OCR: server-side Gemini key (Charlie's), per-user monthly page cap. Matt-scale cost is cents per book.
- Harness work (cleanup QA, metadata extraction, completeness): Claude via Agent SDK with Charlie's Max OAuth held server-side (the plan-credits path). Per-user Claude OAuth is the later public-tier upgrade, not v2.

**Stack:** keep the Python backend (PyMuPDF, `image_ocr_pipeline.py`, design-to-book build scripts are all Python/CLI); new front end replaces the single-file template — server-rendered + htmx or a small React app, Terminal Cream tokens. Railway deploy continues on `doodle-scanner` repo.

## Build phases

1. **Foundation** — accounts (magic link), R2 + Postgres ledger, book + manifest data model. Port existing scan/OCR flows onto it unchanged.
2. **Ingest** — camera mode and PDF upload write into a *book*, not a loose output folder. Add Google Drive link ingestion. Book statuses including pre-scan ("sourcing").
3. **Review** — the centerpiece, per Matt: page-by-page pager, actual scan image ↔ OCR markdown side-by-side, accept/fix per page, completeness flags surfaced inline (missing title page = a red page slot, not a buried warning). Claude QA pass (markitright ruleset) and metadata extraction run before review so the human is confirming, not correcting.
4. **Typeset + export** — design-to-book is the formatting engine (CLM house template first), deterministic rebuild, interior PDF per book. Export trio: interior PDF / clean structured DOCX / clean markdown — footnotes, header hierarchy, and typography intact at every tier — plus email delivery.
5. **Library** — series grouping + ordering (Hull's *Studies* as the test case), provenance display, export-as-folder, KDP handoff bundle (interior PDF + kdp-cover-wrap inputs).

Each phase ships usable to Matt. Phase 1+2 alone replaces the email+Drive workflow; Phase 3 is where the harness earns its keep and where his page-by-page ask lands; Phase 4–5 completes the factory. Benchmark book: one 34–80pp Treasury volume, scan to emailed interior in a single sitting.

## Open questions (don't block Phase 1)

- Where does the Agent SDK host live — same Railway service or a separate worker?
- Does the CLM Publishing catalog (Cross & Plough, Rerum Novarum) migrate into the library as seed content?
- v2.1: graft Conspire Canvas Studio (PDF.js + comment layer) onto the review screen for collaborator annotations?

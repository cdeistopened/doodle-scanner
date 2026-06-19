# WORKBENCH — ScanDoc / Scanner

## Now
- **Hackathon Sat 2026-06-13, Shack15 SF** — demo arc rehearsed end-to-end (2 acts: pure-Fable strip-reading / Gemini-bulk economics + ensemble gate). BLOCKED on: application/approval status — ping Bence.
- Innocent dossier at cols 529–538 (letter 1 ✓, letter 2 started, Innocent's reply ✓, Innocent-to-clergy cut mid-sentence). One more run (~cols 539–545) closes the dossier → then design-to-book mini edition for the cousin.

## Blocked
- Bence: hackathon application status + what he actually meant by "crowdsourced skill learning" (verified: NO such Anthropic feature exists — our learning-loop convention is the implementation).
- Cousin's grading of the emendation log (first external test of the verify layer).

## Recent Decisions
- 2026-06-19 — **Hosted-SaaS launch track spec'd** (separate from the Migne/OCR-quality track): productize ScanDoc into a Clerk-auth + Stripe-billing hosted app ("Tier 2 Hosted Mode"). **Spec: `scandoc-hosted-launch-spec-20260619.md`** — hand to a fresh build session. **Decision: v1 OCR uses the Gemini API directly (`google-genai`, Gemini 3.5, one key), NOT the Antigravity SDK** (deferred to v2 — the pipeline is deterministic + a thin retry loop, no agent harness needed). Real gap = SaaS shell (Postgres + object storage + job queue; today job state is in-memory and dies on Railway redeploy), not OCR. Stripe CLI installed; Clerk CLI not (prior Clerk wiring in `crux/gui/`). Billing = credits + per-job cost ceiling (book OCR ~$3–4/job kills flat-sub margins). [[project-scandoc-hosted-launch]]
- 2026-06-12 — **Engine: gemini-3-flash-preview** replaces 2.5-flash for bulk OCR (5× fewer errors on hard Greek; 3.1-flash-lite NOT better). thinkingBudget:0 fine, no runaways in 9 calls.
- 2026-06-12 — **Ensemble gate is mandatory**: second engine family + programmatic diff + eye adjudication. 132 disputes → 38 corrections to "verified" text; ~2–3 residual errors/page survive ANY single verifier (Opus included). Every [sic] needs 2 independent engine families (two spurious flags killed: ἐσμεν, μετῳκίζετο; one over-correction restored: exagitatuin).
- 2026-06-12 — **Verifier recipe locked**: pre-cut strip grid by script + diff worklist + Sonnet (≤14 reads) = 73K tok/page vs 126K Opus (−42%); Opus/Fable only at escalations (~1 per 5 pages). Cost is loop length, not model tier.
- 2026-06-12 — **Fable-vision pilot**: matched 3-flash raw accuracy blind (75 vs 79 errwords), caught 2 errors that survived the 2-engine gate. Demo Act 1.
- 2026-06-12 — **Contribution protocol designed**: `corpus-factory-protocol-20260612.md` (PR = proof-of-work bundle, CI reviewer agent, pgocr CLI, GitHub-yes-with-caveats, engine-agnostic).
- 2026-06-11 — v3 plan locked (`scandoc-v3-unscanned-literature-plan-20260611.md`): v2 book factory stays the spine; skill commons + registry + 2-tier distribution wrap it.
- 2026-06-11 — PG 57 benchmark (`phase0-pg-benchmark/RESULTS.md`): resolution is the master variable; agency beats model size; Haiku never on vision; Sonnet honest-refusal = legibility gate.

## Learnings
- 20-entry evidence-cited ledger at `pg52-innocent-letters/run-learnings.jsonl` — the skill-commons seed corpus. Read it before touching the pipeline.
- Canonical-text accuracy does NOT transfer: ~1 err/page on famous homilies vs ~10–35/page on the letters. Budget verification by text fame.
- OCR errors mimic 19th-c compositor typo classes — only image adjudication separates them.

## Next Session
1. Run cols 539–545 (finish dossier) with: 3-flash engine + ensemble gate inline + structured verdicts.jsonl (enables script-patching instead of agent-patching).
2. ~12 open Fable-vs-truth divergences: `pg52-innocent-letters/results/open-adjudications-idx153.md`.
3. design-to-book mini edition of the dossier.
4. Notion lab is live: page 37dbe9d0-0a42-81ad-a3c6-c960c4a45069 (myStrong, REST-only). Keep in sync if text changes.
5. Repo hygiene: pg52-innocent-letters/ + phase0-pg-benchmark/ = ~143MB untracked in the doodle-scanner repo — gitignore pages/ + *.pdf + native PNGs before any commit (images never enter git).

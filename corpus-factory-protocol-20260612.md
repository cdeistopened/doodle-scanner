# The Corpus Factory — Contribution Protocol Design

*Drafted 2026-06-12, the night before the Fable 5 hackathon. Extends `scandoc-v3-unscanned-literature-plan-20260611.md` with the crowdsourced-contribution layer, designed from two days of validated pipeline runs (PG 57 benchmark + PG 52 Innocent dossier, 10 pages, 17 ledger learnings).*

## Thesis

Anyone with an AI account and a manuscript contributes chunks of the canon. The human's only job is providing the source; their agent runs the pipeline. Contributions arrive as **pull requests carrying proof of work**; a **reviewer agent** enforces quality before anything enters the canon; and because every PR includes the contributor's method log, **the skills improve with every contribution** — including when someone breaks the orthodoxy with a cheaper engine or a better trick.

## What exists vs. what we invent

**Verified (research, 2026-06-11):** Anthropic has NO self-improving-skill protocol. The Agent Skills spec is static; plugin marketplaces are git-repo-backed with manual update pulls; nothing writes back. The "crowdsourced continuous learning" layer is therefore a *convention we define* — which is an advantage: it's engine-agnostic and we own the standard.

**Already validated in-house (the convention's seed):** `run-learnings.jsonl` (17 entries, all evidence-cited), patch-the-gap skill edits, 3-way verification taxonomy, the cross-model ensemble gate. The protocol below is these practices, formalized for strangers.

## Is GitHub the right tool? Yes — with three caveats

1. **Images never enter git.** Page scans and strips live in object storage (R2) or reference Internet Archive originals; manifests carry URLs + SHA-256 hashes. Git holds text: transcripts, verdicts, ledgers, skills. (Hard-won rule: the 50 GB tmp_pack incident; also IA already hosts most source scans.)
2. **Non-technical contributors never see git.** The front-end website wraps it: upload/point at a manuscript → your agent runs → "Submit" opens the PR via the app's GitHub App credentials. Wikipedia UX, git backend. Technical contributors can skip the site and PR directly.
3. **The reviewer agent runs in CI** (GitHub Action → Agent SDK), so review is automatic, logged, and itself version-controlled.

Everything else GitHub gives for free: diff review, attribution/blame (contributor credit = commit history), versioned skills doubling as a Claude Code plugin marketplace, issues as the most-wanted registry's work queue, stars/forks as distribution.

## Repo anatomy

```
corpus/                      skills/                    harness/
  pg/052/                      pg-ocr/                    pgocr (CLI, python)
    manifest.json                segment/  transcribe/    reviewer/
    cols-529-538/                verify/   adjudicate/      reviewer-agent.md
      verified/*.md              ensemble/ stitch/          spot-audit.py
      disputes/*.jsonl         registry/                  templates/
      verdicts/*.jsonl           most-wanted.md             CONTRIBUTING.md
      run-learnings.jsonl                                   pr-template.md
      stats.json
```

## The pgocr CLI — programmatize everything that isn't judgment

Lesson of runs 1→2: cost lives in agent loop length, not model intelligence. Every stage that is deterministic becomes Python; agents survive only where vision + judgment are irreducible.

| Stage | Run-1 reality | Target | Agent? |
|---|---|---|---|
| `pgocr prep` | ad-hoc Bash | script: extract, classify leaf (lang/cols), full+col jpgs, strip grid, app strip | no |
| `pgocr transcribe` | ad-hoc runner | script: engine matrix (default gemini-3-flash-preview), runaway guard (out-tokens > 2× sibling median → retry), usage ledger | no |
| `pgocr diff` | written mid-run | script: dual-pass diff → disputes.jsonl (structured, not md) | no |
| `pgocr verify` | agents cropping by hand | script feeds pre-cut strips + dispute worklist to ONE verifier agent per page (Sonnet-tier, ≤14 reads), structured verdict output | **yes — vision** |
| `pgocr ensemble` | invented on day 2 | script: second-engine pass + diff → dispute list → adjudicator agents (batched by region), verdicts.jsonl | **yes — vision judgment** |
| `pgocr patch` | an agent doing string edits | script: apply verdicts.jsonl mechanically (the patch agent was rote work — structured verdicts make it a 50-line script) | no |
| `pgocr stitch` | agent | script for assembly + ONE agent pass for section alignment of parallel editions | mostly no |
| `pgocr export` | agents (Notion/files) | scripts per target (md, Notion, design-to-book) | no |
| escalations | orchestrator | stays with the orchestrating model — rare by design (1–2 per 5 pages) | **yes — the judgment core** |

Agent surface shrinks to: per-page verification reads, dispute adjudication, escalations, and review. Everything else is reproducible by anyone, with any wallet.

## The contribution protocol (what a PR must contain)

A contribution = one **chunk** (a column range / letter / chapter) as a PR:

1. `manifest.json` — source (IA identifier or scan hashes), pages, language map, engines + versions used, token/cost ledger.
2. `verified/*.md` — diplomatic transcripts, `[sic]` flags inline.
3. `disputes.jsonl` + `verdicts.jsonl` — the full dispute trail: every dual-pass divergence, every adjudication with crop evidence references. **This is the proof of work.** A transcript without its dispute trail is unreviewable and gets auto-rejected.
4. `run-learnings.jsonl` — what the run taught (may be empty; rarely is).
5. `stats.json` — pages, error counts by class, adjudication coverage (must be 100% of substantive disputes), escalation count, residuals from the ensemble gate.

The PR template makes the proof-of-work bundle the default output of `pgocr` — contributors following the harness produce it for free.

## The reviewer agent (CI)

Runs on every PR, in escalating cost order — cheap structural checks kill bad PRs before vision spends a token:

1. **Structural validation (script, no tokens):** manifest schema, hash checks, dispute-coverage = 100%, stats arithmetic consistent with files, no canon files touched outside the claimed chunk.
2. **Vision spot-audit (agent):** sample N tokens per page (weighted toward diacritics, apparatus sigla, proper nouns — the known failure classes) and verify against the source scan crops. Pass bar from our measured baseline: sampled error rate must not exceed the residual we ship ourselves (~1 per 300 words post-ensemble).
3. **Canonical-bias probes (agent):** check `[sic]`-flagged and apparatus-variant readings specifically — the places where engines silently normalize. (ἐδίδασκεν-class detection.)
4. **Methodology diff (script + agent):** if `manifest.json` declares a non-standard method (different engine, skipped stage, new trick) AND the audit passes at equal-or-better stats per token — flag for a **skill amendment PR**: the contributor's method becomes a documented option, credited. This is the "really looking to break it" loop: beating the baseline is rewarded with influence over the canon's tooling.
5. **Verdict:** approve / request-changes with specific failed samples / reject. Human maintainer holds merge for v1; auto-merge on green is the later graduation.

## Quality thresholds (v1, from measured data)

- Ensemble gate mandatory: two engine families minimum; all substantive disputes adjudicated by eye with crop evidence.
- Spot-audit pass: ≥ 99.7% sampled-token accuracy (our post-ensemble level).
- `[sic]` discipline: every flag carries a proposed emendation + reason; every "fix" of apparent print error must cite the adjudication (over-correction is a rejection class — the exagitatuin lesson).
- No silent engine fallbacks: every model call logged in the manifest.

## Engine-agnostic by design

The standard judges **outputs and proof of work, never engines**. Claude account, Gemini key, open-source model — all welcome; the thresholds don't move. This widens the contributor pool to "anyone with any AI account" (the actual installed base) and makes the leaderboard meaningful: efficiency innovations are comparable because the bundle format is fixed.

## What stays beyond v1

- Auto-merge on green CI; reviewer-agent council (multiple models voting) for contested PRs.
- Registry pledges wired to PR completion ("claimed by @user, in progress").
- The phone-capture tier feeding `pgocr prep` directly.
- Federation: per-corpus repos (PG, technical manuals, …) sharing the harness + reviewer as a template repo.

## Open questions

- Reviewer-agent budget: who pays CI inference? (Hackathon answer: Anthropic credits. Real answer: maintainer-funded per-corpus, or contributor runs review locally and CI re-verifies a sample.)
- License/attribution standard for the canon (PD source + CC0 transcripts + contributor credit file?).
- When does a methodology PR change the *default* vs. add an *option*? (Proposal: 3 independent contributors adopting it.)

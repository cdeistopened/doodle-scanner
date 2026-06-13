# ScanDoc v3 — Infrastructure for the World's Unscanned Literature

*Planned 2026-06-11. Companion to `scandoc-v2-book-factory-plan-20260609.md` — v2 is the spine, not replaced. Immediate forcing function: Anthropic Fable 5 hackathon, Saturday June 13, Shack15 SF.*

---

## Thesis

Billions of cameras, millions of books that exist nowhere digital. ScanDoc v3 frames the v2 book factory as the first node of a larger system: **a capture tool designed to be used with Claude, a skill commons that learns from every run, and a registry that tells the world what's worth scanning.** Contributors bring books; their agents bring skill improvements. The corpus grows on both axes.

The pitch line: *every scan makes the next scan better.* The book goes into the corpus; what the run learned goes into the skills.

## Decisions locked (2026-06-11 interview)

| Question | Decision |
|---|---|
| v2 vs v3 | **v2 is the spine.** Matt's book factory remains the build priority; v3 is the distribution + learning layer wrapped around it. |
| OCR engine story | **Test Claude vision first** (Phase 0 benchmark, before Saturday). Architecture stays engine-agnostic: Claude orchestrates, vision OCR is a routed tool call (Fable/Opus or Gemini Flash-Lite per document). |
| Wedge corpus | **Rare religious reprints**, public-domain-first (pre-1930). Matt's Treasury volumes + CLM Publishing catalog + Hildegard. "Not on Internet Archive / Google Books" is the registry's admission test. |
| Bounties | **Registry + pledge field.** Public most-wanted list; entries can carry pledges ("seeking, will pay $X"), settled person-to-person. Platform never touches money. Escrowed marketplace = v4+ question, contingent on contributor traction. |
| Distribution | **Two tiers.** (1) ScanDoc Claude Code plugin for contributors — runs on their own plan, feeds the skill commons. (2) Hosted web app (v2 as planned, Agent SDK credits) for non-technical users like Matt. |
| Hackathon | **Real: Sat June 13.** Demo spec below. Apply/confirm attendance immediately. |

## Ground truth from research (2026-06-11)

What the plan can and cannot assume about the Anthropic stack:

- **No crowdsourced-skill-learning protocol exists.** The Agent Skills spec treats skills as static instruction sets — no run→learn→amend mechanism, no write-back, no shared-learnings field. The loop must be hand-built as a convention (design below). This is a feature for the hackathon: we're building something the platform doesn't have yet.
- **Plugin marketplaces are the distribution rail.** Git-repo-backed, versioned via `marketplace.json`, updated by `/plugin marketplace update` (manual pull, no auto-update). A GitHub repo IS a marketplace — Charlie's instinct confirmed.
- **Consumer subscription OAuth in third-party apps is forbidden** (Consumer ToS). A standalone phone app cannot draw on the user's Claude subscription. The legal shapes: Claude Code plugin (user's own session does the work) and Agent SDK apps drawing on plan credits ($20 Pro / $100 Max5x / $200 Max20x per month — the path v2 already uses for the hosted harness).
- **MCP Apps + Claude Connectors Directory** exist for claude.ai-embedded interfaces — a later distribution channel for the registry/capture UI, not a v3 dependency.

## Architecture — three layers

```
1. CAPTURE                      2. SKILL COMMONS                 3. CORPUS + REGISTRY
phone browser camera            github.com/<org>/scandoc-skills  R2: books/{slug}/ bodies
(v2 web app, motion detect)     = plugin marketplace             Postgres: ledger + index
PDF / Drive ingestion           OCR · element-mapping ·          (v2 storage plan unchanged)
Claude Code plugin bridge       cleanup · typeset skills         +
(contributor tier)              + LEARNING LOOP (below)          MOST-WANTED REGISTRY
                                                                 entries · provenance ·
                                                                 pledges · credit
```

Layers 1 and 3 are v2's capture and storage with new names. Layer 2 — the skill commons with a learning loop — is the genuinely new build.

### The skill commons

One GitHub repo, structured as a Claude Code plugin marketplace:

- **The skills**: the existing chain, extracted and packaged — page OCR (vision), element mapping (images/charts/tables identified hybrid: programmatic PDF tools + vision), element rendering (tables→markdown, charts→clean SVG/compressed PNG at max-resolution-min-filesize), cleanup/QA (markitright ruleset), metadata + completeness, typeset (design-to-book). Most of this already exists in `markitright`, `image_ocr_pipeline.py`, `design-to-book`, and the Hildegard/Hébert pipelines — v3 packages it, doesn't rewrite it.
- **Engine-agnostic interface**: each skill states *what* (e.g., "transcribe this page image to structured markdown, preserving footnotes") and routes *how* (Fable vision / Opus vision / Gemini Flash-Lite BYOK) by document difficulty and what keys the user has. Phase 0 benchmark calibrates the routing table.

### The learning loop (hand-built protocol)

Since no platform primitive exists, define the convention:

1. **Every run emits a learnings ledger.** The orchestrating agent appends structured entries to `run-learnings.jsonl` as it works: workaround discovered, failure mode hit, ruleset gap (e.g., "blackletter ligatures misread as 'ck' — added disambiguation note"), routing surprise ("Gemini refused page 40, Fable succeeded").
2. **Wrap-up step drafts a skill amendment.** At job end, the agent diffs its learnings against the current skill text and, where a learning generalizes, opens a **GitHub PR against the commons repo** — patch-the-gap edits only (one rule, in the section it affects), with the run evidence cited in the PR body.
3. **Human merge gate.** Charlie (later: maintainers) reviews and merges. One session's pattern is a data point, not doctrine — the merge gate enforces that.
4. **Contributors pull updates** via `/plugin marketplace update`; the hosted app redeploys from the same repo.

This is just GitHub + PRs + a disciplined agent prompt — which is exactly why it works today. The moat is the convention and the corpus of merged learnings, not novel infrastructure.

### The registry

A public page (extend the v2 library): works sought, with provenance notes ("not on IA; one copy at seminary library X"), status (`wanted → sourcing → scanning → published`, already in v2's data model), optional pledge text, and contributor credit on completion. Seeded from Matt's "not on Internet Archive" lists and the CLM catalog. Public-domain-first policy stated on the page: pre-1930 (US) works only for the open corpus; in-copyright scanning stays private to the scanner.

The gold-rush story ("old bookstores as gold mines — anyone holding a rare volume can earn by scanning it") is the *narrative* for pledges. The *machinery* (escrow, scan verification, disputes) is explicitly deferred to v4+, gated on ≥10 active contributors.

## Two-tier distribution

| Tier | Who | Runs on | What they get |
|---|---|---|---|
| **Plugin** (`scandoc` Claude Code plugin) | Technical contributors | Their own Claude Code + plan; BYOK Gemini optional | Full pipeline locally; phone capture via the web app's camera page pointed at their session (capture bridge = MCP server or watched folder); their runs propose skill PRs |
| **Hosted app** (v2) | Invited non-technical users (Matt) | Railway + Agent SDK on Charlie's plan credits | The v2 book factory exactly as planned — capture, review pager, typeset, exports |

Both tiers consume the same skill commons; the hosted harness is just contributor #0.

---

## Hackathon — Saturday June 13, Shack15 SF

**Frame fit:** "The longer and more complex the task, the better" — ScanDoc's demo IS one long-horizon task: physical rare book → finished typeset edition, agent-orchestrated end to end, with the system amending its own skills along the way. The mission framing ("compress decades of progress") maps cleanly: the world's unscanned literature, ingested by the cameras already in everyone's pocket.

**Immediately (today, June 11):** confirm application/approval — attendance is apply-and-approve, rolling review. Ping Bence: (a) application status, (b) what he actually meant by the crowdsourced-skill-learning protocol (does Anthropic have something unreleased, or was he describing the PR convention?).

**Phase 0 (Friday June 12) — the Fable vision benchmark.** Decides the demo story:
- Test set: ~10 hardest pages from the existing corpus — Hildegard-grade scans, dense tables, charts/figures, footnote-heavy Treasury pages, one blackletter sample.
- Fable 5 vision vs Opus vision vs Gemini Flash-Lite: accuracy (spot-verified), structure preservation (footnotes, hierarchy), refusal/RECITATION behavior, cost per page.
- Output: a routing table in the skill commons + the demo claim ("Fable handles X natively" or "Claude routes to the right engine per page" — honest either way).
- Also Friday: stage the repo skeleton (commons structure, plugin manifest, registry page stub) so Saturday is assembly, not setup. "The idea you've been sitting on" language confirms prior work is expected.

**Saturday build scope (one day, must be demo-able):**
1. **The run**: bring a physical Treasury/CLM volume; phone-camera capture (existing v1 scanner UI) → Fable 5 in Claude Code orchestrates: OCR → element mapping → cleanup → metadata/completeness → design-to-book typeset → interior PDF. Target: a 34–80pp book end to end during the day (v2's benchmark book, on stage).
2. **The loop, live**: the run's learnings ledger produces a real PR against the skill commons during the demo — "the system you just watched got better at its job, and here's the diff."
3. **The registry**: most-wanted page with seeded entries + pledges; the demo book's entry flips `wanted → published` at the end.

Demo arc (3 beats): *here's a book that exists nowhere digital → watch one agent take it from camera to typeset edition → and here's the PR it just filed to make the next book faster.* Close on the registry: "ten thousand contributors, each with a phone and a Claude subscription."

**Explicitly out of demo scope:** accounts/magic links, R2/Postgres (local files fine for the demo), email delivery, Drive ingestion, review pager UI (show the markdown diff in terminal instead).

## Build phases (post-hackathon, reconciled with v2)

The v2 phases (Foundation → Ingest → Review → Typeset/Export → Library) stand unchanged for the hosted tier. v3 adds, in order:

1. **Commons extraction** — pull the pipeline skills out of the monorepo into the public `scandoc-skills` repo, plugin-marketplace structure, engine routing table from Phase 0. (Much falls out of hackathon prep.)
2. **Learning loop hardening** — the run-ledger → PR convention as a reusable skill itself; merge-gate checklist (patch-the-gap discipline, evidence required, no doctrine from single runs).
3. **Registry v1** — public most-wanted page + pledge field, fed by the v2 library's status model; seed from Matt's lists + CLM catalog; PD-first policy page.
4. **Contributor tier** — plugin install docs, capture bridge polish, first 3 external contributors invited (Matt's circle of fellow reprint publishers is the natural pool).
5. **v2 Phases 3–5 continue in parallel** — Matt's review pager remains the centerpiece commitment; nothing here preempts it.

## Open questions

- Bence: application status for Saturday; the real story on crowdsourced skill learning; whether judges weight self-improvement loops.
- Phase 0 outcome: if Fable vision underperforms Gemini badly on the hard set, demo story becomes "orchestrated toolbox" — decide Friday night, not Saturday.
- Capture bridge for the plugin tier: MCP server vs watched-folder sync — prototype whichever is faster Friday, decide by use.
- Commons governance once contributors >1: maintainer model, skill versioning policy, what "merged learning" attribution looks like.
- Registry hosting: static page in the commons repo (free, simple) vs a route on the v2 app — lean static until the v2 library ships.

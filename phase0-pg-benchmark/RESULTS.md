# Phase 0 Benchmark — Patrologia Graeca OCR (PG 57, Chrysostom *In Matthæum*)

*Run 2026-06-11, evening before the Fable 5 hackathon (Sat 06-13). Source: archive.org `patrologiae_cursus_completus_gr_vol_057_chrysostom_matthew`, Google scan, native ~4080×6210/page. Test pages (PDF index): 60 Latin control · 90 faded Greek · 150 homily opening · 250 dense apparatus · 350 ligature-dense. Benchmark images: full page 1600px wide, column crops 1400px. Ground truth: `results/GROUND_TRUTH_p150_col1_top.md`, transcribed from a native-res crop where every diacritic is unambiguous; apparatus blocks verified against native-res crops of p150 + p250.*

## Scorecard

| Model | Method | Body text (canonical Greek) | Apparatus (non-canonical) | Failure mode | Speed / cost |
|---|---|---|---|---|---|
| **Gemini 2.5 Flash** | single-shot full page | **~99%+** — all diacritic traps correct (περιῄει, ἀγορᾷ μέσῃ, ἐρημίᾳ) | ~85–90%: superscript digits systematically misread (⁵⁰→⁸² series), sigla garbled (Mor.→Mur., A.F.→Α.Β.), **canonical-bias substitution ἐδίδαξεν→ἐδίδασκεν** | drifts toward the canonical reading exactly where the page's value is its deviation | ~37s/page, ~2¢/page incl. thinking |
| **Claude Opus 4.8** | subagent, went **agentic unprompted** (56–61 tool calls, self-cropped/zoomed) | **~GT-level** — verified word-for-word vs native crops on p150 + p250 | **~GT-level incl. footnote numbering ⁴⁸–⁶³ correct**; flagged its own 2 low-confidence sigla | minor sigla blur at worst, self-flagged | ~10 min/page, ~100K subagent tokens (plan, not API) |
| **Claude Fable 5** | orchestrator, agentic (native-res strips) | GT-level on sampled region (is also the GT author — see caveats) | reads apparatus cleanly at native res | — | in-session |
| **Claude Sonnet 4.6** | subagent, single-shot | **honest refusal** — diagnosed resolution starvation, marked rather than guessed, recommended per-column 300+ DPI crops | declined | none — exemplary behavior under uncertainty | ~2.5 min |
| **Claude Haiku 4.5** | subagent, single-shot | **0% — total confabulation.** Fluent fake patristic Greek unrelated to the page, all four runs; hallucinated header "ΕΥΣΕΒΙΟΥ ΚΑΙΣΑΡ." on a Chrysostom volume; admitted "reconstructing from context" | fabricated | confident fabrication — worst possible mode | ~30–60s |

Side data: Gemini on p350 (ligature-dense) burned 15.7K *thinking* tokens and hit MAX_TOKENS after 654 output tokens — page difficulty is measurable in thinking spend; raise the cap or disable thinking for OCR calls. Gemini's column-crop run introduced an error the full-page run didn't (θεραπεύσας for θεραπεῦσαι) — crops are not automatically better for Gemini.

## Findings

1. **Resolution is the master variable.** Every Claude single-shot failure traces to the vision pipeline downscaling a full page below diacritic legibility. The fix is not a better model but better inputs: programmatic layout segmentation → native-res region crops. This is the hybrid thesis confirmed.
2. **Agency beats model size.** Opus succeeded *because it behaved like an agent* — it self-cropped 60 times rather than accepting the thumbnail. The same model single-shot would have been Sonnet-or-worse. The product is the loop, not the engine.
3. **The canonical-bias error is the demo.** Gemini genuinely reads (apparatus structure proves it's not reciting), but where the page disagrees with the famous reading, its prior wins (ἐδίδαξεν→ἐδίδασκεν). For textual scholarship the deviation IS the data. "Cheap model transcribes, Claude adversarially verifies against the page image and catches the canonical-bias substitution" is a stage-ready beat.
4. **Tier discipline:** Haiku must never touch vision transcription at this density (text-only cleanup gates remain fine). Sonnet is a trustworthy cheap *legibility gate* (it knows what it can't read). Opus/Fable are verification and apparatus duty. Gemini Flash is the bulk-body workhorse at ~2¢/page.
5. **Recitation-assist caveat for the wedge.** Body-text accuracy on Chrysostom is partly prior-assisted; ScanDoc's rare-book corpus is by definition NOT in priors. Expect real-world Gemini accuracy between its "body" and "apparatus" numbers here — which is exactly why the verify pass exists.

## Routing table v0 (for the skill commons)

| Region type | Route |
|---|---|
| Layout segmentation | programmatic (PyMuPDF/OpenCV), never a vision call |
| Body text, bulk | Gemini Flash on native-res region crops, thinking off |
| Apparatus, superscripts, sigla, marginalia | Claude agentic zoom (Opus tier+) |
| Verification / diacritic QA / canon-bias audit | Claude (Sonnet+ text pass; Opus/Fable vision spot-checks against crops) |
| Legibility triage ("can anything read this?") | Sonnet single-shot — trust its refusals |
| Any vision transcription | never Haiku |

## Caveats

n=2 pages graded in depth (150, 250); spot-CER on anchored regions, not corpus CER; Fable graded itself on the GT region (uncontaminated comparison region was strip 2, also clean); Claude subagent token counts include harness overhead and are not API-cost comparable; Latin control (p60) and faded page (p90) transcribed by Gemini but not yet graded.

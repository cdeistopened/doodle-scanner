# Discrepancies — idx150 (PG 52, cols 529–530)

Verification of `gemini_idx150_full.md` against native scan (4030×6206), strip-by-strip at native resolution. Diplomatic standard: corrected = as printed.

| # | Location | Transcript read | Print says | Verdict | Proposed emendation | Confidence |
|---|----------|-----------------|------------|---------|---------------------|------------|
| 1 | Monitum, l.5–6 ("de die / mensis...") | `de die mensis aut quæstio` | `de die / mensis est quæstio` | GEMINI-ERROR | fix `aut` → `est` | High |
| 2 | Monitum, "EPISTOLA INNOCENTII AD JOANNEM..." l.17 | `...Sozomenus, tota / observantiæ caritatisque plena` (dropped words) | `...Sozomenus, tota consolatoria est, et / observantiæ caritatisque plena` | GEMINI-ERROR | insert `consolatoria est, et` | High |
| 3 | Monitum, "EPISTOLA ... AD CLERUM..." l.20–21 | `est responsio ad epistolam / puli` (line skip) | `est responsio ad epistolam ejusdem cleri et po-/puli` | GEMINI-ERROR | insert `ejusdem cleri et po-` (the `po-` + `puli` form the hyphenated word `populi`) | High |
| 4 | Monitum, l.23 | `admodum opiare` | `admodum optare` | GEMINI-ERROR | fix `opiare` → `optare` | High |
| 5 | Monitum, "EPISTOLA HONORII..." l.24 (small caps) | `ARCADIUIM` | `ARCADIUM` | GEMINI-ERROR | fix `ARCADIUIM` → `ARCADIUM` (confirmed at zoom: A-R-C-A-D-I-U-M, no double I) | High |
| 6 | Monitum, l.24–25 | `ex Codice Vaticano / nali Baronio publicata` (line skip; `nali` orphaned) | `ex Codice Vaticano educta, et a Cardi-/nali Baronio publicata` | GEMINI-ERROR | insert `educta, et a Cardi-` (`nali` is the tail of hyphenated `Cardi-nali`) | High |
| 7 | Monitum, l.29 | `pestringeret` | `pestringeret` | MIGNE-TYPO (transcript matches print) | keep `pestringeret [sic]`; classical/intended form is `perstringeret` | High |
| 8 | Monitum, l.35 | `lisdem epistola altera` | `Iisdem epistola altera` | GEMINI-ERROR | fix `lisdem` → `Iisdem` (capital I, not l) | High |
| 9 | Gemini output tail, l.86–92 `[APPARATUS]` block | footnotes (a) and ¹ re-emitted a second time | (single occurrence each, in position) | GEMINI-ERROR (duplication) | drop the duplicate `[APPARATUS]` block; footnotes appear once in their correct positions | High |
| 10 | Footnote (a), italic | `*domino meo` (single unclosed asterisk) | `domino meo` set in italics | OK (minor markup) | close italics: `*domino meo*` | High |

## Suspects adjudicated (from brief) — outcomes
- "de die mensis aut quæstio" → **#1** `est`, not `aut`. Not "erit".
- "ARCADIUIM" → **#5** `ARCADIUM`.
- "ex Codice Vaticano nali Baronio publicata" → **#6** line-skip; full text `ex Codice Vaticano educta, et a Cardi-/nali Baronio publicata`. So "a Card. Baronio" hypothesis ≈ correct in spirit ("a Cardinali Baronio"); "nali" is the orphaned tail of "Cardi-nali", not garbage.
- "admodum opiare" → **#4** `optare`.
- "pestringeret" → **#7** printed as `pestringeret` (Migne typo for `perstringeret`); kept `[sic]`.
- "lisdem epistola altera" → **#8** `Iisdem`.
- "est responsio ad epistolam puli, a Germano" → **#3** line-skip; full `...ad epistolam ejusdem cleri et po-/puli, a Germano...`.
- "subjunguntur, quia ad eamdem negotiarum/negotiorum seriem" → **NOT PRESENT** on this page.
- Salutation small-caps line + "(a)" placement → **OK**: `DOMINO MEO REVERENDISSIMO, PIENTISSIMOQUE INNOCENTIO JOANNES IN DOMINO SALUTEM (a).` — (a) correctly at end of salutation.

## Spot-checks (no error found)
- Header `529 EPISTOLÆ. 530` — OK.
- Title `INNOCENTIO EPISCOPO ROMÆ.` — OK.
- Col 1 body §1 opening `1. Etiam antequam redditæ sunt literæ nostræ...` — OK verbatim through `de omnibus vos`.
- Col 1 names `Demetrio, Pansophio, Pappo, et Eugenio` — OK.
- Footnote (a) `...Alberti Fabricii Codice quem Fabricianum vocamus ; scripta anno 404...` — OK.
- Col 2 `manifeste docentes, quo quantocius rebus succurratur...` through `...advenerant ¹,` — OK verbatim.
- Col 2 `...collecta multitudine episcoporum non paucorum huc venit, quo præludio...` — OK.
- Note ¹ `Duo optimi Mss. Coislinianus et Reg. unus, et eos qui` — OK (ends mid-phrase; italic variant reading).
- Editorial folio marker `[515]` in monitum — OK.

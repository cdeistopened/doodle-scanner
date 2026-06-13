# Discrepancies — PG52 idx151 (cols. 531–532)

Verified strip-by-strip against `pages/idx151_native.png` (native 4030×6206) at zoom. Diplomatic standard: corrected = exactly what is printed. **No printed Migne typos found on this page** — every adjudicated suspect resolved to a Gemini OCR error or an OK.

| # | Location | Transcript read | Print says | Verdict | Proposed emendation | Confidence |
|---|----------|-----------------|------------|---------|---------------------|------------|
| 1 | Col 531, l. 8–9 (suspect) | `Unde nos ca-` / `videntes` | `Unde nos ea videntes` ("ea" = neuter pl.; **no hyphen**, "ea" is a whole word) | GEMINI-ERROR | "ca-" → "ea" (and drop the spurious hyphen; line ends after "ea", "videntes" begins next line) | High |
| 2 | Col 531, l. 21 (incidental) | `virosque honorem deferentes` | `viroque honorem deferentes` | GEMINI-ERROR | "virosque" → "viroque" | High |
| 3 | Col 531, l. 22 (suspect) | `super hæc re ejus literas` | `super hac re ejus literas` (correct abl., not "hæc") | GEMINI-ERROR | "hæc" → "hac". **Not a Migne typo** — print is correct Latin | High |
| 4 | Col 531, l. 36 heading (suspect) | `2. *Chrysostomus cur coram...` (lone `*`, no closing) | Printed: `2.` then the heading **fully in italic**; **no printed asterisk** | GEMINI-ERROR (markup artifact) | The `*` is the OCR's italic-open marker, not a glyph. Render heading as italic (`2. *Chrysostomus … recusarit.*`); do not keep a literal `*` | High |
| 5 | Col 531, l. 41 (suspect) | `Pesinuntis` | `Pesinuntis` (Greek side Πισινοῦντος) | OK | — | High |
| 6 | Col 531, l. 41–42 (suspect) | `Eulysium` / `Apameæ` | `Eulysium Apameæ` | OK | — | High |
| 7 | Col 531, l. 42 (suspect) | `Lupicinum Appiariæ` | `Lupicinum Appiariæ` | OK | — | High |
| 8 | Col 531, l. 50–51 (suspect) | `ut in judicis thronum conscen-/dat sibi minime congruentem` | identical | OK | — | High |
| 9 | Col 531, apparatus note * (suspect) | `et primus prius Editorum le-ctioni` | `si stemus prius Editorum le-/ctioni` | GEMINI-ERROR | "et primus" → "si stemus" | High |
| 10 | Col 531, apparatus note * opening | `* Cum ipso erant :` (capital C, colon) | `cum ipso erant ;` (lowercase c — it's a continuation lemma; semicolon, not colon) | GEMINI-ERROR | "Cum … :" → "cum … ;" (the `*` is the real footnote marker and is kept) | Med-High |
| 11 | Col 531, note ² (suspect) | `etiamsi reum inimicumque accusent` | `etiamsi reum inimicumque accusent` | OK | — | High |
| 12 | Col 532, l. 87 (suspect) | `Curioso urbis` (capital C) | `Curioso urbis` (capital C as printed) | OK | — | High |
| 13 | Col 532, l. 92 heading (suspect) | `Chrysostomus a priori exsilio redux...` (no `*` in transcript) | Heading **fully italic**, complete, ends `Violenta in Ecclesiam irruptio.—` | OK (content) / render as italic | Mark heading italic for consistency with #4 | High |
| 14 | Col 532, l. 97 (suspect) | `quæ mala hæc sistantur` | `quo mala hæc sistantur` ("ut … quo" purpose clause; not "quæ", not "quibus") | GEMINI-ERROR | "quæ" → "quo" | High |
| 15 | Col 532, note (a) | `significat libellum accusationis complexionem` | `significat libellum accusationes complecten-/tem` | GEMINI-ERROR | "accusationis complexionem" → "accusationes complectentem" | High |

## Spot-checks (not on suspect list) — all OK

Col 531: l. 7 "bene instructa : at neque" ✓ · l. 25–26 "prioribus conatibus cumulum addere contendens" ✓ · l. 30 "desertæ in dies ecclesiæ, dum e singulis ecclesiis ab-" ✓ · note ¹ "Sic omnes Mss. recte. In Editis hæc … desunt." ✓

Col 532: l. 76 "et ad synodum appellaremus, judiciumque requireremus" ✓ · l. 102–103 "quasi incendium depascens omnia, fugientes, in suas regiones" ✓ · l. 113 "Imperatoris literæ in omnia loca mitterentur et unde-cumque" ✓ · l. 120 "neque putarent ea quæ ipsi una solum præsente parte" ✓

Header/column numbers: `531` · `S. JOANNIS CHRYSOST. ARCHIEP. CONSTANTINOP.` · `532` — all ✓

## Notes
- Footnote markers `*`, `¹`, `²`, `(a)` are all REAL printed markers and retained.
- The `*` Gemini placed after section heading "2." (item #4) is NOT a printed marker — it is the OCR's italic-open convention, left unclosed. Distinguish from the genuine footnote `*` (item #10).
- No `[sic]` inserted anywhere: no genuine Migne typo was found. Every flagged suspect was either correct Latin already in print (hac, viroque, quo, Pesinuntis) or a pure OCR corruption.

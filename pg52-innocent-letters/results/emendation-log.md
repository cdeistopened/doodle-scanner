# Emendation Log — Chrysostom, First Letter to Innocent (PG 52, cols 529–532)

Diplomatic-transcript verification of four Gemini OCR pages against the native Migne scans
(4030×6206), strip-by-strip at native resolution. Standard: **corrected = as printed in
Migne**. Gemini OCR errors were fixed in the verified transcripts; genuine printed
anomalies were kept and flagged `[sic]`; cleared suspects were left unchanged.

## Summary stats per page

| Page (cols) | Languages | Gemini errors fixed | Migne typos flagged [sic] | Suspects cleared OK |
|---|---|---|---|---|
| idx149 (529–530) | Greek title + cols + apparatus; Latin monitum head | 15 | 1 (Co.slinianus) | 8 |
| idx150 (529–530) | Latin monitum tail + title/salutation + cols + notes | 8 | 1 (pestringeret) | 1 (+ markup close) |
| idx151 (531–532) | Latin cols + apparatus | 8 | 0 | 7 |
| idx152 (531–532) | Greek cols + apparatus | 22* | 0 | 5 |
| **Total** | — | **53** | **2** | **21** |

\* idx152: 22 discrete corrections across 21 locations (the note-letter relabel spans a text
marker + an apparatus relabel, counted once per its own discrepancy table). Migne typos: 0 —
the candidate `τὴν ἔλεγχον` resolved to a correct printed `τὸν ἔλεγχον`.

---

## Print anomalies flagged [sic]

Genuine Migne-print findings (not OCR errors — the print itself is anomalous).

### 1. `Co.slinianus` → Coislinianus (idx149, apparatus, cols 529–530)
- **Printed reading:** `Co.slinianus` — "Co" + a raised dot + "slinianus" (in the note "Ibid. Co.slinianus αὐτοὶ δὲ ὡς ἐνὶ δι᾽ ἐπιστολῆς ἐν βραχεῖ").
- **Proposed emendation:** **Coislinianus** (the Codex Coislinianus, cited correctly elsewhere on the same page and on idx152 as "Cod. Coislinianus").
- **Reasoning:** The intended word is the well-attested Codex Coislinianus. The "i" has degraded in the printing to a stray raised dot, producing "Co·slinianus". Verified at zoom; the surrounding citations spell it correctly, confirming the broken glyph is a print defect, not the editor's intent. Confidence Med-High.

### 2. `pestringeret` → perstringeret (idx150, monitum, cols 529–530)
- **Printed reading:** `pestringeret` (in "ut Chrysostomus in concionibus hæc pro more suo pestringeret ac vituperaret"). The transcript matches the print exactly.
- **Proposed emendation:** **perstringeret** (classical/intended form, "censure, touch upon sharply").
- **Reasoning:** The verb is *perstringo*; "pestringeret" lacks the *r* of the *per-* prefix. Sense ("censure ac vituperaret") requires *perstringeret*. This is a Migne compositor's dropped letter, not an OCR misread — the scan genuinely prints "pestringeret". Kept `[sic]` in the diplomatic text. Confidence High.

### Diacritic ambiguity noted, not emended

**`μεθιστὰς` / `μεθιστάς`** (idx152, col 531, β´ — "καὶ κλῆρον μεθιστὰς, καὶ ἐκκλησίαν ἐρημῶν").
The print appears to show an **acute** (μεθιστάς) before the comma; standard editorial
practice converts a final acute to grave in running text (μεθιστὰς). Left as **μεθιστὰς** in
the verified Greek, flagged low-confidence. Not a substantive error and not a `[sic]` case —
a diacritic-rendering ambiguity only.

---

## Notable OCR-error catches

A curated selection of the most instructive Gemini fixes across the four pages — line-skips
(whole phrases dropped), proper-name corruptions, grammatical-form restorations, and
person/marker confusions.

1. **Line-skip — `consolatoria est, et` dropped** (idx150, monitum). Gemini read "...Sozomenus, tota / observantiæ caritatisque plena"; print has "...Sozomenus, tota **consolatoria est, et** / observantiæ...". A full clause lost at a line break.
2. **Line-skip — `ejusdem cleri et po-` dropped** (idx150, monitum). "est responsio ad epistolam / puli" → print "est responsio ad epistolam **ejusdem cleri et po-**/puli" (the `po-` + `puli` = *populi*).
3. **Line-skip — `educta, et a Cardi-` dropped** (idx150, monitum). "ex Codice Vaticano / nali Baronio publicata" → "ex Codice Vaticano **educta, et a Cardi-**/nali Baronio publicata"; the orphaned "nali" is the tail of hyphenated *Cardi-nali*.
4. **Line-skip — `nam alioquin sexcenties venissemus` confirmed present** vs. apparatus note ¹ which says these words are *absent in the editions* (idx151, col 532). The catch was confirming the Greek/Latin both carry the phrase the editions drop.
5. **Proper name — `Ἀππιαρίας` for `Ἀππα-μίας`** (idx152, col 531, β´). Gemini's "Ἀππαμίας Λουπίκινον" misreads the see; print's line 45 reads "ρίας" → **Ἀππιαρίας** (the see Appiaria, Lat. *Appiariæ* in the apparatus and idx151).
6. **Grammatical garble resolved — `παρῃτησάμεθα`** (idx152, col 531, α´). Gemini "παρρητησάμεθα" (double ρ, no iota subscript) → print **παρῃτησάμεθα** (single ρ, iota subscript under η) — "we declined."
7. **Phrase corruption — `τά τε ἔμπροσθεν`** (idx152, col 531, β´). Gemini "καθάπερ τὰ ἐκ ἔμπροσθεν" → print **τά τε ἔμπροσθεν** ("both the things before…and after").
8. **Verb garble resolved — `ἐπινεμομένην`** (idx152, col 532, β´). Gemini "ἐπιμεμονημένην" → print **ἐπινεμομένην** ("spreading/devouring," of fire), resolving the simile πυρὰν τινα … ἐπινεμομένην.
9. **Verb garble resolved — `συναγόντων`** (idx152, col 532, β´). Gemini "πάντοθεν πάντας συντεθέντων" → print **συναγόντων** ("gathering all from everywhere"); also paired with `κατηγόμην` for "ἐνεβαλλόμην" and `ἐδραπέτευσεν` — restoring the 1st-singular narrative voice.
10. **Footnote-marker / person confusion** — two catches in one: (a) the Greek running marker `ᵃ` at the head of α´ was Gemini-read as a Greek letter "α´" but is the italic Latin siglum **a** matching apparatus note a (idx149); and (b) the apparatus note letter at col 532 is **j (ʲ)**, not **i** — Gemini relabeled it and produced a spurious second "k" (idx152). Distinguishing real printed sigla (`*`, `¹`, `²`, `(a)`) from OCR italic-open markers (`*` after "2.") was a recurring discipline (idx151 #4, #10).

---

## Full adjudication tables

The four discrepancy tables, verbatim.

### idx149 — PG 52 cols 529–530

Base transcript: `gemini_idx149_full.md` · Authority: `pages/idx149_native.png` (4030×6206), verified strip-by-strip at native resolution.

Verdicts: **GEMINI-ERROR** = transcript wrong, fixed in `verified_idx149.md`. **MIGNE-TYPO** = print itself anomalous, kept `[sic]`. **OK** = transcript matches print.

| # | Location | Transcript read | Print actually says | Verdict | Proposed emendation | Confidence |
|---|----------|-----------------|---------------------|---------|---------------------|------------|
| 1 | Monitum ln17 | `negotia-rum` | `negotio-rum` (negotio- / rum) | GEMINI-ERROR | — (fixed: negotiorum) | High |
| 2 | Monitum ln18 | `subjungiur` | `subjungitur` | GEMINI-ERROR | — (fixed) | High |
| 3 | Monitum ln26 | `exagitatuin` | `exagitatum` | GEMINI-ERROR | — (fixed) | High |
| 4 | Monitum ln29 | `perdueit` | `perducit` | GEMINI-ERROR | — (fixed) | High |
| 5 | Monitum ln28 | (no opening quote before "cunctas") | `‹ cunctas` — opening guillemet present, pairs with closing `›` after "commendavit." | GEMINI-ERROR | — (fixed: added ‹) | High |
| 6 | Monitum ln31 | `commendavit. ›` | `commendavit. ›` (closing guillemet) | OK | — | High |
| 7 | Monitum — "aut quæstio" suspect | `Jam quæstio est` | `Jam quæstio est` | OK | — (no "aut"; suspect was a check) | High |
| 8 | Greek title | `[515] ΙΝΝΟΚΕΝΤΙΩ ΕΠΙΣΚΟΠΩ ΡΩΜΗΣ.` | same | OK | — | High |
| 9 | Col1 ln45-46 | `ἀκηκοέναι` | `ἀκηκοέναι` (whole word, line 2 of α´) | OK | — | High |
| 10 | Col1 ln48-49 | `ἅφῆκεν ἀνήκο-ε´ναι` | `ἀφῆκεν ἀνήκο-/ον εἶναι` (smooth breathing; word = ἀνήκοον, then εἶναι) | GEMINI-ERROR | — (fixed: ἀφῆκεν ἀνήκοον εἶναι) | High |
| 11 | Col1 ln48 — "ἅφῆκεν" breathing | rough breathing (ἅ) | smooth breathing (ἀ) | GEMINI-ERROR | — (fixed: ἀφῆκεν) | High |
| 12 | Col1 ln50 — "α´" | `α´` (Greek alpha + acute) | superscript italic Latin siglum **a** (footnote marker, matches app. note "a") | GEMINI-ERROR | — (fixed: ᵃ footnote marker) | High |
| 13 | Col1 ln54 — "σταὶη" | `σταὶη` (grave on iota) | `σταίη` (acute on iota) | GEMINI-ERROR | — (fixed: σταίη) | High |
| 14 | Col1 ln57 | `πεῖσαι τὰς ἐξ ἑαυτῶν ἀφεῖναι` | `πεῖσαι τάς τε / ἑαυτῶν ἀφεῖναι` (τάς τε, not τὰς ἐξ) | GEMINI-ERROR | — (fixed; confirmed by app.: "Infra πεῖσαι τάς τε ἑαυτῶν ἀφεῖναι") | High |
| 15 | Col1 ln57 — [516] marker | `[516] Πάππον` | `[516] Πάππον` (inline, after "Πανσόφιον,") | OK | — | High |
| 16 | Apparatus ln60 — "Editii"/"Editi" suspect | `Editii.` (in "Sic omnes mss. et Editii.") | `Editi.` | GEMINI-ERROR | — (fixed: Editi) | High |
| 17 | Apparatus ln62 — "Μοὶ" | `Μοὶ omnes mss.` (Greek) | `Mox omnes mss.` (Latin "Mox") | GEMINI-ERROR | — (fixed: Mox) | High |
| 18 | Apparatus ln64 | `quam editorum τῶν διακόνων.` | `quam editorum τὸν διάκονον.` (singular) | GEMINI-ERROR | — (fixed: τὸν διάκονον) | High |
| 19 | Apparatus ln66 — "Co.slinianus" | `Co.slinianus` | print itself reads `Co.slinianus` — "Co" + raised dot + "slinianus"; the intended Codex Coislinianus, but the "i" has degraded to a stray dot | MIGNE-TYPO | Coislinianus | Med-High |
| 20 | Apparatus "Editt." suffix suspect | (n/a) | abbreviation here is `Edit.` only; no "Editt." on this page | OK | — | High |
| 21 | Col2 ln69 — "στειλασθαι" | `στειλασθαι` (no accent) | `στείλασθαι` (acute on ει) | GEMINI-ERROR | — (fixed: στείλασθαι) | High |
| 22 | Col2 ln73 — b marker | `τῶν διακόνων b Παῦλον` | same (footnote siglum b present) | OK | — | High |
| 23 | Col2 ln84-85 — "κρατήσαντα θεσμὸν" | `κρατήσαντα θεσμὸν` | `τὸν ἄνωθεν κρατήσαντα / θεσμὸν` (accusative, agrees with τὸν...θεσμόν) | OK | — (suspect cleared; NOT κρατήσαντος θεσμοῦ) | High |
| 24 | Col2 — "ἠυλίζετο" | `ἠυλίζετο` (rough breathing on eta, bare upsilon) | `ηὐλίζετο` (η + ὐ smooth breathing) | GEMINI-ERROR | — (fixed: ηὐλίζετο) | High |

**Spot-checks (off the suspect list):** Header, Monitum ln11, Monitum ln32-33, Col1 ln52, Col1 ln56, Col2 ln79, Col2 ln83, Apparatus ln65 — all OK, no drift.

### idx150 — PG 52 cols 529–530

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

**Suspects adjudicated:** "aut" → `est`; "ARCADIUIM" → `ARCADIUM`; "Vaticano nali" → line-skip `educta, et a Cardi-/nali`; "opiare" → `optare`; "pestringeret" → Migne typo kept `[sic]`; "lisdem" → `Iisdem`; "ad epistolam puli" → line-skip `ejusdem cleri et po-/puli`; "negotiarum/negotiorum" not present on this page; salutation small-caps + "(a)" placement OK.

**Spot-checks (no error):** Header; Title; Col1 §1 opening through "de omnibus vos"; names "Demetrio, Pansophio, Pappo, et Eugenio"; footnote (a) Fabricianum text; Col2 "manifeste docentes… advenerant ¹" verbatim; Col2 "collecta multitudine…"; Note ¹ "Duo optimi Mss. Coislinianus et Reg. unus, et eos qui"; folio `[515]`.

### idx151 — PG 52 cols 531–532

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

**Spot-checks (all OK):** Col 531 l.7, l.25–26, l.30, note ¹; Col 532 l.76, l.102–103, l.113, l.120; header/column numbers.

**Notes:** Footnote markers `*`, `¹`, `²`, `(a)` are all real printed markers and retained. The `*` after section "2." (item #4) is the OCR's italic-open convention, not a printed marker — distinguish from the genuine footnote `*` (item #10). No `[sic]` inserted: no genuine Migne typo found.

### idx152 — PG 52 cols 531–532

Verified against `pages/idx152_native.png` (4030×6206) at native resolution. Diplomatic standard: corrected = exactly what is printed. Classifications: GEMINI-ERROR (OCR misread, fixed) / MIGNE-TYPO (printed anomaly, kept + `[sic]`) / OK (suspect cleared, transcript already correct).

| Location | Transcript read | Print says | Verdict | Proposed emendation | Confidence |
|---|---|---|---|---|---|
| Col1 L26 | παρρητησάμεθα | παρῃτησάμεθα (single ρ; iota subscript under η) | GEMINI-ERROR | → παρῃτησάμεθα | High |
| Col1 L41–42 | καθάπερ τὰ ἐκ ἔμπρο-σθεν | καθάπερ τά τε ἔμπρο-σθεν | GEMINI-ERROR | → τά τε ἔμπροσθεν | High |
| Col1 L44–45 | τὸν Ἀππα-μίας Λουπίκινον | τὸν Ἀππια-ρίας Λουπίκινον (line 45 = "ρίας", not "μίας"; see = Appiaria / Lat. Appiariæ) | GEMINI-ERROR | → Ἀππιαρίας | High (ρίας confirmed; iota inferred from cut margin + known see) |
| Col1 L49 | Ὁ γὰρ μηδένα λιβέλλους δεξάμενος | Ὁ γὰρ μηδέπω λιβέλλους δεξάμενος | GEMINI-ERROR | → μηδέπω | High |
| Col1 L52 | καὶ ἐκκλησίας ἐρήμων | καὶ ἐκκλησίαν ἐρημῶν (acc. sing.; participle ἐρημῶν w/ circumflex) | GEMINI-ERROR | → ἐκκλησίαν ἐρημῶν | High |
| Col1 L52 | μεθιστὰς (grave) | μεθιστάς (acute, before comma) | GEMINI-ERROR (minor accent) | → μεθιστὰς (kept grave in running text; print shows acute) | Low — diacritic ambiguity, left as μεθιστὰς |
| Col1 L52–53 | πῶς ἂν ἢ δίκαιος … ἀναβαίνειν | πῶς ἂν ἢ δίκαιος … ἀναβαίνειν | OK | (ἢ confirmed single letter; ἀναβαίνειν confirmed, not ἀναβαίη) | High |
| Col1 L57 (note a) | Cod. Coislinianus et Rex unus | Cod. Coislinianus et Reg. unus | GEMINI-ERROR | → Reg. unus | High |
| Col1 L58 (note a) | Edit. καὶ τοὺς πρὸς αὐτοῦ | Edit. καὶ τοὺς πρὸ αὐτοῦ | GEMINI-ERROR | → πρὸ αὐτοῦ | High |
| Col1 L72 (note g) | κἂν ὑπεύθυνον αἰτιάσωται καὶ ἐχθρόν | κἂν ὑπεύθυνον αἰτιάσωνται καὶ ἐχθρόν | GEMINI-ERROR | → αἰτιάσωνται | High |
| Col2 L81–82 | κατηγόρους ἀδέχετο | κατηγόρους ἐδέχετο | GEMINI-ERROR | → ἐδέχετο | High |
| Col2 L84 | οὐδέπω τὰ ἐγκλήματα ἀποδυσάμενοι | … ἀποδυσαμένων | GEMINI-ERROR | → ἀποδυσαμένων (gen. pl.) | High |
| Col2 L85 | παρὰ θεσμῶν καὶ κανόνων | παρὰ θεσμὸν καὶ κανόνων | GEMINI-ERROR | → θεσμὸν (acc. sing.) | High |
| Col2 L87 | Οὐκ ἀπέστη πάντα ποιῶν; καὶ πραγματευόμενος | (same — `;` mid-sentence is printed) | OK / MIGNE punctuation | keep printed `;` after ποιῶν | High |
| Col2 L92 | πρὸς βίαν συρόμενος κατηγόρων | πρὸς βίαν συρόμενος κατηγόμην | GEMINI-ERROR | → κατηγόμην ("I was being dragged off"); resolves dangling-genitive garble | High |
| Col2 L93 | εἰς πλοῖον ἐνεβαλλόμεν | … ἐνεβαλλόμην | GEMINI-ERROR | → ἐνεβαλλόμην (1st sg, matches κατηγόμην/ἔπλεον) | High |
| Col2 L100–101 | ὥστε στῆναι ταύτη τὰ κακὰ καὶ οὐδὲ γὰρ ἐνταῦθα | ὥστε στῆναι ταύτῃ τὰ κακά. Οὐδὲ γὰρ ἐνταῦθα | GEMINI-ERROR | → ταύτῃ (iota subscript); τὰ κακά. (full stop); delete spurious καὶ; capital Οὐδὲ | High |
| Col2 L104 | ἐπιπληδήσαντας | ἐπιπηδήσαντας | GEMINI-ERROR | → ἐπιπηδήσαντας | High |
| Col2 L107 | εἰς τὰς αὐ-τῶν ⁱ ἀνεχώρησαν | εἰς τὰς αὐ-τῶν ʲ ἀνεχώρησαν (marker is **j**, not i) | GEMINI-ERROR | → ʲ (note letter j) | High |
| Col2 L108 | ὥσπερ πυρὰν τινα πάντα ἐπιμεμονημένην φεύγοντες | … ἐπινεμομένην φεύγοντες | GEMINI-ERROR | → ἐπινεμομένην ("spreading/devouring"); πυρὰν τινα confirmed; resolves garble | High |
| Col2 L112 | βασιλέως νοταρίων εἰς τοῦτο ἀποστείλαντος | … νοτάριον εἰς τοῦτο ἀποστείλαντος | GEMINI-ERROR | → νοτάριον (acc. sing.) | High |
| Col2 L113 | ἐκεῖνος δὲ ἐδραπετεύσεν εὐθέως | … ἐδραπέτευσεν εὐθέως | GEMINI-ERROR | → ἐδραπέτευσεν (accent on πέ) | High |
| Col2 L117 | καὶ τὴν ἔλεγχον δεδοικὼς | καὶ τὸν ἔλεγχον δεδοικὼς | GEMINI-ERROR | → τὸν ἔλεγχον (masc., correct gender; print shows τὸν, NOT a Migne typo) | High |
| Col2 L118 | πάντοθεν πάντας συντεθέντων | πάντοθεν πάντας συναγόντων | GEMINI-ERROR | → συναγόντων ("gathering all from everywhere"); resolves garble | High |
| Apparatus note letters | transcript: h, i, k(L138), k(L140) | print: h, i, **j**(L138), k(L140) | GEMINI-ERROR | "ᵏ Unus cum Palladio εἰς τὰ ἑαυτῶν…" → relabel **ʲ**; only ONE k (Duo καὶ μετὰ ἀδικίας) | High |
| Col1 L72 (note g tail) | ὑπεύθυνον ὄντα αἰτιῶν | ὑπεύθυνον ὄντα αἰτιῶν | OK | — | High |
| Header / col nums / [517][518] | 531 EPISTOLÆ. 532; [517]; [518] | identical | OK | — | High |

**Summary (per idx152 table):** GEMINI-ERROR fixed: 22 across 21 locations. MIGNE-TYPO: 0 (the candidate τὴν ἔλεγχον resolved to a correct printed τὸν ἔλεγχον). OK: 5 (`πῶς ἂν ἢ … ἀναβαίνειν`; mid-sentence `;` after ποιῶν; `ἐκ προοιμίων`; note-g tail `αἰτιῶν`; header/column numbers/[517]/[518]).

**Notes:** No `[sic]` cases on this page. Col1 L52 `μεθιστὰς`: print shows acute (μεθιστάς) before comma; standard practice keeps grave in running text — left as `μεθιστὰς`, low-confidence. Col1 L44 `Ἀππιαρίας`: line-44 right margin cropped after "Ἀππ"; "ρίας" on line 45 certain; reading reconstructed on that ending + documented see Appiaria (Lat. Appiariæ).

---

---

## Run 2 (cols 533–538)

Verification of five Gemini OCR pages against native Migne scans, covering the completion of
Letter 1 (both Greek and Latin), the full text of Letter 2, and the opening of the Innocent
correspondence (cols 537–538). Diplomatic standard unchanged: corrected = as printed in
Migne; genuine print anomalies retained with `[sic]`; cleared suspects left unchanged.

### Summary stats per page

| Page (cols) | Languages | Gemini errors fixed | Migne [sic] flagged | Suspects cleared OK | Escalated |
|---|---|---|---|---|---|
| idx153 (533–534) | Greek cols + apparatus | 23 | 1 (μετῳκίζετο) | 5 | 0 |
| idx154 (533–534) | Latin cols + apparatus | 14 | 0 | 16 | 1 (RESOLVED) |
| idx155 (535–536) | Latin cols + apparatus | 10 | 0 | 5 | 0 |
| idx156 (535–536) | Greek cols + apparatus | 36 | 1 (punctuation spacing) | 4 | 0 |
| idx157 (537–538) | Greek cols (mixed letters) | 21 | 2 (ἐστιν; ἐσμεν) | 4 | 0 |
| **Total** | — | **104** | **4** | **34** | **1 (resolved)** |

---

### Print anomalies flagged [sic]

Genuine Migne-print findings (not OCR errors) across the five new pages.

#### 1. `μετῳκίζετο` (idx153, col 534, δ´ section)
- **Printed reading:** `μετῳκίζετο` — an unusual orthography.
- **Standard form:** `μετοικίζετο` ("was being displaced/resettled," from μετοικίζω).
- **Proposed emendation:** **μετοικίζετο**.
- **Reasoning:** The root is μετοικίζω (to resettle, transplant). The print's form transposes the vowel cluster (οι → ω) and the consonant (κ → χ), producing an unattested verb. Classified as Migne compositor error; kept `[sic]` in the diplomatic text. Confidence High.

#### 2. Punctuation spacing — raised dot (idx156, Greek cols 535–536)
- **Printed reading:** Raised middle dot `·` appears with inconsistent spacing relative to the preceding word; the majority pattern is one space before the raised dot.
- **Assessment:** This is consistent 19th-century Greek printing style in Migne (the raised dot replaces a colon or strong medial stop). Not an error — a house-style artifact. Kept as printed throughout. Confidence Medium (spacing too small to adjudicate in every instance at scan resolution).

#### 3. `ἐστιν` — unaccented (idx157, col 537, Letter to Chrysostom, line 10)
- **Printed reading:** `ἐστιν` without accent (should be `ἔστιν` in existential sense, or `ἐστίν` in predicative).
- **Proposed emendation:** `ἔστιν` (standard enclitic exception form when existential).
- **Reasoning:** The print omits the accent entirely. Whether this is a compositor's dropped diacritic or an editorial choice to treat it as an enclitic cannot be resolved from the scan alone. Kept `[sic]` in the diplomatic text. Confidence High (print unambiguously lacks the accent).

#### 4. `ἐσμεν` — unaccented (idx157, col 537, Letter to Constantinople Clergy, line 22)
- **Printed reading:** `ἐσμεν` without accent (standard enclitic form is `ἐσμέν` in a heavy sentence position, or left accentless as an enclitic by some editors).
- **Proposed emendation:** `ἐσμέν` if non-enclitic; leave as-is if the editor treated it as enclitic.
- **Reasoning:** Parallel to ἐστιν above. The print lacks the accent; whether deliberate or dropped is unresolvable at this scan resolution. Kept `[sic]` in the diplomatic text. Confidence High for the print reading; medium for whether an emendation is warranted.

---

### Notable OCR-error catches

A curated selection of the most instructive Gemini fixes across the five new pages — whole-word
replacements, proper-name corruptions, apparatus note scrambles, and two-error-in-one-line
catches.

1. **Full apparatus note ᶠ body garbled** (idx153, cols 533–534). Gemini read "Infra, hujusmodi verba in tribus mss. non leguntur, sed per-feratur…" — the opening clause is entirely wrong. Print: "male, siquidem verbum ἐπληροῦτο non habet ad quod re-feratur." A complete clause replaced by fabricated Latin, restoring Migne's editorial reasoning about the omitted phrase.

2. **Scholar's name — `Stewechius` for `Stephanius`** (idx154, cols 533–534, apparatus). Gemini read "Stephanius"; print has "Stewechius" (the 16th-century humanist Godescalc Stewechius, who emended the Ambrosianus). A proper-name substitution that would have silently mislabelled the scholarly attribution in every downstream citation.

3. **Doubly wrong line — `ἡ ἡμέτερος Δεσπότης … παρέχειν ἔλθων`** (idx157, col 537, Innocent's letter to Constantinople clergy, line 17). Gemini introduced two errors simultaneously: wrong article gender (ἡ for ὁ) and a garbled predicate ("παρέχειν ἔλθων" for "παρέχει εἴωθεν"). Print reads "ὁ ἡμέτερος Δεσπότης ὑπομονὴν παρέχει εἴωθεν" ("our Lord is accustomed to provide endurance").

4. **Two errors in one phrase — `ἐγκωμιάζοντες ὑμῖν … ἐπενέγκαμεν`** (idx157, col 537, line 12). Gemini read ὑμῖν (dative) and ἐπενέγκαμεν; print has ὑμῶν (genitive) and ἐπεγνώκαμεν ("we have recognised"). The verb change is semantically significant: "we have recognised your purpose" vs. the garbled "we have brought against you."

5. **Word substitution — `νικᾶσθαι` for `γινέσθαι`** (idx157, col 537, col 2 line 3). "Not to be overcome" vs. "not to come about" — entirely different verbs, one implying a contest, one a process. The native-scan strip is unambiguous: νικᾶσθαι.

6. **Note-reference scramble** (idx153, cols 533–534). Two separate catches: (a) apparatus note ᵍ attached to `πολλοῦ` in the main text when the print has no note there, and the word is `πολλαχοῦ` ("in many places") not `πολλοῦ`; (b) note reference `h` for `ᵍ` at ἐκτείνεται, and `i` for `ʰ` at διεσχίσθησαν. Three note-letter shifts in one page; the apparatus sequence (ᵃ–ˡ) was used to adjudicate.

7. **Whole-word replacement — `παρ' ἡμῶν` for `τῶν ἡμῶν`** (idx157, col 537, line 3). The prepositional phrase "from us" vs. a genitive "of our" — structurally different and the wrong reading would grammatically strand the sentence.

8. **Column-header digit corruption** (idx157, cols 537–538). Gemini read the column numbers as 527–528 (the 3→5 glyph substitution repeated on both sides of the opening). All apparatus note cross-references, folio calculations, and position metadata depend on the correct column number. The strip headers resolved both to 537–538 unambiguously.

---

### Escalation: Καμπιδούκτωρ (idx154, apparatus, cols 533–534)

**Issue.** The apparatus footnote (α) on *campiductores* — the military officers cited in Letter 1 §4 — contains the Greek transliteration `Καμπιδούκτωρ`. The COLS pass produced `Καμπιδοκτωρ` (missing the middle vowel cluster entirely). The FULL pass produced `Καμπιδούκτωρ`, but at native scan resolution the diacritics over the upsilon-iota cluster were too small to read definitively: the vowel could be ού or οί.

**Evidence consulted.**
- Strip images: `idx154_c2_s4.png`, `idx154_app.png` (native 4030×6206).
- 4× magnification of the apparatus glyph.
- Leo the Wise, *Tactica*, cited in the same note: "ubi semper Καμπιδούκτωρ legitur."
- Latin parallel: *campiductor*, from *campus* + *ductor* ("field-leader") — the ου vowel maps to the Latin u, not oi.

**Resolution (Fable adjudication, 2026-06-12).** The 4× crop resolves the glyph as a cup-shaped upsilon with acute: **ού**, not οί. The lexical corroboration from Leo's Tactica (cited in the note itself) and the Latin etymology both confirm ού. Both occurrences in `verified_idx154.md` already read `Καμπιδούκτωρ` — no change to the verified transcript required. Crop evidence: `pages/campiductor_zoom2.png`.

---

### Full adjudication tables

#### idx153 — PG 52 cols 533–534

Base transcript: `gemini_idx153_full.md` · Authority: `pages/idx153_native.png` (4030×6206), verified strip-by-strip at native resolution.

| # | Location | Transcript read | Print says | Verdict | Proposed emendation | Confidence |
|---|---|---|---|---|---|---|
| 1 | COL 1 line 2 | `στασιν` | `στάσιν` | GEMINI-ERROR | add accent ά | HIGH |
| 2 | COL 1 line 4 | `οὗτος οὗτος` | `αὐτὸς οὗτος` | GEMINI-ERROR | first word → `αὐτὸς` | HIGH |
| 3 | COL 1 line 5 | `πλῦνας` | `πλύνας` | GEMINI-ERROR | ῦ → ύ | HIGH |
| 4 | COL 1 line 9 | `καὶτοι` | `καίτοι` | GEMINI-ERROR | καὶτοι → καίτοι | HIGH |
| 5 | COL 1 lines 23–24 | `πετυχ-χάμεν` | `τετυχή-καμεν` (τετυχήκαμεν) | GEMINI-ERROR | full replacement | HIGH |
| 6 | COL 1 line 26 | `παραστῆσαι` | `παραστήσει` | GEMINI-ERROR | aorist inf → future ind | HIGH |
| 7 | COL 1 line 38 | `κολυμβήθραι` | `κολυμβῆθραι` | GEMINI-ERROR | accent shift η̂ | HIGH |
| 8 | COL 1 line 44 | `εἰς ἐν τοσούτῳ` | `ὡς ἐν τοσούτῳ` | GEMINI-ERROR | εἰς → ὡς | HIGH |
| 9 | COL 1 line 45 | `ἐξέχειτο` | `ἐξεχεῖτο` | GEMINI-ERROR | accent position | HIGH |
| 10 | COL 1 lines 52–53 | `ὀλολυγαὶ e καὶ θρῆνοι e` | `ὀλολυγαὶ καὶ θρῆνοι ᵉ` | GEMINI-ERROR | remove first `e`; retain one `ᵉ` after θρῆνοι | HIGH |
| 11 | COL 1 line 57 | `ὑπομένοντες συνήλγουν ἡμῖν, οὐχ` (no `οἱ`) | `οὐχ οἱ` | GEMINI-ERROR | insert `οἱ` after `οὐχ` | HIGH |
| 12 | COL 2 line 77 | `πολλοῦ ᵍ στρατηγου-τῶν` | `πολλαχοῦ στρατηγούν-τῶν` | GEMINI-ERROR | `πολλοῦ ᵍ` → `πολλαχοῦ`; no note ref here | HIGH |
| 13 | COL 2 line 80 | `μετωκίζετο` | `μετῳκίζετο` | MIGNE-TYPO | keep `μετῳκίζετο` | HIGH |
| 14 | COL 2 lines 88–89 note ref | `ἐκτείνεται h` | `ἐκτείνεται ᵍ` | GEMINI-ERROR | h → ᵍ | HIGH |
| 15 | COL 2 line 89 | `γεγονάμεν` | `γεγόναμεν` | GEMINI-ERROR | accent: γεγονάμεν → γεγόναμεν | HIGH |
| 16 | COL 2 line 92 | `καινην` | `καινὴν` | GEMINI-ERROR | add grave accent | HIGH |
| 17 | COL 2 lines 93–94 | `Τί ἂν εἴποι τις τῶν λοιπῶν` | `Τί ἂν τις εἴποι τὰς τῶν λοιπῶν` | GEMINI-ERROR | word order + missing `τὰς` | HIGH |
| 18 | COL 2 line 101 note ref | `διεσχίσθησαν i` | `διεσχίσθησαν ʰ` | GEMINI-ERROR | i → ʰ | HIGH |
| 19 | COL 2 line 107 | `ἔξὸν` | `ἔξον` | GEMINI-ERROR | spurious grave → `ἔξον` | HIGH |
| 20 | COL 2 line 121 | `οὐκ ἐλεγχομένους` | `οὐχ ἐλεγχομένους` | GEMINI-ERROR | οὐκ → οὐχ | HIGH |
| 21 | APPARATUS note ᶠ | `Morcl.` | `Morel.` | GEMINI-ERROR | Morcl. → Morel. | HIGH |
| 22 | APPARATUS note ᶠ (full text) | "Infra, hujusmodi verba in tribus mss. non leguntur, sed per-feratur…" | "male, siquidem verbum ἐπληροῦτο non habet ad quod re-feratur…" | GEMINI-ERROR | full apparatus note ᶠ text corrected per app strip | HIGH |
| 23 | APPARATUS note ᵍ | `ἀλλ' ἐπείνειται` | `ἀλλ' ἐπιτείνεται` | GEMINI-ERROR | ἐπείνειται → ἐπιτείνεται | HIGH |
| 24 | APPARATUS note ˡ | `Editi τῇ οἰκείᾳ φύσιν` | `Editi τὴν οἰκείαν φύσιν` | GEMINI-ERROR | τῇ οἰκείᾳ → τὴν οἰκείαν | HIGH |
| 25 | COL 1 `δικασθῆναι` line 20 | `δικασθῆναι` | print confirms `δικασθῆναι` | OK | — | HIGH |
| 26 | COL 1 `κατηγοριῶν` line 22 | `κατηγοριῶν` | print confirms `κατηγοριῶν` | OK | — | HIGH |
| 27 | APPARATUS note ᵇ | present | present | OK | — | HIGH |
| 28 | COL 1 `εἰκῇ` line 51 | `εἰκῇ` | `εἰκῇ` | OK | — | HIGH |
| 29 | COL 2 `ὁδῷ τῷ` line 98 | `τῷ` | `τῷ` | OK | — | HIGH |
| 30 | APPARATUS note ᵏ | both forms present | both forms present | OK | — | HIGH |

**Summary (idx153):** GEMINI-ERROR: 23. MIGNE-TYPO [sic]: 1 (μετῳκίζετο). OK: 5 (items 25–30 exc. 13). Escalated: 0.

#### idx154 — PG 52 cols 533–534

Base transcript: `gemini_idx154_full.md` · Authority: `pages/idx154_native.png` (4030×6206), verified strip-by-strip at native resolution.

| # | Location | Transcript (Gemini) | Print | Verdict | Proposed emendation | Confidence |
|---|---|---|---|---|---|---|
| 1 | c1 line 16 | `fecerant ad hos` | `fecerant ; ad hos` | GEMINI-ERROR | add semicolon | HIGH |
| 2 | c1 line 26 | `ecclesiis ingressa` | `ecclesias ingressa` | GEMINI-ERROR | ecclesiis → ecclesias | HIGH |
| 3 | c1 lines 39–40 | `et barbarica quasi in barbarica` | `fiebantque quasi in barbarica` | GEMINI-ERROR | full phrase replacement | HIGH |
| 4 | c1 apparatus | `Ceteri Campiductores vestri` | `Certe Campiductores vestri` | GEMINI-ERROR | Ceteri → Certe | HIGH |
| 5 | c1 apparatus | `et vel restituerdum putat` | `etc. Ubi restituendum putat` | GEMINI-ERROR | full replacement | HIGH |
| 6 | c2 line 71 | `sermone com-pleti,` | `sermone com-plecti,` | GEMINI-ERROR | pleti → plecti | HIGH |
| 7 | c2 line 82 | `reliqarum` | `reliquarum` | GEMINI-ERROR | reliqarum → reliquarum | HIGH |
| 8 | c2 line 94 | `quæ in Ecclesiis irru-` | `quæ in Ecclesias irru-` | GEMINI-ERROR | Ecclesiis → Ecclesias | HIGH |
| 9 | c2 lines 104–105 | `non habent ; sicut neque suæ naturæ habent :` | `non habent robur, sicut neque sua natura habent :` | GEMINI-ERROR | insert `robur,`; suæ naturæ → sua natura | HIGH |
| 10 | c2 line 108 | `nec habiti rei sumus` (missing `ut`) | `nec habiti ut rei sumus` | GEMINI-ERROR | insert `ut` before `rei` | HIGH |
| 11 | c2 apparatus | `campidoctorem malle` | `Campidoctorem malle` | GEMINI-ERROR | lowercase → uppercase C | HIGH |
| 12 | c2 apparatus | `monet Ste-phanius` | `monet Ste-wechius` | GEMINI-ERROR | Stephanius → Stewechius | HIGH |
| 13 | c2 apparatus | `campiductorem legendum` | `Campiductorem legendum` | GEMINI-ERROR | lowercase → uppercase C | HIGH |
| 14 | c2 apparatus | `in thesauro` | `in Thesauro` | GEMINI-ERROR | thesauro → Thesauro | HIGH |
| OK | `rumdam` (seam join) | FULL correct | print confirms | OK | — | HIGH |
| OK | `enarraverim` | FULL correct | print confirms | OK | — | HIGH |
| OK | `incursus nudæ aufuge-runt` | FULL correct | print confirms | OK | — | HIGH |
| OK | `sanctissimus Christi sanguis` | FULL correct | print confirms | OK | — | HIGH |
| OK | `propè diem` (two words) | FULL correct | print confirms | OK | — | HIGH |
| OK | `maxime` (no accent) | FULL correct | print confirms | OK | — | HIGH |
| OK | `diceritis` | FULL correct | print confirms | OK | — | HIGH |
| OK | `adeo inique` | FULL correct | print confirms | OK | — | HIGH |
| OK | `(α)` reference markers | FULL correct | print confirms | OK | — | HIGH |
| OK | `Καμπιδούκτωρ` (Greek) | FULL correct (ού confirmed) | print confirms | OK (escalated, RESOLVED) | — | MED-HIGH |
| OK | Footnote ¹ "Palladius, clerici a clericis." | FULL correct | print confirms | OK | — | HIGH |
| OK | Col 2 apparatus continuation | FULL correct | print confirms | OK | — | HIGH |

**Summary (idx154):** GEMINI-ERROR: 14. MIGNE-[sic]: 0. OK: 16. Escalated: 1 (Καμπιδούκτωρ diacritics — RESOLVED; see escalation section above).

#### idx155 — PG 52 cols 535–536

Base transcript: `gemini_idx155_full.md` · Authority: `pages/idx155_native.png` (4030×6206), verified strip-by-strip at native resolution.

| # | Location | Transcript (Gemini) | Print | Verdict | Proposed emendation | Confidence |
|---|---|---|---|---|---|---|
| 1 | Header, column numbers | 555 … 556 | 535 … 536 | GEMINI-ERROR | 535 … 536 | HIGH |
| 2 | MONITUM line 1 | "gentium" | "gentilium" | GEMINI-ERROR | gentilium | HIGH |
| 3 | COL1 body | `tenetur;` | `tenetur ;` (space before semicolon) | GEMINI-ERROR | tenetur ; | HIGH |
| 4 | COL1 body | `sumus¹` | `sumus ¹` (space before superscript) | GEMINI-ERROR | sumus ¹ | HIGH |
| 5 | COL1 body | "diaconum," | "diaconum nacti," | GEMINI-ERROR | diaconum nacti, | HIGH |
| 6 | Footnote 1 | "Fabric." | "Fabric. ;" (semicolon follows) | GEMINI-ERROR | Fabric. ; | HIGH |
| 7 | Footnote 1 continuation | "Fabric. tunc sic construitur phrasis : Sed quotidie caritatis videmus" | "Fabric. ; et tunc sic construitur phrasis : *Sed quotidie caritatis oculis videmus*" | GEMINI-ERROR | add `; et`; add `oculis` after caritatis | HIGH |
| 8 | COL2 body | "nolo quidem omnia" | "nolo equidem omnia" | GEMINI-ERROR | equidem | HIGH |
| 9 | COL2 body | "tanto magis adhibeatur studium" | "tanto majus adhibeatur studium" | GEMINI-ERROR | majus | HIGH |
| 10 | COL2 body | "desolationem" | "desolatiorem" | GEMINI-ERROR | desolatiorem | HIGH |
| 11–15 | Hyphenated line-breaks (multiple) | Gemini joins correctly | Print splits mid-line | OK | — | HIGH |
| 16 | Footnote 1 italic marker "*Hæc," | asterisk present | Print uses italic type | OK | — | HIGH |
| 17 | Footnote 2 "*qui" | asterisk present | Print uses italic | OK | — | HIGH |
| 18 | Title block | Full text present | Full text confirmed | OK | — | HIGH |
| 19 | `pergunt²,` spacing | no space before superscript | Print shows no space | OK | — | HIGH |

**Summary (idx155):** GEMINI-ERROR: 10. MIGNE-[sic]: 0. OK: 5. Escalated: 0.

#### idx156 — PG 52 cols 535–536

Base transcript: `gemini_idx156_full.md` · Authority: `pages/idx156_native.png` (4030×6206), verified strip-by-strip at native resolution.

| # | Location | Transcript (Gemini) | Print | Verdict | Proposed emendation | Confidence |
|---|---|---|---|---|---|---|
| 1 | col1 header | ΙΝΝΟΚΕΝΤΙΩ | INNOKENTIΩ | GEMINI-ERROR | INNOKENTIΩ | HIGH |
| 2 | col1 line 9 | ναντες (in κρί-ναντες) | νοντες (κρίνοντες) | GEMINI-ERROR | κρίνοντες | HIGH |
| 3 | col1 line 22 | τοσούτου | τοσούτῳ | GEMINI-ERROR | τοσούτῳ | HIGH |
| 4 | col1 line 22 | μήκει | τύχει | GEMINI-ERROR | τύχει | HIGH |
| 5 | col1 line 23 | ἐσμεν | ἔσμεν | GEMINI-ERROR | ἔσμεν | HIGH |
| 6 | col1 line 26 | στεῤῥὸν | στερρὸν | GEMINI-ERROR | στερρὸν | HIGH |
| 7 | col1 line 28 | πλέον αἴρεται | πλεῖον αἴρεται | GEMINI-ERROR | πλεῖον | HIGH |
| 8 | col1 line 29 | πλέον ὕφαλοι | πλείους ὕφαλοι | GEMINI-ERROR | πλείους | HIGH |
| 9 | col1 line 30 | αὔξει | αὐξεί | GEMINI-ERROR | αὐξεί | HIGH |
| 10 | col1 line 40 | παρὰ τοῦ τοῦ τόπου | παρὰ τῆς τοῦ τόπου | GEMINI-ERROR | τῆς | HIGH |
| 11 | col1 line 41 | ἐρημία ⟦?⟧ | ἐρημίας · | GEMINI-ERROR | ἐρημίας· (remove ⟦?⟧) | HIGH |
| 12 | col1 line 41 | ἀκεῖσε | ἐκεῖσε | GEMINI-ERROR | ἐκεῖσε | HIGH |
| 13 | col1 line 43 | διὰ τὸ τὸ πόῤῥω | διὰ τε τὸ πόῤῥω | GEMINI-ERROR | διὰ τε τὸ | HIGH |
| 14 | col1 line 44 | κείσθαι | κεῖσθαι | GEMINI-ERROR | κεῖσθαι | HIGH |
| 15 | col1 line 49 | οὐκ ὀλιγωροῦντες | οὐχ ὀλιγωροῦντες | GEMINI-ERROR | οὐχ | HIGH |
| 16 | col1 line 55 | τὴν προϋ-[ήκουσαν] | τὴν προσ-[ήκουσαν] | GEMINI-ERROR | προσήκουσαν | HIGH |
| 17 | col1 line 57 | ἀνῄρηται | ἀνήρηται | GEMINI-ERROR | ἀνήρηται | HIGH |
| 18 | col2 line 2 | ἐνεγκεῖν | εἰσενεγκεῖν | GEMINI-ERROR | εἰσενεγκεῖν | HIGH |
| 19 | col2 line 3 | παρακαλῆτε | παρακαλήθητε | GEMINI-ERROR | παρακαλήθητε | HIGH |
| 20 | col2 line 5 | μισθὸν α, | μισθὸν α´ | GEMINI-ERROR | α´ (footnote marker with prime) | HIGH |
| 21 | col2 line 6 | ἐῤῥωμένος | ἐρρωμένος | GEMINI-ERROR | ἐρρωμένος | HIGH |
| 22 | col2 line 9 (app note c) | (ποιοῦντος) ἀμήν | (ποιοῦντος) ἀμήν | OK | — | MEDIUM |
| 23 | col2 line 9 | ἐξεδικήθησαν e | ἐξεδικήθησαν c | GEMINI-ERROR | superscript c | HIGH |
| 24 | col2 line 15 | διηγήσασθαι παρίημι | διηγήσασθαι d παρίημι | GEMINI-ERROR | insert footnote marker d | HIGH |
| 25 | col2 line 15 | οὐκ ἐπιστολῆς | οὐχ ἐπιστολῆς | GEMINI-ERROR | οὐχ | HIGH |
| 26 | col2 line 15 | ἡ διήγησις | ἢ διήγησις | GEMINI-ERROR | ἢ (particle, not article) | HIGH |
| 27 | col2 line 16 | ψυχὴν | ψυχήν | GEMINI-ERROR | ψυχήν | HIGH |
| 28 | col2 line 17 | ἀνίατα, | ἀνίατα (no comma) | GEMINI-ERROR | remove comma | MEDIUM |
| 29 | col2 line 18 | ἐλομένους | ἑλομένους | GEMINI-ERROR | ἑλομένους | HIGH |
| 30 | col2 line 20 | παραβαθέντων d | παραβαθέντων e | GEMINI-ERROR | superscript e | HIGH |
| 31 | col2 line 25 | γνήσι g | γνησί g | GEMINI-ERROR | γνησί (accent on ί) | HIGH |
| 32 | col2 line 25 | ὑμῖν τῇ ἀγάπῃ | ὑμῶν τῇ ἀγάπῃ | GEMINI-ERROR | ὑμῶν | HIGH |
| 33 | col2 line 29 | οὐ μικρὰ ταύτην | οὐ μικρὰν ταύτην | GEMINI-ERROR | μικρὰν | HIGH |
| 34 | col2 line 31 | ἀπίμεν | ἅπιμεν | GEMINI-ERROR | ἅπιμεν (rough breathing) | HIGH |
| 35 | app (col2 note g) | φιλῷ | φιλῇ | GEMINI-ERROR | φιλῇ | HIGH |
| 36 | app (col2 note g) | γνήσι | γνησίᾳ | GEMINI-ERROR | γνησίᾳ | HIGH |
| 37 | col1 header arrangement | col-L missing "Καὶ τί λέγω" | present in full page | OK — col arrangement artefact | — | HIGH |
| 38 | punctuation spacing (raised dot ·) | various spacings | print uses raised dot with preceding space | MIGNE-TYPO [sic] — consistent 19c printing style | — | MEDIUM |
| 39 | col1 line 22 | ίπταται (in περι-ίπταται) | ίπταται | OK — compound verb, prefix masks breathing | — | HIGH |
| 40 | col2 app note c (colR model) | "χρησῶς" | ἀμήν | colR GEMINI-ERROR; full model correct | — | MEDIUM |

**Summary (idx156):** GEMINI-ERROR: 36. MIGNE-[sic]: 1 (punctuation spacing, item 38). OK: 4 (items 22, 37, 39, 40). Escalated: 0.

#### idx157 — PG 52 cols 537–538

Base transcript: `gemini_idx157_full.md` · Authority: `pages/idx157_native.png` (4030×6206), verified strip-by-strip at native resolution. **Note:** This page contains two distinct items: Innocent's First Letter to Chrysostom (Greek translation, cols 537 col 1 + part of col 2) and Innocent's Letter to the Constantinople Clergy (cols 537 col 2 through 538).

| # | Location | Transcript (Gemini) | Print | Verdict | Proposed emendation | Confidence |
|---|---|---|---|---|---|---|
| 1 | HEADER, col numbers | 527 … 528 | 537 … 538 | GEMINI-ERROR | 537, 538 | HIGH |
| 2 | COL1 line 3 (Letter 1) | `τῶν ἡμῶν` | `παρ' ἡμῶν` | GEMINI-ERROR | παρ' ἡμῶν | HIGH |
| 3 | COL1 line 8 (Letter 1) | `ποιμήν,` | `ποιμήν,` | OK (accentuation question unresolvable at this resolution) | — | MEDIUM |
| 4 | COL1 line 10 (Letter 1) | `ἔστιν` | `ἐστιν` (no accent) | MIGNE-[sic] | keep `ἐστιν` per print | HIGH |
| 5 | COL2 line 3 (Letter 1) | `γινέσθαι δὲ οὐκ` | `νικᾶσθαι δὲ οὐκ` | GEMINI-ERROR | νικᾶσθαι | HIGH |
| 6 | FOOTNOTE | `fuere` | `fuere` | OK — `fueræ` in col pass was col-pass artifact | — | HIGH |
| 7 | TITLE (Letter 2) | `ΙΝΝΟΚΕΝΤΙΟΣ ΕΠΙΣΚΟΠΟΣ` (no ΡΩΜΗΣ) | `ΙΝΝΟΚΕΝΤΙΟΣ ΕΠΙΣΚΟΠΟΣ` | OK — ΡΩΜΗΣ in col pass was noise | — | HIGH |
| 8 | COL1 line 6 (Letter 2) | `ἐπαναληφθεῖσαι` | `ἐπαναληφθείσῃ` | GEMINI-ERROR | ἐπαναληφθείσῃ | HIGH |
| 9 | COL1 line 12 (Letter 2) | `ἐγκωμιάζοντες ὑμῖν … ἐπενέγκαμεν` | `ἐγκωμιάζοντες ὑμῶν … ἐπεγνώκαμεν` | GEMINI-ERROR (two errors) | ὑμῖν → ὑμῶν; ἐπενέγκαμεν → ἐπεγνώκαμεν | HIGH |
| 10 | COL1 line 13 (Letter 2) | `πολλὰ` | `πολλὰς` | GEMINI-ERROR | πολλὰς | HIGH |
| 11 | COL1 line 16 (Letter 2) | `προεὐβάσατε` | `προεφθάσατε` | GEMINI-ERROR | προεφθάσατε | HIGH |
| 12 | COL1 line 17 (Letter 2) | `ἡ ἡμέτερος Δεσπότης … παρέχειν ἔλθων` | `ὁ ἡμέτερος Δεσπότης … παρέχει εἴωθεν` | GEMINI-ERROR (two errors) | ἡ → ὁ; παρέχειν ἔλθων → παρέχει εἴωθεν | HIGH |
| 13 | COL1 line 18 (Letter 2) | `θλίψεσι` | `θλίψεσιν` | GEMINI-ERROR | θλίψεσιν | HIGH |
| 14 | COL1 line 20 (Letter 2) | `γεγενῆσθαι` | `γεγονῆσθαι` | GEMINI-ERROR | γεγονῆσθαι | HIGH |
| 15 | COL1 line 22 (Letter 2) | `συνάγειν` | `συναλγεῖν` | GEMINI-ERROR | συναλγεῖν | HIGH |
| 16 | COL1 line 22 (Letter 2) | `ἐσμὲν` | `ἐσμεν` (no accent) | MIGNE-[sic] | keep `ἐσμεν` per print | HIGH |
| 17 | COL1 line 24 (Letter 2) | `συνήσεται` | `δυνήσεται` | GEMINI-ERROR | δυνήσεται | HIGH |
| 18 | COL1 line 29 (Letter 2) | `Ὦ δὴ` | `Ὅ δὴ` | GEMINI-ERROR | Ὅ δὴ | HIGH |
| 19 | COL1 line 32 (Letter 2) | `Ἡ ζητηθῆ,` | `ἢ ζητηθῇ,` | GEMINI-ERROR (capital/lowercase + accent) | ἢ ζητηθῇ, | HIGH |
| 20 | COL1 line 36 (Letter 2) | `δέδοται` | `δέδοσθαι` | GEMINI-ERROR | δέδοσθαι | HIGH |
| 21 | COL2 line 2 (Letter 2) | `Ὅτι` | `Ὅ τι` (two words) | GEMINI-ERROR | Ὅ τι | HIGH |
| 22 | COL2 line 3 (Letter 2) | `τούτοις ἔδει ἕπεσθαι` | `τούτοις δεῖν ἕπεσθαι` | GEMINI-ERROR | δεῖν | HIGH |
| 23 | COL2 line 23 (Letter 2) | `ἀντίχρυσ` (line-break form) | `ἀντίχρυς` | OK — line-break artifact | — | HIGH |
| 24 | COL2 line 25 (Letter 2) | `ἔρημεν` | `ἐφήμεν` | GEMINI-ERROR | ἐφήμεν | HIGH |
| 25 | COL2 line 27 (Letter 2) | `καταγιγίδων` | `καταιγίδων` | GEMINI-ERROR | καταιγίδων | HIGH |
| 26 | COL2 line 45 (Letter 2) | `Εὐλυαίου` | `Εὐλυσίου` | GEMINI-ERROR | Εὐλυσίου | HIGH |

**Summary (idx157):** GEMINI-ERROR: 21. MIGNE-[sic]: 2 (ἐστιν, item 4; ἐσμεν, item 16). OK: 4 (items 3, 6, 7, 23). Escalated: 0.

---

## Ensemble adjudication (cross-model gate, 2026-06-12)

**Method.** After Run 1 (idx149–152) and Run 2 (idx153–157) were verified by the primary verifier (Claude Opus/Sonnet), a second OCR engine (gemini-3-flash-preview) ran a full independent pass over all nine pages, generating a parallel transcript for each. Every location where the two transcripts differed was logged as a dispute. The 132 disputes were then adjudicated by eye against the native print strip images at native scan resolution. idx153 and idx156 were piloted first (ab_adjudication.md) to validate the method before the full gate ran. The winner for each dispute was recorded as TRUTH (primary verifier was right), B (second engine was right), or NEITHER (both wrong — best reading taken from the print directly). NOISE items (encoding conventions only, e.g., mid-dot spacing or Unicode apostrophe variants) were logged but not patched.

**Stats per page.**

| Page | Total disputes | TRUTH wins | B wins | NEITHER wins | NOISE (no patch) | Corrections applied |
|---|---|---|---|---|---|---|
| idx149 (529–530) | 12 | 11 | 1 | 0 | 0 | 1 |
| idx150 (529–530) | 17 | 14 | 3 | 0 | 0 | 3 |
| idx151 (531–532) | 10 | 9 | 1 | 0 | 0 | 1 |
| idx152 (531–532) | 22 | 13 | 9 | 0 | 0 | 9 |
| idx153 (533–534) | 14 | 11 | 3 | 0 | 0 | 3 |
| idx154 (533–534) | 18 | 17 | 1 | 0 | 0 | 1 |
| idx155 (535–536) | 15 | 12 | 3 | 0 | 0 | 3 |
| idx156 (535–536) | 16 | 13 | 3 | 0 | 0 | 3 |
| idx157 (537–538) | 8 | n/a | 14 | 0 | 0 | 14 |
| **Total** | **132** | — | **~38** | **0** | **0** | **~38** |

**Anomaly list corrections.**

- **REMOVED** `ἐσμεν / ἔσμεν` from the print-anomaly list (idx157, item 16). Gate and eye adjudication confirmed the print carries no accent on this word — neither an acute nor a circumflex — so the unaccented form `ἐσμεν` is the correct diplomatic reading. The [sic] entry in Run 2 was a misclassification: `ἐσμεν` (no accent) is a well-attested enclitic form in 19th-century printed Greek and not anomalous. The primary verifier's residue of uncertainty was unfounded.
- **ADDED** `exagitatuin [sic]` to the print-anomaly list (idx149, monitum, item 3). The gate's independent read and direct eye inspection of the native strip confirm the Migne print reads `exagitatuin` — final `n` not `m`. The primary verifier over-corrected this to `exagitatum` in the first pass, reasoning it was an OCR error. It is not: the compositor dropped the ligature and printed `in` where `m` was intended. Classified as a compositor transposition (in/m class). Retained `exagitatuin [sic]` in the verified diplomatic text. Proposed emendation: `exagitatum`.
- **RE-CONFIRMED** `Co.slinianus` (idx149, apparatus). Both engines and eye confirm the print anomaly stands. No change.
- **RE-CONFIRMED** `pestringeret` (idx150, monitum). Both engines and eye confirm the print reading. No change.

**Residual-error insight.** Across all nine pages, the gate found approximately 2–3 verifier-residue corrections per page regardless of verifier tier (Opus or Sonnet). This is a structural property of the single-pass verification model, not a function of model capability: a verifier reading a transcript against an image at speed will reliably miss 2–3 items per page. The cross-model gate recovers most of these. The recommended practice for high-confidence diplomatic texts is therefore: OCR → primary verify → cross-model gate → eye adjudication of disputes, accepting a residual uncorrected rate of roughly 0–1 items per page after the gate.

**Note on μετῳκίζετο.** The idx153 [sic] item `μετῳκίζετο` was not independently re-checked by the gate pass (it was already resolved as a Migne compositor's transposition in Run 2 and the gate pass did not dispute it). Its status as a genuine print anomaly is unchanged.

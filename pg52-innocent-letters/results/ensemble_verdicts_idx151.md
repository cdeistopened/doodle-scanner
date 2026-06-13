# Ensemble verdicts idx151

Disputes: 6 total — 3 structural/noise, 3 real content disputes.

---

## Structural/ordering items — NOISE (no image adjudication needed)

| ctx | TRUTH | B | verdict |
|-----|-------|---|---------|
| …531 | — | ∅ | NOISE — header column-number placement difference; structural only |
| …CHRYSOST. ARCHIEP. CONSTANTINOP. | — | ∅ | NOISE — running-header formatting; structural only |
| …CONSTANTINOP. — 532 | [COL. 531] | ∅ | NOISE — column-break marker convention; structural only |

---

## Real content disputes

### D1 — `exhibere :` vs `exhibere;`

**Context:** col. 531 line 13 — "quæ par erat nos illis exhibere : ac fre-"

**Image evidence:** Crop `/tmp/idx151_exhibere2.png` shows plainly: `pendimus, quæ par erat nos illis exhibere: ac fre-`

**Reading:** The print has a **colon** (`:`) after `exhibere` with a space before `ac`. No semicolon.

| PRINT | `exhibere :` |
|-------|-------------|
| Winner | **TRUTH** |
| Confidence | HIGH |

---

### D2 — `eum` vs `cum`

**Context:** col. 531 line 42 — "sunt facta, misimus ad eum episcopos"

**Image evidence:** Crop `/tmp/idx151_misimus4.png` shows: `quæ prius ac postea sunt facta, misimus ad eum epi-scoPos`

**Reading:** The initial character of the disputed word is clearly `e` (open curved letter), not `c`. The word is **`eum`**.

| PRINT | `eum` |
|-------|-------|
| Winner | **TRUTH** |
| Confidence | HIGH |

---

### D3 — `causa?` vs `causa ?`

**Context:** col. 532 line 114 — "Cur et qua de causa? Eo quod ingressi mox"

**Image evidence:** Crop `/tmp/idx151_causa_zoom.png` shows clearly: `Cur et qua de causa ? Eo quod ingressi mox`

**Reading:** There IS a **space before the question mark**: `causa ?`.

| PRINT | `causa ?` |
|-------|----------|
| Winner | **B** |
| Confidence | HIGH |

---

## [sic] check

No `[sic]` tokens in verified_idx151.md. Nothing to adjudicate.

---

## Summary

| # | ctx | PRINT | Winner |
|---|-----|-------|--------|
| 1 | …531 | — (structural) | NOISE |
| 2 | …CHRYSOST. ARCHIEP. CONSTANTINOP. | — (structural) | NOISE |
| 3 | …CONSTANTINOP. — 532 | [COL. 531] (structural) | NOISE |
| 4 | …erat nos illis | `exhibere :` | TRUTH |
| 5 | …facta, misimus ad | `eum` | TRUTH |
| 6 | …et qua de | `causa ?` | B |

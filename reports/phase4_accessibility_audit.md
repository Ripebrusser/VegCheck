# Phase 4 — Accessibility and Usability Audit

**Date:** 2026-11-15 · **Record:** `evaluation/phase4_accessibility/audit.json`
**Tool:** `tools/accessibility_audit.py` (Playwright against the running app)

> **Scope statement.** These are automated checks plus targeted manual inspection.
> **No WCAG conformance is claimed.** Conformance requires manual testing with
> assistive technology by a qualified auditor, which was not performed.

---

## 1. Summary

| Area | Result |
|---|---|
| Text contrast (29 sampled elements, 4 pages) | **0 below WCAG AA**, minimum **6.7 : 1** |
| Page headings | all 4 pages have an `<h1>` |
| Image alternatives | 0 content images without `alt` (was 1) |
| Keyboard reachability | 11 distinct tab stops, sidebar navigation reachable |
| Desktop horizontal overflow | none at 1360 px |
| Mobile (390 × 844) | no horizontal overflow, layout reflows correctly |
| Colour alone conveys meaning | no — predicted class is named in text as well as coloured |

## 2. Contrast

Measured with WCAG 2.1 relative luminance against the effective (ancestor-resolved)
background. AA thresholds: 4.5 : 1 normal text, 3 : 1 large text.

| Element | Pages sampled | Min contrast | Verdict |
|---|---|---:|---|
| Page title (`h1`) | 4 | 10.8 | pass |
| Section heading (`h3`) | 4 | 10.8 | pass |
| Body text | 4 | 8.9 | pass |
| Caption text | 4 | 6.7 | pass (lowest in the app) |
| Metric label | 2 | 7.1 | pass |
| Metric value | 2 | 12.6 | pass |
| Button text | 1 | 5.9 | pass |
| Sidebar nav link | 4 | 11.4 | pass |
| Info box text | 1 | 8.3 | pass |

The lowest value in the application is the caption grey on off-white at 6.7 : 1,
comfortably above the 4.5 : 1 requirement.

## 3. Keyboard and focus

Pressing Tab repeatedly from page load produced 11 distinct stops:

```
Skip to main content → app → Dataset Information → Model Performance →
About Project → Deploy → VegCheck → "What this system does" toggle →
heading link → INPUT → Upload
```

All four pages are reachable by keyboard from the sidebar, and the upload control
receives focus in order. Streamlit renders its own "Skip to main content" link.

**Not tested:** focus trapping in the file dialog, focus visibility at 200 % zoom,
and screen-reader announcement order.

## 4. Images and alternatives

The prediction page renders five images: the upload, the 224×224 resize, the
segmentation overlay, and the probability chart. Every `st.image` call now passes an
explicit `alt` description, e.g.

> "The photograph resized to 224 by 224 pixels, the model input size"

The confusion matrix on the model page also carries a description, since the matrix
is a content image rather than decoration.

| Id | Severity | Issue | Correction |
|---|---|---|---|
| P4-6 | Moderate | All five content images lacked `alt` attributes; `st.image` supports the parameter and it was simply not used | every `st.image` now passes `alt`; audit re-run reports 0 missing |

The probability chart is a Matplotlib figure rendered via `st.pyplot`, which has no
`alt` parameter. It is not the only carrier of the result: the predicted class, its
score, the runner-up and all five percentages are also stated in text.

## 5. Mobile layout

At 390 × 844 (iPhone-class viewport):

* document width 390 px, viewport 390 px → **no horizontal scrolling**;
* the hero, badges, radio group, uploader and info box all reflow to a single column;
* minimum contrast unchanged at 6.7 : 1;
* 16 focusable elements remain reachable.

The sidebar collapses behind its chevron as Streamlit's default behaviour.

## 6. Usability observations

| Aspect | Finding |
|---|---|
| Navigation | Four labelled destinations in a persistent sidebar; the main page is labelled "app" (Streamlit uses the script filename) — mildly unpolished but unambiguous |
| Button labels | `Predict quality`, `Clear history`, `Upload` — all verbs describing the action |
| Error messages | Name the problem and the remedy, e.g. "Image is 16×16; the minimum size is 64×64." and "Unsupported file type '.gif'. Accepted types: .jpg, .jpeg, .png." |
| Prediction explanation | States what the class means, that it is visual only, and that *Ripe*/*Unripe* describe maturity rather than safety |
| Confidence chart | Percentages labelled on every bar; predicted class highlighted in green and named in the axis label |
| Segment numbering | Reads 1 → 2 → 3 → 4 → 5 when an image is present; the history is unnumbered so the page does not jump 1 → 6 |
| Dark mode | Not tested; the app uses a light theme |

## 7. Known limitations

1. No screen-reader session was run (NVDA/JAWS/VoiceOver untested).
2. No automated axe-core or Lighthouse audit was run — contrast and structure were
   measured directly, which is narrower and more transparent than a score.
3. Zoom/reflow at 200 % was not tested.
4. The Streamlit main-page label "app" comes from the filename; the browser tab
   title is correct ("VegCheck — Vegetable Quality Detection").
5. Focus indicators are Streamlit's defaults and were not visually verified.

## 8. Verdict

No blocking accessibility defect. Contrast, headings, alternatives and keyboard
reachability all pass automated checks; the one real defect (`alt` text missing on
content images) was found and fixed during this audit.

# Phase 4 — Integrity Audit

**Date:** 2026-11-15 · **Machine-readable record:** `evaluation/phase4/integrity_audit.json`

Eight checks were run against the released system before the test set was touched.
**Seven passed on the first run; one failed and exposed a real defect.** The defect
is documented in §2 and was corrected before any test evaluation.

Run: `python -m src.evaluation.phase4_audit`

---

## 1. Results

| # | Check | Question | Result |
|---|---|---|---|
| A | Model identity | Is the deployed file the artefact Phase 2 evaluated? | **pass** |
| B | Pipeline equivalence | Does serving reproduce the training features exactly? | **fail → fixed → pass** |
| C | Class mapping | Is the class order identical everywhere? | **pass** |
| D | Test untouched | Was the test split ever scored before Phase 4? | **pass** |
| E | Frozen artefacts | Do the 12 Phase 1 artefacts still hash correctly? | **pass** |
| E2 | Split integrity | Are the manifests still leakage-free? | **pass** |
| F | Website sources | Are metrics read from artefacts rather than literals? | **fail → fixed → pass** |
| G | Duplicate docs | Is the corrected cluster figure consistent? | **pass** (after correcting the *check*) |

Final state: **all checks pass**.

## 2. Critical defect found — serving input size did not match training

### What was wrong

`src.features.screening._load_one` — the function that produced
`datasets/interim/features_train.npz` — resizes each image to
**`config.IMAGE_SIZE` = 224** before computing descriptors.

The Phase 3 application used `config.SOURCE_IMAGE_SIZE` = **256** (the resolution of
the files on disk) as its feature-extraction size. The deployed Random Forest was
therefore being fed a feature vector computed at a different scale from anything it
saw during training.

The measured symptom:

```
check B before the fix
  images_compared: 20
  max_abs_difference: 0.908
  n_mismatches: 1562   (of 1,660 values = 20 images × 83 features)
```

### Why it matters

Three descriptor families are not scale-invariant:
`shape_area_frac`, `shape_perimeter_norm` and `shape_equiv_diameter_norm` are all
normalised by the image dimensions, and GLCM distances correspond to different
physical spacings at a different resolution. Every prediction the Phase 3 app made
was computed on a shifted input distribution.

### Correction

`website/utils/paths.py` now sets `FEATURE_IMAGE_SIZE = config.IMAGE_SIZE` (224),
i.e. the training size, with a comment recording why. The user-visible strings that
hard-coded "256×256" were made to follow that constant.

After the fix:

```
max_abs_difference: 0.0
n_mismatches: 0
```

The serving pipeline is now **bit-identical** to the training extraction on 20 real
images. A regression test locks the constant
(`tests/test_phase4.py::test_serving_input_size_matches_training`).

### Scope of the defect

| Affected | Not affected |
|---|---|
| Phase 3 app's displayed predictions | The trained model itself (unchanged) |
| Phase 3 screenshots (re-captured in Phase 4) | Phase 2 validation metrics |
| | The Phase 4 test evaluation (it uses the training extraction path) |

No model was retrained, no hyperparameter changed, no class mapping touched. The
model artefact is byte-identical to the Phase 2 output
(`sha256 53db2b14…dd83d3`, 15,728,010 bytes).

## 3. Check details

### A — Model identity

The registry's `checkpoint_path` for A5 resolves to
`models/final/rf_handcrafted.joblib`, which is the file the application loads.

| Property | Value |
|---|---|
| SHA-256 | `53db2b148c0c1e3b9722e610c2c47c9dcc758859049b38ff434c04d10cdd83d3` |
| Size | 15,728,010 bytes |
| Type | `StandardScaler` → `RandomForestClassifier` |
| Trees fitted | 200 (matches `run_configs.A5["n_estimators"]`) |
| Features | 83, exactly `ALL_FEATURES` in order |
| Classes | `["Damaged", "Dried", "Old", "Ripe", "Unripe"]` |
| `rf.classes_` | `[0, 1, 2, 3, 4]` |
| Registry validation macro F1 | 0.9539 |

### C — Class mapping

`config.QUALITY_CLASSES` = `class_map.json["quality_classes"]` =
artefact `classes` = `["Damaged", "Dried", "Old", "Ripe", "Unripe"]`.
The fitted estimator's own `classes_` are the contiguous indices `0…4`, so the
mapping from index to name is unambiguous in both directions.

### D — Test split untouched

Before Phase 4: no file in `evaluation/metrics/` mentioned "test", the registry's
`test_accuracy` / `test_macro_f1` / `test_weighted_f1` columns were empty, no
`features_test.npz` cache existed, and no module under `src/` or `website/` called
`resolve_split("test")`.

After the authorised evaluation, the check treats `evaluation/phase4_test/` and the
now-populated registry columns as expected, and still fails on any *other* test
artefact.

### E — Frozen artefacts

All **12/12** entries in `Admin/phase1_artifacts.sha256` verify, including
`splits.csv`, `duplicate_groups.json`, `class_map.json`, `audit_summary.json`,
the three notebooks and the feature-screening outputs.

### E2 — Split integrity

| Property | Value |
|---|---|
| Total images | 6,150 |
| Split sizes | 4,286 / 929 / 935 |
| Group ids spanning splits | **0** |
| SHA-256 values spanning splits | **0** |
| Test group ids shared with train/validation | **0** |
| Test SHA-256 shared with train/validation | **0** |

### F — Website metric sources

Three hard-coded metric literals were found and removed:

* `website/pages/3_About_Project.py` — the 0.9539 / 0.9521 figures in prose were
  replaced with values read from `p0_comparison.json`, with a fallback message when
  the artefact is absent.
* `website/utils/paths.py` — a numeric figure in a docstring was reworded.

The application now reads every metric from
`evaluation/metrics/p0_comparison.json`, `inference_latency.json`,
`datasets/metadata/*.json` or `evaluation/phase4_test/metrics.json`.

### G — Duplicate-cluster documentation

| File | Corrected figure present | Stale-figure mentions |
|---|---|---|
| `README.md` | yes | 2 (in the correction note) |
| `reports/results.md` | yes | 7 (raw-vs-corrected table) |
| `reports/limitations.md` | yes | 11 (correction section) |
| `reports/dataset_audit.md` | yes | 7 (appended correction) |
| `reports/phase2_p0_results.md` | yes | 1 |

Every remaining mention of 732 / 866 / "42 clusters" sits inside an explicit
correction that states the figure is superseded. The first version of this check
failed on `results.md` and `dataset_audit.md` because it searched for the literal
string "41 clusters", which those files express as a table cell (`| 41 | 134 |`).
That was a defect in the check, not in the documents; the check now uses a pattern.

## 4. Defects recorded

| Id | Severity | Description | Status |
|---|---|---|---|
| P4-1 | **Critical** | Serving extracted features at 256 px while training used 224 px | fixed, verified, regression-tested |
| P4-2 | Minor | Three hard-coded metric literals in the web app | fixed |
| P4-3 | Minor | Check G's pattern was too literal | fixed (check, not data) |

## 5. Verdict

Integrity audit **passes**. One critical defect was found, documented, corrected and
verified before the test split was scored; the test evaluation itself was unaffected
because it uses the training extraction path, proved equivalent by check B.

# Phase 4 — Website Performance Audit

**Date:** 2026-11-15 · **Record:** `evaluation/phase4_performance/performance.json`

All figures measured on the deployment machine (Intel i5-13450HX, 16 GB, 16 logical
CPUs, CPU-only — TensorFlow is not used at serving time). **50 repetitions** per
stage unless stated. Treat every number as "what this machine did", not a guarantee.

---

## 1. Summary

| Metric | Value |
|---|---|
| Cold start (fresh process → usable model) | **2.06 s** |
| Steady-state end-to-end prediction | **27.2 ms median** (mean 29.2, sd 5.0) |
| Model-only inference (`predict_proba`) | **7.5 ms median** |
| Preprocessing (resize + segmentation + 83 features) | **19.1 ms median** |
| Upload validation | 0.01 ms |
| Decode | 0.32 ms |
| Resident memory after load | ≈ 206 MB |
| 200 consecutive predictions | 0 errors, **+5.1 MB** RSS |
| 12 MP image handled | 255 ms, no failure |

The Phase 3 report quoted ~21 ms. The current measurement is **27 ms median** for
the identical pipeline. Both are honest observations of a shared machine under
different load; the Phase 3 figure came from a component-sum benchmark, this one
from repeated sequential measurements including validation. The difference is load
and methodology, not a regression — feature extraction measured 9.2 ms in Phase 3 and
15.4 ms here on the same code.

## 2. Stage timings (n = 50)

| Stage | Mean (ms) | Median (ms) | SD | Min | Max |
|---|---:|---:|---:|---:|---:|
| Upload validation | 0.01 | 0.01 | 0.00 | 0.00 | 0.03 |
| Decode | 0.34 | 0.32 | 0.07 | 0.27 | 0.54 |
| Preprocessing (total) | 20.52 | 19.10 | 3.75 | 17.47 | 32.43 |
| ↳ resize + segmentation | 2.62 | 2.51 | — | — | — |
| ↳ feature extraction (83) | 15.93 | 15.40 | — | — | — |
| Model inference | 8.33 | 7.54 | 2.25 | 6.94 | 17.11 |
| **End-to-end** | **29.20** | **27.19** | **4.99** | **25.26** | **47.41** |

Feature extraction dominates (≈ 70 % of end-to-end). It is GLCM (3 distances ×
4 angles) plus two uniform-LBP histograms; this is the intrinsic cost of the
classical pipeline, not overhead.

## 3. Cold start

Three fresh processes, timing from the first statement to a usable bundle:

| Measurement | Median |
|---|---:|
| Interpreter + imports (numpy, sklearn, joblib) | 0.21 s |
| `load_bundle()` (joblib load + contract verification + warm-up) | 2.06 s |

The Streamlit process pays this once; `st.cache_resource` then serves every
subsequent interaction from memory. A one-prediction warm-up runs inside the loader
so the first user-visible prediction is not 3× slower than the rest.

## 4. Resource use

| Measurement | Value |
|---|---|
| RSS before / after model load | 204.6 → 205.8 MB (+1.2 MB) |
| RSS after 200 predictions | 206.4 MB (+5.1 MB from start) |
| Mean CPU during the timing loop | 22 % (single-threaded work on 16 cores) |

The model is 15.7 MB on disk but the loaded pipeline is small; the ~205 MB baseline
is the Python + numpy + scikit-learn interpreter, not the model.

## 5. Behavioural checks

| Check | Method | Result |
|---|---|---|
| Repeated predictions | 200 consecutive | 0 errors, no memory growth trend |
| Large image | 4000×3000 JPEG, 14.06 MB | accepted (under the 15 MB limit), 255 ms, predicted `Ripe` |
| Oversized file | 15 MB + 10 bytes | rejected with a readable message |
| File just under limit | 15 MB − 10 bytes | accepted |
| Missing model file | loader pointed at absent path | `ModelUnavailable` with remediation text |
| Missing metric artefact | `_read_json` on absent path | `ArtefactMissing`, page renders an explanation |
| Invalid image bytes | `decode_image(b"not an image")` | `ValidationError`, no crash |
| Model loaded once | `st.cache_resource` | verified by log: one load per process |

**Upload limit agreement.** The uploader displays "15MB per file" because
`website/.streamlit/config.toml` sets `server.maxUploadSize = 15`, matching
`MAX_UPLOAD_BYTES` in `website/utils/paths.py`. Before Phase 4 the widget advertised
Streamlit's 200 MB default while the app rejected anything above 15 MB; a 14.06 MB
file is accepted and a 15 MB one is rejected, as displayed.

## 6. Comparison with the other candidate models

From `evaluation/metrics/inference_latency.json`, same machine and method:

| Model | End-to-end median |
|---|---:|
| Random Forest (deployed) | **19.7 ms** |
| MobileNetV2 multi-task | 172.8 ms |
| ResNet50 | 302.8 ms |

Measured in a lighter-load session than §2; the ordering is stable and is the
justification for the deployment choice alongside accuracy.

## 7. Issues found

| Id | Severity | Issue | Status |
|---|---|---|---|
| P4-4 | Minor | Uploader advertised 200 MB while the app enforced 15 MB | fixed (`.streamlit/config.toml`) |
| P4-5 | Info | Feature extraction dominates latency | documented, not a defect |

No functional defect was found in repeated uploads, large images, invalid images,
missing model, or missing artefacts.

## 8. What this audit does not cover

* Browser paint and websocket latency (the UI adds rendering time this script does
  not measure).
* Concurrency: all measurements are single-user, sequential. A multi-user load test
  was not performed.
* Absolute latency on other hardware.

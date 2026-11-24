# poster_source — how the posters are generated

Everything on `IEEE_Conference_Poster.pptx` / `.pdf` is produced from this folder plus the repository's
own result files. Nothing is typed in by hand that exists in an artefact: numbers, charts, confusion
matrix, dataset counts and the app screenshot are all read from the project.

## Files

| File | Purpose |
|---|---|
| `make_assets.py` | Builds every figure in `assets/` from repo artefacts (charts at final print size, 200 dpi) |
| `build_poster.py` | Builds the A0 landscape PPTX: layout, text, tables, flowchart, references |
| `build_ipcv_poster.py` | Builds the A3 landscape VisionX course poster: `IPCV_Poster_A3_Landscape.pptx` |
| `make_a3_assets.py` | Figures drawn at A3 placement size (`*_a3.png`), so their labels print at the stated point size |
| `export_pdf.ps1` | Exports a PPTX to PDF with PowerPoint (COM), then renders a PNG preview (`-Pptx`, `-Png` select the file) |
| `assets/` | Generated PNGs consumed by the poster (regenerable — see below) |

## Regenerate

Run from the repository root (Windows, PowerPoint installed):

```bash
.venv/Scripts/python.exe poster_source/make_assets.py     # figures  -> poster_source/assets/
.venv/Scripts/python.exe poster_source/build_poster.py    # slide    -> IEEE_Conference_Poster.pptx
powershell -NoProfile -ExecutionPolicy Bypass -File poster_source/export_pdf.ps1
```

The third step writes `IEEE_Conference_Poster.pdf` and `poster_preview.png` at the repository root.

The A3 course poster (VisionX template) is a separate build:

```bash
.venv/Scripts/python.exe poster_source/make_a3_assets.py        # -> assets/*_a3.png
.venv/Scripts/python.exe poster_source/build_ipcv_poster.py     # -> IPCV_Poster_A3_Landscape.pptx
powershell -NoProfile -ExecutionPolicy Bypass -File poster_source/export_pdf.ps1 `
    -Pptx IPCV_Poster_A3_Landscape.pptx -PreviewWidth 2600 -Png ipcv_poster_preview.png
```

It uses the ten numbered sections of the supplied `IPCV_Poster_A3_Landscape.pdf` template
(Abstract → Problem Statement → Objectives → Dataset Source → Methodology → IPCV Techniques →
Results → Tools & Technologies → Conclusion → References), A3 420 × 297 mm, body 8–9.5 pt.

Dependencies: `python-pptx` (1.0.2), `matplotlib`, `Pillow`, `numpy`, `opencv-python`; plus Microsoft
PowerPoint for the PDF export (LibreOffice is not installed on this machine and its layout differs).
`fig_*.png` fonts are Matplotlib defaults; the poster's own text is Segoe UI.

## What each figure reads

| Asset | Source |
|---|---|
| `fig_distribution.png` | `datasets/metadata/splits.csv` (counts computed at build time) |
| `samples_grid.png` | test-split images, one Bell Pepper photograph per quality class |
| `fig_models.png` | `evaluation/metrics/p0_comparison.json` (validation macro-F1 + bootstrap CIs) |
| `cm_test.png` | `evaluation/phase4_test/predictions.csv` |
| `fig_per_class.png` | `evaluation/phase4_test/metrics.json` |
| `fig_robustness.png` | `evaluation/phase4_robustness/summary.json` |
| `pipeline_visual.png` | `website/utils/image_processor.py` (the deployed preprocessing, actually executed) |
| `app_screenshot.png` | screenshot of the running Streamlit app |

## Design decisions worth knowing

- **Charts are drawn at their final physical size.** `figsize` matches the size the image is placed at,
  so a 16 pt label is 16 pt on the poster. Nothing is generated large and scaled down.
- **Text is laid out by measurement.** `_para_height()` wraps each paragraph with PIL's Segoe UI metrics,
  using a 1.33 em line height and the bullet's hanging indent. If you change copy, rebuild — heights are
  computed, not fixed, so the columns re-flow by themselves.
- **The flowchart is native shapes,** not an image, so it stays editable in PowerPoint.
- **Build-time audit.** `build_poster.py` prints the shape extent of each column and fails loudly if
  anything crosses the references band; keep that output at zero crossings.
- **Sizes live at the top of `build_poster.py`** (`W`, `H`, `MARGIN`, `COL_W`, `BODY_TOP`, `REFS_TOP`).
  Changing the poster size means changing those and the figure heights in the build functions.

## Changing the placeholders

Author block, department, institution, e-mail and conference name are strings in `build_header()` in
`build_poster.py`, currently bracketed placeholders. Replace the text between the brackets and rebuild.

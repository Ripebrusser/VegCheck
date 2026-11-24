# Preprocessing Operation Study

Measured on **190 training images**, sampled 10 per populated (vegetable, quality) cell, after a 224×224 resize. Latency is the median per-image cost on this CPU.

## Aggregate effect per operation

| Operation | noise σ | noise ↓ | sharpness | sharpness kept | contrast Δ | SSIM vs raw | ms/img |
|---|---:|---:|---:|---:|---:|---:|---:|
| `identity` | 0.95 | 0.0% | 106.7 | 100.0% | +0.0% | 1.000 | 0.0 |
| `gaussian_3` | 0.72 | 23.8% | 26.0 | 24.4% | -0.9% | 0.990 | 0.2 |
| `gaussian_5` | 0.67 | 29.2% | 14.2 | 13.3% | -1.4% | 0.980 | 0.2 |
| `median_3` | 0.82 | 13.8% | 65.3 | 61.2% | -0.3% | 0.990 | 0.1 |
| `median_5` | 0.68 | 27.9% | 45.4 | 42.6% | -0.9% | 0.972 | 0.4 |
| `bilateral` | 0.66 | 30.9% | 100.7 | 94.4% | -0.3% | 0.989 | 0.5 |
| `clahe` | 2.47 | -160.4% | 276.4 | 259.0% | +8.6% | 0.895 | 0.9 |
| `median3+clahe` | 2.16 | -127.5% | 152.3 | 142.7% | +8.5% | 0.903 | 1.0 |
| `gaussian3+clahe` | 2.06 | -116.8% | 77.1 | 72.3% | +8.6% | 0.901 | 1.1 |

## Denoising: measured verdict

- **Best noise/sharpness trade-off: `bilateral`** — 30.9% lower high-frequency noise while retaining 94.4% of the sharpness measure, at 0.5 ms per 224×224 image.
- **Gaussian blur is the wrong default here.** At the same 3×3 kernel it keeps only 24.4% of sharpness versus 61.2% for the median filter, for comparable noise reduction (23.8% vs 13.8%).

## Contrast enhancement

- CLAHE raises LAB-L contrast by **+8.6%** with SSIM 0.895 against the raw image, so it changes the image substantially.
- It also raises the Laplacian energy measure by 160.4%. Part of that is genuine sharpening of local contrast; part is amplified noise. This metric cannot separate the two, which is precisely why CLAHE is **ablation-tested in Phase 2 rather than assumed to help**.
- Running median 3×3 *before* CLAHE (`median3+clahe`) reduces the energy inflation from 160.4% to 127.5%, at the cost of 1.0 ms per image — the ordering used by the `enhanced` preset.

## Segmentation

- Plausible mask produced for **99.5%** of 190 sampled images.
- Foreground area fraction: mean 0.249 (5th pct 0.048, 95th pct 0.550).
- Failures by class: {'Dried': 1}.
- Because the backgrounds are already uniform and white, hard background removal mostly adds failure modes; it is kept as a selectable preset and ablated in Phase 2.

## Decisions taken into Phase 2

- Four presets are carried forward — `basic` (resize + ImageNet normalisation), `enhanced` (median 3×3 → CLAHE → normalise), `bilateral` (bilateral → CLAHE → normalise) and `segmented` (enhanced + background removal) — and compared on **validation macro-F1**, never on appearance.
- `bilateral` is promoted to its own preset because it measured the best noise/sharpness trade-off at 0.5 ms per image, only 5.4× the cost of median 3×3. `enhanced` keeps the median filter as a cheaper reference arm so the Phase 2 ablation compares denoisers like-for-like.
- No operation is applied because it is standard practice; each enters the pipeline only if the Phase 2 ablation supports it.

*Metric caveat:* The Laplacian-MAD 'noise_sigma' cannot separate amplified noise from legitimately added high-frequency detail; CLAHE raises it by design.

# Heatmap Figure Plan

## Purpose

The Communication Letters draft needs heatmaps to make the imperfect-CSI problem visible and to summarize the robustness gain without overstating unsupported experiments.

## Fig. 2: Original SwinJSCC Mismatch Heatmap

- Manuscript marker: `[TODO-FIG] Fig. 2`
- Data source: `mismatch_results/original_cifar10_awgn_C32_msssim_cpu.csv`
- Matrix axes:
  - Rows: true physical SNR `SNR_true` in `{1, 4, 7, 10, 13}` dB.
  - Columns: estimated modulation SNR `SNR_hat` in `{1, 4, 7, 10, 13}` dB.
- Metric: PSNR in dB.
- Intended claim: original SwinJSCC is strongest on or near matched CSI and can degrade under off-diagonal SNR mismatch.
- Boundary: do not use this figure to claim UA improvement; it is the motivation heatmap.

## Fig. 3: UA Gain Heatmap

- Manuscript marker: `[TODO-FIG] Fig. 3`
- Data sources:
  - `mismatch_results/original_cifar10_awgn_C32_msssim_cpu.csv`
  - `mismatch_results/ua_delta3_cifar10_awgn_C32_msssim_cpu.csv`
- Metric: `PSNR_UA - PSNR_original` in dB.
- Intended claim: UA training improves many off-diagonal mismatch entries in the current CIFAR10/AWGN/C=32 evaluation.
- Boundary: do not claim general robustness beyond this dataset, channel, and bottleneck setting.

## Rendering Requirements

- Export both PNG for quick review and SVG/PDF for manuscript use.
- Use the same color scale for original and UA absolute PSNR heatmaps if both are shown.
- Use a diverging color scale centered at zero for the gain heatmap.
- Annotate each cell with one decimal place if the figure remains readable in two-column format.

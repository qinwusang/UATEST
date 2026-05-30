# LaTeX Communication Letters Draft

This directory contains the IEEE Communications Letters-style manuscript for the SNR-mismatched SwinJSCC study.

## Current Manuscript Assets

- `main.tex`: full manuscript.
- `references.bib`: IEEE-style bibliography.
- `generate_figures.py`: regenerates the compact result tables, PSNR heatmap, combined trade-off figure, and qualitative reconstruction figure.
- `generated_metrics.tex`: generated scalar metric macros used by the manuscript.
- `table_tradeoff_rows.tex`: generated rows for the main AWGN SNR-mismatch trade-off table.
- `table_auxiliary_rows.tex`: generated rows for auxiliary Tail/Consistency robustness checks.
- `table_rayleigh_rows.tex`: generated rows for the Rayleigh check table.
- `figures/fig1_framework.tex`: native TikZ framework diagram.
- `figures/fig2_psnr_heatmaps.{pdf,png,svg}` and `.tex`: PSNR heatmaps used as Fig. 2.
- `figures/fig3_tradeoff_magnitude.{pdf,png,svg}` and `.tex`: double-column trade-off figure with metric-level and mismatch-magnitude panels.
- `figures/fig4_qualitative_recon.{pdf,png,svg}` and `.tex`: representative qualitative reconstruction grids under selected SNR mismatches.

Older figure/table fragments may remain in the directory for traceability, but the current `main.tex` uses only the files listed above.

## Required Data

The generator expects CPU MS-SSIM CSV files under `mismatch_results/`.

Required for the current draft:

- `original_cifar10_awgn_C32_msssim_cpu.csv`
- `ua-d0-eval_..._EP5_msssim_cpu.csv` (treated as the D0 equal-budget fine-tuning result for this manuscript)
- `ua-d1-eval_..._EP10_msssim_cpu.csv`
- `ua_delta3_cifar10_awgn_C32_msssim_cpu.csv`
- `ua-d6-eval_..._EP10_msssim_cpu.csv`
- `random-hat-eval_..._EP10_msssim_cpu.csv`
- Tail/Consistency auxiliary CSV files
- Rayleigh original and UA-D3 CPU CSV files

The current manuscript uses the available D0 fine-tuning CSV as the Perfect-SNR FT baseline with the same fine-tuning-budget interpretation as the other fine-tuned variants.

## Regenerate Assets

From the repository root:

```powershell
python latex\generate_figures.py
```

## Compile

Compile from the `latex/` directory:

```powershell
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The local Windows environment used for this edit does not expose `pdflatex` or `bibtex`; compile on the remote server or another machine with a TeX distribution installed.

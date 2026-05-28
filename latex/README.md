# LaTeX Communication Letters Draft

This directory contains the IEEE Communications Letters-style manuscript for the SNR-mismatched SwinJSCC study.

## Current Manuscript Assets

- `main.tex`: full manuscript.
- `references.bib`: IEEE-style bibliography.
- `generate_figures.py`: regenerates the compact result table, Rayleigh check table, and PSNR heatmap.
- `generated_metrics.tex`: generated scalar metric macros used by the manuscript.
- `table_tradeoff_rows.tex`: generated rows for the main AWGN SNR-mismatch trade-off table.
- `table_rayleigh_rows.tex`: generated rows for the Rayleigh check table.
- `figures/fig1_framework.tex`: native TikZ framework diagram.
- `figures/fig2_psnr_heatmaps.{pdf,png,svg}` and `.tex`: PSNR heatmaps used as Fig. 2.

Older figure/table fragments may remain in the directory for traceability, but the current `main.tex` uses only the files listed above.

## Required Data

The generator expects CPU MS-SSIM CSV files under `mismatch_results/`.

Required for the current draft:

- `original_cifar10_awgn_C32_msssim_cpu.csv`
- `ua-d1-eval_..._EP10_msssim_cpu.csv`
- `ua_delta3_cifar10_awgn_C32_msssim_cpu.csv`
- `ua-d6-eval_..._EP10_msssim_cpu.csv`
- `random-hat-eval_..._EP10_msssim_cpu.csv`
- Rayleigh original and UA-D3 CPU CSV files

The formal Perfect-CSI fine-tuning row requires a `ua-d0-eval_..._EP10_msssim_cpu.csv` file. If it is missing, the script emits a TODO row and does not substitute the available EP5 file.

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

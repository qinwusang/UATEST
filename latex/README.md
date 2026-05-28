# LaTeX Communication Letters Draft

This directory contains an IEEE Communications Letters-style LaTeX manuscript for the imperfect-CSI SwinJSCC project. The current draft uses only real CIFAR10/AWGN/C=32 CSV results copied into `mismatch_results/`.

## Main Files

- `main.tex`: full manuscript.
- `references.bib`: IEEE-style bibliography.
- `generate_figures.py`: regenerates result tables and figure assets.
- `generated_metrics.tex`: generated scalar metric macros.
- `table_aggregate_rows.tex`: generated rows for the main aggregate table.
- `table_gain_rows.tex`: generated rows for gains over the original checkpoint.
- `representative_cases_rows.tex`: generated representative mismatch case rows.

## Figures

- `figures/fig1_framework.tex`: native TikZ model framework diagram.
- `figures/fig2_psnr_heatmaps.{pdf,png,svg}` and `.tex`: PSNR mismatch heatmaps.
- `figures/fig3_aggregate_bars.{pdf,png,svg}` and `.tex`: aggregate robustness chart.
- `figures/fig4_msssim_heatmaps.{pdf,png,svg}` and `.tex`: MS-SSIM(dB) mismatch heatmaps.

Older generated TikZ fragments are retained for traceability but are not used by the current `main.tex`.

## Data Sources

The manuscript tables and plots are generated from:

- `mismatch_results/original_cifar10_awgn_C32_msssim_cpu.csv`
- `mismatch_results/ua_delta3_cifar10_awgn_C32_msssim_cpu.csv`
- `mismatch_results/tail_ua_delta3_cifar10_awgn_C32_msssim_cpu.csv`
- `mismatch_results/cons_ua_delta3_cifar10_awgn_C32_msssim_cpu.csv`
- `mismatch_results/tail_cons_ua_delta3_cifar10_awgn_C32_msssim_cpu.csv`

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

The current local machine used for this edit does not expose `pdflatex` or `bibtex`; compilation should be run on a machine with a TeX distribution installed.

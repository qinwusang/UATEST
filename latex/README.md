# LaTeX Letter Draft

This directory contains an IEEE Communications Letters-style LaTeX draft generated from the local imperfect-CSI SwinJSCC manuscript and real CIFAR10/AWGN/C=32 CSV results.

## Files

- `main.tex`: main IEEEtran manuscript.
- `references.bib`: BibTeX references.
- `generated_metrics.tex`: metrics generated from real CSV files.
- `representative_cases_rows.tex`: representative case table rows generated from real CSV files.
- `generate_figures.py`: standard-library script that regenerates TikZ data figures.
- `figures/fig1_framework.tex`: manually written model framework diagram.
- `figures/fig2_mismatch_heatmaps.tex`: generated PSNR heatmaps.
- `figures/fig3_aggregate_tradeoff.tex`: generated aggregate trade-off chart.
- `figures/fig4_representative_gains.tex`: generated representative gains chart.

## Regenerate Figures

From the repository root:

```powershell
python latex\generate_figures.py
```

The script uses:

- `mismatch_results/original_cifar10_awgn_C32_msssim_cpu.csv`
- `mismatch_results/ua_delta3_cifar10_awgn_C32_msssim_cpu.csv`

## Compile

Compile from the `latex/` directory:

```powershell
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

If `IEEEtran.cls` is unavailable locally, install the IEEEtran LaTeX package through the TeX distribution before compiling.

# Figure Data Manifest

No fake figures are created in this draft pass. The current figure inventory is:

| Figure | Source | Intended content | Status |
|---|---|---|---|
| Fig. 1 | `figures/model-framework.mmd` | Imperfect-CSI SwinJSCC framework; \(\gamma\) enters the AWGN channel and \(\hat{\gamma}\) enters encoder/decoder Channel ModNet | Diagram source exists |
| Fig. 2 | `mismatch_results/original_cifar10_awgn_C32_msssim_cpu.csv`; `figures/heatmap-plan.md` | Original SwinJSCC PSNR mismatch heatmap | [TODO-FIG] real data exists, export pending |
| Fig. 3 | `mismatch_results/original_cifar10_awgn_C32_msssim_cpu.csv`, `mismatch_results/ua_delta3_cifar10_awgn_C32_msssim_cpu.csv`; `figures/heatmap-plan.md` | UA minus original PSNR gain heatmap | [TODO-FIG] real data exists, export pending |
| Fig. 4 | future regenerated visualization directories; `figures/reconstruction-plan.md` | Original image, original mismatch reconstruction, UA mismatch reconstruction | [TODO-FIG] needs rerun to avoid checkpoint mixing |
| LaTeX Fig. 1 | `latex/figures/fig1_framework.tex` | Native TikZ framework diagram for IEEEtran draft | Created |
| LaTeX Fig. 2 | `latex/figures/fig2_mismatch_heatmaps.tex` | Original PSNR, UA PSNR, and UA-original PSNR gain heatmaps | Created from real CSV via `latex/generate_figures.py` |
| LaTeX Fig. 3 | `latex/figures/fig3_aggregate_tradeoff.tex` | Off-diagonal and diagonal PSNR aggregate bar chart | Created from real CSV via `latex/generate_figures.py` |
| LaTeX Fig. 4 | `latex/figures/fig4_representative_gains.tex` | Representative off-diagonal PSNR gains | Created from real CSV via `latex/generate_figures.py` |

Available data files:

| Data file | Intended use | Status |
|---|---|---|
| `mismatch_results/original_cifar10_awgn_C32_msssim_cpu.csv` | Original PSNR/MS-SSIM matrix and Fig. 2 | Real local data |
| `mismatch_results/ua_delta3_cifar10_awgn_C32_msssim_cpu.csv` | UA PSNR/MS-SSIM matrix and Fig. 3 | Real local data |
| `mismatch_results/CIFAR10_awgn_SwinJSCC_w-_SA_C32_mismatch.csv` | GPU PSNR mismatch matrix and possible consistency check | Real local data |
| `figures/model-framework.mmd` | Fig. 1 framework diagram | Diagram source |
| `latex/generated_metrics.tex` | LaTeX aggregate metric macros | Generated from real CSV |
| `latex/representative_cases_rows.tex` | LaTeX representative case table rows | Generated from real CSV |

Captions should describe the measured SNR mismatch condition, not claim general robustness beyond CIFAR10/AWGN/C=32.

# Figure Data Manifest

No fake figures are created in this draft pass. The current figure inventory is:

| Figure | Source | Intended content | Status |
|---|---|---|---|
| Fig. 1 | `figures/model-framework.mmd` | Imperfect-CSI SwinJSCC framework; \(\gamma\) enters the AWGN channel and \(\hat{\gamma}\) enters encoder/decoder Channel ModNet | Diagram source exists |
| Fig. 2 | `mismatch_results/original_cifar10_awgn_C32_msssim_cpu.csv`; `figures/heatmap-plan.md` | Original SwinJSCC PSNR mismatch heatmap | [TODO-FIG] real data exists, export pending |
| Fig. 3 | `mismatch_results/original_cifar10_awgn_C32_msssim_cpu.csv`, `mismatch_results/ua_delta3_cifar10_awgn_C32_msssim_cpu.csv`; `figures/heatmap-plan.md` | UA minus original PSNR gain heatmap | [TODO-FIG] real data exists, export pending |
| Fig. 4 | future regenerated visualization directories; `figures/reconstruction-plan.md` | Original image, original mismatch reconstruction, UA mismatch reconstruction | [TODO-FIG] needs rerun to avoid checkpoint mixing |
| LaTeX Fig. 1 | `latex/figures/fig1_framework.tex` | Native TikZ framework diagram for IEEEtran draft | Created |
| LaTeX Fig. 2 | `latex/figures/fig2_psnr_heatmaps.{pdf,png,svg}` | Original PSNR, UA-D3 PSNR, and UA-D3 minus original PSNR gain heatmaps | Created from real CSV via `latex/generate_figures.py` |
| LaTeX Fig. 3 | `latex/figures/fig3_tradeoff_magnitude.{pdf,png,svg}` | Double-column figure: metric-level PSNR trade-off plus method-level mismatch magnitude gain curves | Created from real CSV via `latex/generate_figures.py` |
| LaTeX Fig. 4 | `latex/figures/fig4_qualitative_recon.{pdf,png,svg}` | Representative CIFAR10 input/reconstruction grids under selected SNR mismatch cases | Created from existing `mismatch_vis_cifar10/` visualization export; qualitative only |
| LaTeX legacy Fig. 3 | `latex/figures/fig3_delta_tradeoff.{pdf,png,svg}` | Single-column metric-level PSNR trade-off curve | Generated for traceability; not used by current `latex/main.tex` |
| LaTeX legacy Fig. 4 | `latex/figures/fig4_mismatch_magnitude.{pdf,png,svg}` | Single-column method-level PSNR gain by mismatch magnitude | Generated for traceability; not used by current `latex/main.tex` |
| LaTeX Table I | `latex/table_tradeoff_rows.tex` | AWGN Delta/baseline trade-off table with bounded and full off-diagonal metrics | Created from real CSV via `latex/generate_figures.py` |
| LaTeX Table II | `latex/table_auxiliary_rows.tex` | Tail/Consistency auxiliary robustness variants | Created from real CSV via `latex/generate_figures.py` |
| LaTeX Table III | `latex/table_rayleigh_rows.tex` | Rayleigh original vs UA-D3 check | Created from real CSV via `latex/generate_figures.py` |

Available data files:

| Data file | Intended use | Status |
|---|---|---|
| `mismatch_results/original_cifar10_awgn_C32_msssim_cpu.csv` | Original PSNR/MS-SSIM matrix and Fig. 2 | Real local data |
| `mismatch_results/ua_delta3_cifar10_awgn_C32_msssim_cpu.csv` | UA PSNR/MS-SSIM matrix and Fig. 3 | Real local data |
| `mismatch_results/ua-d1-eval_*_EP10_msssim_cpu.csv` | UA-D1 row in the Delta table | Real local data |
| `mismatch_results/ua-d6-eval_*_EP10_msssim_cpu.csv` | UA-D6 row in the Delta table | Real local data |
| `mismatch_results/random-hat-eval_*_EP10_msssim_cpu.csv` | Independent random SNR conditioning baseline | Real local data |
| `mismatch_results/rayleigh-original-eval_*_msssim_cpu.csv` | Rayleigh original row | Real local data |
| `mismatch_results/rayleigh-ua-d3-eval_*_EP10_msssim_cpu.csv` | Rayleigh UA-D3 row | Real local data |
| `mismatch_results/tail_ua_delta3_cifar10_awgn_C32_msssim_cpu.csv` | Tail-UA aggregate table and robustness comparison | Real local data |
| `mismatch_results/cons_ua_delta3_cifar10_awgn_C32_msssim_cpu.csv` | Cons-UA aggregate table and robustness comparison | Real local data |
| `mismatch_results/tail_cons_ua_delta3_cifar10_awgn_C32_msssim_cpu.csv` | Tail+Cons-UA aggregate table and robustness comparison | Real local data |
| `mismatch_results/CIFAR10_awgn_SwinJSCC_w-_SA_C32_mismatch.csv` | GPU PSNR mismatch matrix and possible consistency check | Real local data |
| `figures/model-framework.mmd` | Fig. 1 framework diagram | Diagram source |
| `latex/generated_metrics.tex` | LaTeX aggregate metric macros | Generated from real CSV |
| `latex/representative_cases_rows.tex` | LaTeX representative case table rows | Generated from real CSV |
| `latex/table_aggregate_rows.tex` | LaTeX rows for five-method aggregate table | Generated from real CSV |
| `latex/table_gain_rows.tex` | LaTeX rows for gains over original checkpoint | Generated from real CSV |
| `latex/figures/fig2_psnr_heatmaps.{pdf,png,svg}` | PSNR mismatch heatmaps | Generated from real CSV |
| `latex/figures/fig3_tradeoff_magnitude.{pdf,png,svg}` | Combined trade-off and mismatch magnitude figure | Generated from real CSV |
| `latex/figures/fig4_qualitative_recon.{pdf,png,svg}` | Qualitative reconstruction grids | Generated from existing visualization PNGs; not a method ranking figure |

Captions should describe the measured SNR mismatch condition, not claim general robustness beyond CIFAR10/AWGN/C=32.

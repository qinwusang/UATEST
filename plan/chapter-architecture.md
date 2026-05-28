# Communication Letters LaTeX Redraft Architecture

## Required manuscript files

- `latex/main.tex` | role=full Communication Letters-style manuscript | min_chars=9000 | owner=main-agent | placeholders=no
- `latex/references.bib` | role=verified IEEE-style bibliography used by the LaTeX manuscript | min_chars=2500 | owner=main-agent | placeholders=no
- `latex/generated_metrics.tex` | role=auto-generated aggregate metric macros and table rows | owner=script | placeholders=no
- `latex/figures/fig1_framework.tex` | role=model framework diagram | owner=main-agent | placeholders=no
- `latex/figures/fig2_psnr_heatmaps.tex` | role=PSNR mismatch heatmap figure wrapper | owner=script | placeholders=no
- `latex/figures/fig3_aggregate_bars.tex` | role=aggregate performance chart wrapper | owner=script | placeholders=no
- `latex/figures/fig4_msssim_heatmaps.tex` | role=MS-SSIM(dB) mismatch heatmap figure wrapper | owner=script | placeholders=no

## Chapter roles inside `latex/main.tex`

- Abstract: 75--100 words; state imperfect CSI, SNR decoupling, UA training, and measured gains.
- I. Introduction: motivate Deep JSCC and SwinJSCC under practical CSI estimation errors; integrate related work compactly.
- II. System Model and Problem Formulation: define encoder, normalization, AWGN convention, imperfect CSI model, perfect/imperfect risks.
- III. Imperfect-CSI SwinJSCC: describe code-level SNR decoupling, UA training, tail-risk and consistency variants, and theoretical limitation.
- IV. Experimental Results: report CIFAR10/AWGN/C=32 setup, main five-method CPU table, PSNR/MS-SSIM heatmaps, and bounded claims.
- V. Conclusion: summarize the supported contribution and future validation scope.

## Evidence boundaries

- Use only real CSV files under `mismatch_results/` for numerical claims.
- Use CPU MS-SSIM CSV files as the main quantitative evidence.
- Do not treat old `main.py` mismatch CSV files with partial MS-SSIM fields as primary table evidence.
- Do not claim Rayleigh, Kodak, multiple CBR, or repeated-seed conclusions.

## Agent provenance note

The multi-agent tool is not invoked because the current user did not explicitly authorize parallel sub-agents. The task is therefore executed as a single-agent redraft with an explicit audit trail in `plan/progress.md`.

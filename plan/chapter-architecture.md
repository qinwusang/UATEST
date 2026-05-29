# Communication Letters LaTeX Redraft Architecture

## Required manuscript files

- `latex/main.tex` | role=full Communication Letters-style manuscript | min_chars=9000 | owner=main-agent | placeholders=no
- `latex/references.bib` | role=verified IEEE-style bibliography used by the LaTeX manuscript | min_chars=2500 | owner=main-agent | placeholders=no
- `latex/generated_metrics.tex` | role=auto-generated aggregate metric macros and table rows | owner=script | placeholders=no
- `latex/table_tradeoff_rows.tex` | role=auto-generated AWGN trade-off table rows | owner=script | placeholders=no
- `latex/table_auxiliary_rows.tex` | role=auto-generated auxiliary Tail/Consistency robustness table rows | owner=script | placeholders=no
- `latex/table_rayleigh_rows.tex` | role=auto-generated Rayleigh check table rows | owner=script | placeholders=no
- `latex/figures/fig1_framework.tex` | role=model framework diagram | owner=main-agent | placeholders=no
- `latex/figures/fig2_psnr_heatmaps.tex` | role=PSNR mismatch heatmap figure wrapper | owner=script | placeholders=no
- `latex/figures/fig3_delta_tradeoff.tex` | role=metric-level robustness--fidelity trade-off curve wrapper | owner=script | placeholders=no
- `latex/figures/fig4_mismatch_magnitude.tex` | role=method-level mismatch-magnitude robustness curve wrapper | owner=script | placeholders=no

## Chapter roles inside `latex/main.tex`

- Abstract: 75--100 words; state SNR mismatch, SNR decoupling, D0 fine-tuning baseline, UA training, and measured gains.
- I. Introduction: motivate Deep JSCC and SwinJSCC under practical SNR estimation errors; integrate related work compactly.
- II. System Model and Problem Formulation: define encoder, normalization, AWGN convention, imperfect CSI model, perfect/imperfect risks.
- III. SNR-Mismatch Training: describe code-level SNR decoupling, UA training, tail-risk and consistency variants, and theoretical limitation.
- IV. Experimental Results: report CIFAR10/AWGN/C=32 setup, bounded/full off-diagonal metrics, D0/D1/D3/D6/random-hat table, metric-level and method-level curves, auxiliary robustness table, Rayleigh check, PSNR heatmap, and bounded claims.
- V. Conclusion: summarize the supported contribution and future validation scope.

## Evidence boundaries

- Use only real CSV files under `mismatch_results/` for numerical claims.
- Use CPU MS-SSIM CSV files as the main quantitative evidence.
- Do not treat old `main.py` mismatch CSV files with partial MS-SSIM fields as primary table evidence.
- Do not claim Rayleigh, Kodak, multiple CBR, or repeated-seed conclusions.

## Agent provenance note

The multi-agent tool is not invoked because the current user did not explicitly authorize parallel sub-agents. The task is therefore executed as a single-agent redraft with an explicit audit trail in `plan/progress.md`.

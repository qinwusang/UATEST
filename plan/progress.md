# Progress

## 2026-05-13 - Planning and Full Draft

- Status: draft structure created
- User confirmation: SCI/conference, semantic communication, compact structure, existing CIFAR10/AWGN/C32 data only
- Files planned: project overview, outline, evidence map, experiment protocol, table schema, figure data manifest, full draft
- Notes: no training code changes should be made for this writing task

## 2026-05-13 - Communication Letters Rewrite

- Status: completed
- User request: add a model framework figure, increase formula density, expand references, and rewrite in Communication Letters style
- Output: `chapters/00-full-draft.md` rewritten with Abstract, Index Terms, numbered sections, 17 equations, two result tables, and IEEE-style references
- Figure: `figures/model-framework.mmd`
- Checks: abstract length is 83 words; numerical claims remain limited to existing CIFAR10/AWGN/C32 CSV results

## 2026-05-13 - CL Rapid Enhancement

- Status: completed
- User request: improve the Communication Letters draft without waiting for new experiments; mark missing experiments clearly.
- Output: `chapters/00-full-draft.md` now includes reproducibility details, corrected AWGN/noise convention, heatmap placeholders, qualitative reconstruction placeholders, and an ablation/baseline plan.
- New figure planning files: `figures/heatmap-plan.md`, `figures/reconstruction-plan.md`.
- Updated planning files: `plan/experiment-protocol.md`, `plan/evidence-map.md`, `figures/data-manifest.md`, `tables/table-schema.md`.
- Checks: abstract length is 87 words; incomplete experiments are marked with `[TODO-RUN]`, `[TODO-FIG]`, `[TODO-TABLE]`, `[TODO-CHECK]`, or `[LIMITATION]`.

## 2026-05-21 - LaTeX Letter Draft

- Status: completed
- User request: create a LaTeX-format Communication Letters-style paper with figures and data charts.
- Output: `latex/main.tex`, `latex/references.bib`, `latex/README.md`.
- Generated data figure files: `latex/figures/fig2_mismatch_heatmaps.tex`, `latex/figures/fig3_aggregate_tradeoff.tex`, `latex/figures/fig4_representative_gains.tex`.
- Manual framework figure: `latex/figures/fig1_framework.tex`.
- Generated result fragments: `latex/generated_metrics.tex`, `latex/representative_cases_rows.tex`.
- Checks: figures regenerated from real CIFAR10/AWGN/C=32 CSV files; local TeX engine is not installed, so PDF compilation could not be executed in this environment.

## 2026-05-26 - Robust Imperfect-CSI Training

- Status: implemented
- User request: start strengthening the method beyond basic UA training.
- Output: `main.py` now supports `--robust-train`, `--num-mismatch-samples`, `--tail-alpha`, `--lambda-tail`, and `--lambda-cons`.
- Method: robust training samples multiple estimated SNRs for the same true SNR, optimizes mean reconstruction loss plus top-tail mismatch loss, and optionally adds reconstruction consistency regularization.
- Experiment guide: `plan/robust-training.md`.
- Checks: `python -m py_compile main.py net/network.py`.

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

## 2026-05-28 - LaTeX CL Manuscript Redraft with Full Results

- Status: drafted
- User request: start writing the paper using all available results.
- Stage: S3/S4, experiment-result integration and full LaTeX drafting.
- Architecture: `plan/chapter-architecture.md`.
- Task packet: `plan/task-packets/2026-05-28-latex-letter-redraft.md`.
- Output: `latex/main.tex` rewritten as a Communication Letters-style manuscript with Abstract, Index Terms, I--V sections, formulas, Fig. 1--4, and two result tables.
- Data copied into `mismatch_results/`: Original, UA-Delta3, Tail-UA, Cons-UA, and Tail+Cons-UA CPU MS-SSIM CSV files.
- Generated assets: `latex/table_aggregate_rows.tex`, `latex/table_gain_rows.tex`, `latex/representative_cases_rows.tex`, `latex/figures/fig2_psnr_heatmaps.{pdf,png,svg}`, `latex/figures/fig3_aggregate_bars.{pdf,png,svg}`, and `latex/figures/fig4_msssim_heatmaps.{pdf,png,svg}`.
- Key result: UA-Delta3 gives the best off-diagonal average gain; Tail+Cons-UA gives the best worst-case gain. Tail/consistency variants are described as robustness ablations, not uniformly superior methods.
- Checks: `python latex\generate_figures.py`; `python -m py_compile latex\generate_figures.py`; all `\input{}` targets in `latex/main.tex` exist. Local `pdflatex`/`bibtex` are not installed, so PDF compilation was not run here.

### Capability-use audit

- Required skills: `paper-orchestration`, `latex-output`, `figures-python`.
- Skills actually used: same three skills, with the workflow adapted to the existing local LaTeX draft.
- Inputs consumed: five CPU MS-SSIM CSV files, existing LaTeX draft, existing framework figure, result summaries, and experiment manual.
- Inputs not used and why: old `main.py` mismatch CSV files are not used as primary table evidence because their MS-SSIM fields are incomplete or inconsistent with CPU evaluation.
- Artifacts produced: full LaTeX draft, generated figures/tables, copied normalized result CSVs, architecture file, task packet, README update, and data manifest update.

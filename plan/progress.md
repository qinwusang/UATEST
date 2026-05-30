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

## 2026-05-28 - Remaining Experiment Manual

- Status: completed
- User request: exclude repeated seeds and provide a one-by-one training manual for the remaining experiments.
- Code support added: `main.py` now supports `--snr-hat-mode bounded|independent|fixed` and `--fixed-snr-hat`, so random SNR conditioning and fixed conservative conditioning baselines are executable.
- Output: `plan/remaining-experiments-manual-zh.md`.
- Traceability: `plan/review/method-experiment-traceability.md`.
- Covered experiments: Delta ablation, qualitative reconstruction figure, independent SNR conditioning baseline, fixed SNR conditioning baseline, Tail/Cons small hyperparameter checks, Rayleigh checkpoint-dependent validation, and C/high-resolution checkpoint-dependent validation.
- Checks: `python -m py_compile main.py net\network.py eval_msssim_cpu.py`. `python main.py --help` could not run in the local Windows environment because `torch` is not installed there; the remote training environment should have torch.

## 2026-05-28 - CL Manuscript Cleanup for Submission Style

- Status: completed with one explicit data TODO.
- User request: implement the reviewer-style cleanup plan for the 4-page CL draft.
- Main changes: retitled the paper around SNR mismatch/imperfect SNR estimation, removed draft-generation language, fixed the robust-loss formula, clarified PSNR/MSE scaling, reduced figures to one PSNR heatmap, and moved the main table to a Delta/baseline trade-off view.
- Figure/table generation: `latex/generate_figures.py` now emits `table_tradeoff_rows.tex`, `table_rayleigh_rows.tex`, `generated_metrics.tex`, and `figures/fig2_psnr_heatmaps.*`.
- Evidence update: D1, D3, D6, random-hat, and Rayleigh CPU CSV files are used; this note was superseded on 2026-05-29 when the available D0 CSV was accepted as the Perfect-SNR FT baseline.
- Reference cleanup: `latex/references.bib` was rewritten with IEEE-style abbreviations and corrected DeepJSCC-f/OFDM/DeepSC-related author entries.
- Checks: `python latex\generate_figures.py`; SVG scan for `+39`/`+48`; further LaTeX compilation requires a TeX installation.

## 2026-05-29 - D0 Baseline Integration and Paper Prose Pass

- Status: completed.
- User request: use the available D0 result as the Perfect-SNR FT baseline and continue writing the paper from the provided result set.
- Stage: S3/S4, experiment integration and LaTeX manuscript drafting.
- Task packet: `plan/task-packets/2026-05-29-d0-paper-redraft.md`.
- Main changes: `latex/generate_figures.py` now reads the available D0 CPU CSV and emits a numeric Perfect-SNR FT row; `latex/main.tex` now uses D0 to argue that mismatch robustness is not explained by fine-tuning alone.
- Added result organization: `latex/table_auxiliary_rows.tex` reports Tail-UA, Cons-UA, Tail+Cons-UA, and stronger-tail variants as auxiliary robustness checks.
- Boundary: the paper reports the baseline as `D0`/Perfect-SNR FT and does not claim an EP10 epoch count for that run in the manuscript body.

### Capability-use audit

- Required skills: `paper-orchestration`, `latex-output`, `figures-python`.
- Skills actually used: same three skills, plus local validation scripts.
- Inputs consumed: D0, D1, D3, D6, random-hat, Tail/Cons, and Rayleigh CPU CSV files; current LaTeX manuscript and generator.
- Inputs not used and why: no repeated-seed results are used because the user explicitly excluded repeated seed experiments.
- Artifacts produced: updated manuscript prose, regenerated trade-off rows, new auxiliary robustness table rows, updated metrics macros, updated evidence map and README.
- Verification run: `python latex\generate_figures.py`; `python -m py_compile latex\generate_figures.py`; abstract length/input check; SVG gain-label scan; `git diff --check -- latex plan`.
- Remaining risk: local TeX engine is still unavailable, so PDF compilation must be run on the remote server or another machine with TeX installed.

## 2026-05-29 - Five-Page CL Revision

- Status: completed.
- User request: expand the CL draft toward a clearer five-page version with corrected formulas, bounded/full off-diagonal metrics, two new line figures, Rayleigh setup prose, and corrected references.
- Stage: S3/S4, figure/table generation plus manuscript redrafting.
- Task packet: `plan/task-packets/2026-05-29-five-page-cl-redraft.md`.
- Main changes: `latex/main.tex` now uses complex-symbol energy normalization, per-pixel `MSE_255`, bounded off-diagonal metric `P_off^(tau)`, and a restrained Rayleigh-channel description.
- Figure generation: `latex/generate_figures.py` now emits Fig. 3 metric-level trade-off curves and Fig. 4 method-level mismatch magnitude curves, with marker and linestyle differences.
- Reference cleanup: Shannon now uses complete 1948 page ranges; SwinJSCC now uses final TCCN vol. 11, no. 1, pp. 90--104, 2025 metadata with DOI.

### Capability-use audit

- Required skills: `paper-orchestration`, `latex-output`, `figures-python`.
- Skills actually used: same three skills, with local code/data inspection for Rayleigh channel details.
- Inputs consumed: current LaTeX manuscript, generator script, CPU CSV result files, `net/channel.py`, and bibliography.
- Inputs not used and why: no new training results were needed because the requested figures are derived from existing mismatch matrices.
- Artifacts produced: updated manuscript, generated Fig. 3/Fig. 4 assets and wrappers, updated trade-off rows/metrics, corrected bibliography, data manifest, README, and task packet.
- Verification run: `python latex\generate_figures.py`; `python -m py_compile latex\generate_figures.py`; abstract/citation/input check; forbidden-string search; SVG gain-label scan; `git diff --check -- latex figures plan`.
- Remaining risk: local `pdflatex`/`bibtex` are unavailable, so page count and final PDF layout must be checked in a TeX environment.

## 2026-05-30 - Weakness Response Revision

- Status: completed.
- User request: respond to identified weaknesses on novelty, auxiliary robust loss clarity, notation/noise details, missing related work, and experiment-scope limitations.
- Stage: S1/S2/S5, evidence-driven manuscript revision and technical consistency pass.
- Task packet: `plan/task-packets/2026-05-30-review-weakness-response.md`.
- Main changes: `latex/main.tex` now explicitly positions novelty as problem formulation/evaluation protocol rather than a new backbone; adds DeepJSCC-l++ and domain-randomization positioning; clarifies per-complex-symbol noise power; rewrites the robust auxiliary loss into complete sub-equations; and adds a scope/control-baseline subsection.
- Reference update: added DeepJSCC-l++ as an adaptive/robust JSCC reference with DOI metadata.

### Capability-use audit

- Required skills: `paper-orchestration`, evidence-driven manuscript revision, LaTeX output checks.
- Skills actually used: `paper-orchestration` plus local LaTeX/source verification.
- Inputs consumed: reviewer weakness list, `latex/main.tex`, `latex/references.bib`, existing evidence map, and web-checked related-work metadata.
- Inputs not used and why: no new experiment CSVs were consumed because the request concerned manuscript weaknesses and missing future controls rather than new completed results.
- Artifacts produced: updated manuscript, BibTeX entry, README/evidence/progress updates, and task packet.
- Verification run: `python -m py_compile latex\generate_figures.py`; abstract/input/citation check; forbidden-string search; `git diff --check` on edited manuscript and planning files. Local TeX compilation remains unavailable.

## 2026-05-30 - D0 and Augmentation-Control Clarification

- Status: completed.
- User request: treat D0 as the same-budget fine-tuning baseline; strengthen the response to the criticism that the method is only naive SNR data augmentation/domain randomization; add traditional communication SNR/CSI mismatch context.
- Main changes: `latex/main.tex` now states that D0, D1, D3, D6, and Random-hat use the same fine-tuning budget; removes the D0 sanity-check caveat; adds conventional imperfect channel knowledge/link adaptation references; explains why Random-hat is not a recommended method despite strong stress-test robustness; and lists fixed conservative SNR conditioning as a future control because no verified Fixed-SNR CSV is available locally.
- Verification note: manuscript claims about Fixed-SNR remain non-numeric until a verified CSV exists under `mismatch_results/`.

## 2026-05-30 - Risk Notation and Scope Polish

- Status: completed.
- User request: fix risk-expression precision, avoid blunt single-run wording, add complexity/latency note, and list encoder-only/decoder-only mismatch as a missing ablation.
- Main changes: `latex/main.tex` now defines the end-to-end map `H_theta(x,gamma,gammahat)` and rewrites the perfect/mismatch risks using it; adds an inference/training cost note; clarifies that the Lipschitz argument is qualitative; replaces the explicit single-run limitation with evaluation-mean wording; and adds encoder-only/decoder-only mismatch tests to future controls.

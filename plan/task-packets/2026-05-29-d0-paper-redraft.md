## Task Packet

- Scope: revise the Communication Letters LaTeX manuscript after user approval to use the available D0 fine-tuning result as the Perfect-SNR FT baseline.
- Files to read: `latex/main.tex`, `latex/generate_figures.py`, `latex/references.bib`, `mismatch_results/*.csv`, `plan/evidence-map.md`.
- Files allowed to edit: `latex/main.tex`, `latex/generate_figures.py`, generated LaTeX table/metric fragments, `latex/README.md`, `plan/evidence-map.md`, `plan/chapter-architecture.md`, `plan/progress.md`.
- Required skills: `paper-orchestration`, `latex-output`, `figures-python`.
- Evidence/data inputs: Original, D0, D1, D3, D6, random-hat, Tail/Cons/Tail+Cons/TailLt1, and Rayleigh CPU CSV files under `mismatch_results/`.
- Required artifacts: updated manuscript prose, regenerated main trade-off table, auxiliary robustness table, generated metric macros, and updated figure/table generation script.
- Rejection checks: do not claim D0 is an EP10 run in the manuscript body; do not overstate Rayleigh as a full fading-channel evaluation; do not introduce placeholder result claims; keep the abstract within 75--100 words.
- Validation commands: `python latex\generate_figures.py`; `python -m py_compile latex\generate_figures.py`; citation-key check; forbidden-string search; `git diff --check -- latex plan`.

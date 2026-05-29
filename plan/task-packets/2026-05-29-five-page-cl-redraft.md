## Task Packet

- Scope: implement the five-page IEEE Communications Letters revision plan for the SNR-mismatched SwinJSCC LaTeX draft.
- Files to read: `latex/main.tex`, `latex/generate_figures.py`, `latex/references.bib`, `mismatch_results/*.csv`, `net/channel.py`, `plan/evidence-map.md`.
- Files allowed to edit: `latex/main.tex`, `latex/generate_figures.py`, generated LaTeX figure/table fragments, `latex/references.bib`, `latex/README.md`, `figures/data-manifest.md`, `plan/chapter-architecture.md`, `plan/evidence-map.md`, `plan/progress.md`.
- Required skills: `paper-orchestration`, `latex-output`, `figures-python`.
- Evidence/data inputs: Original, D0, D1, D3, D6, random-hat, Tail/Cons/Tail+Cons, and Rayleigh CPU CSV files under `mismatch_results/`.
- Required artifacts: bounded off-diagonal metric, corrected normalization and PSNR formulas, Fig. 3 metric trade-off curve, Fig. 4 mismatch magnitude curve, Rayleigh setting prose, and corrected Shannon/SwinJSCC BibTeX metadata.
- Rejection checks: do not modify training logic; do not claim Rayleigh generalization; do not claim D0 equal-budget control if only the current D0 CSV is available; do not use method-specific line styles for Fig. 3.
- Validation commands: `python latex\generate_figures.py`; `python -m py_compile latex\generate_figures.py`; citation/input checks; forbidden-string search; SVG gain-label scan; `git diff --check -- latex figures plan`.

## Task Packet

- Scope: respond to technical weakness feedback on the IEEE Communications Letters draft by tightening novelty claims, correcting equations, clarifying auxiliary robust loss terms, improving related-work positioning, and adding limitations/future-baseline text.
- Files to read: `latex/main.tex`, `latex/references.bib`, `latex/README.md`, `plan/evidence-map.md`, `latex/generate_figures.py`.
- Files allowed to edit: `latex/main.tex`, `latex/references.bib`, `latex/README.md`, `plan/evidence-map.md`, `plan/progress.md`.
- Required skills: `paper-orchestration`, evidence-driven manuscript revision, LaTeX output checks.
- Evidence/data inputs: existing CIFAR10/AWGN/C=32 mismatch results; existing Rayleigh check; web-verified metadata for adaptive/robust JSCC references.
- Required artifacts: corrected noise-power wording, complete auxiliary robust-loss equations, related-work/positioning paragraph covering adaptive/robust JSCC and domain randomization, explicit limitation paragraph for dataset/rate/seed/baseline/error-distribution gaps.
- Rejection checks: do not claim new backbone/algorithmic breakthrough; do not claim high-resolution, multi-CBR, repeated-seed, no-ModNet, or Gaussian-error experiments as completed; do not expand Rayleigh into a generalization claim.
- Validation commands: `python -m py_compile latex\generate_figures.py`; citation/input checks; forbidden-symbol search for `\hat{C}`, `\hat{\tau}`, and ambiguous Rayleigh generalization; `git diff --check`.

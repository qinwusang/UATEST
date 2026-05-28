# Project Overview

## Basic Information

- Paper type: SCI/conference manuscript draft
- Discipline: semantic communication, with a technical focus on Deep JSCC image transmission
- Tentative title: Imperfect-CSI SwinJSCC: Robust Semantic Image Transmission under SNR Mismatch
- Language: English manuscript body
- Output format: LaTeX manuscript plus generated PDF/PNG/SVG figure assets
- Current stage: Communication Letters-style LaTeX redraft with full CIFAR10/AWGN/C=32 result integration

## Research Positioning

This manuscript studies a practical mismatch in adaptive Deep JSCC image transmission. The original SwinJSCC model uses channel modulation to adapt features to a provided SNR value, but the code path assumes that the SNR used by the modulation network is the same as the SNR governing the physical channel noise. The local project decouples these two quantities into `SNR_true` and `SNR_hat`, trains with bounded estimation errors, and evaluates a full true-SNR by estimated-SNR mismatch matrix.

The contribution is positioned as a strong method contribution, but only within the implemented scope: imperfect-CSI formulation, uncertainty-aware training, and mismatch evaluation. The manuscript must not claim a new Swin backbone, a confidence-aware module, or a distributional CSI encoder.

## Confirmed Structure

1. Abstract
2. Introduction
3. Method
4. Experiments
5. Conclusion

Related work is integrated into the Introduction to match a compact SCI/conference style.

## Evidence Boundary

The draft may use the SwinJSCC README, local code changes, `AGENT.md`, and the existing CSV results under `mismatch_results/`. Numerical claims must be derived from real CSV files. Rayleigh, Kodak, rate adaptation ablations, and architectural uncertainty modules are not yet supported by local result files and must be treated as limitations or future work.

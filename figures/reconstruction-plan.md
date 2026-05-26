# Reconstruction Figure Plan

## Purpose

The manuscript needs a qualitative comparison, but the existing `mismatch_vis_cifar10` outputs may have been overwritten by different checkpoints. The final reconstruction figure should therefore be regenerated with explicit output directories.

## Fig. 4: Qualitative Mismatch Reconstruction

- Manuscript marker: `[TODO-FIG] Fig. 4`
- Layout: original image, original SwinJSCC mismatch reconstruction, UA mismatch reconstruction.
- Recommended cases:
  - `SNR_true=1`, `SNR_hat=7`: severe overestimation of a poor channel; largest current PSNR gain.
  - `SNR_true=4`, `SNR_hat=10`: medium-SNR overestimation case.
  - `SNR_true=13`, `SNR_hat=1`: severe underestimation of a good channel.

## Output Directory Requirement

Regenerate visualizations into separate directories to avoid checkpoint mixing:

- `mismatch_vis_cifar10_original/`
- `mismatch_vis_cifar10_ua_delta3/`

Do not use images from `mismatch_vis_cifar10` as final evidence unless the checkpoint provenance is verified.

## Caption Boundary

The caption should state the exact dataset, channel, bottleneck, true SNR, and estimated SNR. It should describe visible reconstruction differences only for the displayed examples and should not claim general perceptual superiority.

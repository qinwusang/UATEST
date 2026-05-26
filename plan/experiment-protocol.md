# Experiment Protocol

## Dataset and Channel

The current draft uses CIFAR10 images and the AWGN channel. The bottleneck dimension is fixed to C=32. The SNR set is `{1, 4, 7, 10, 13}` dB for both true channel SNR and estimated modulation SNR.

Current reproducibility details supported by code:

- CIFAR10 batch size: 256.
- Optimizer: Adam.
- Learning rate: `1e-4`.
- Seed: 42.
- UA training range: `gamma_min=1`, `gamma_max=13`.
- Current UA result assumption: `Delta=3 dB`, matching `ua_delta3_cifar10_awgn_C32_msssim_cpu.csv`.
- Initialization: from the original CIFAR10/AWGN/C32 checkpoint path used by the local training script.
- [TODO-CHECK] Record the exact command line, fine-tuning epoch, and output checkpoint used to generate `ua_delta3_cifar10_awgn_C32_msssim_cpu.csv`.

## Baselines

The baseline is the original SwinJSCC `w/_SA` CIFAR10 AWGN C32 checkpoint. It is fair for the current comparison because both the original and UA evaluations use the same model family, dataset, channel type, bottleneck dimension, and SNR grid.

## Method Under Test

The UA model is trained with SNR estimation uncertainty. The implemented sampling rule chooses a true SNR from the configured SNR set and perturbs it by a bounded error before clipping the estimated SNR to the configured range.

## Metrics

The draft reports PSNR, MS-SSIM, and MS-SSIM(dB). Off-diagonal entries in the mismatch matrix measure imperfect-CSI robustness. Diagonal entries measure the retained perfect-CSI performance.

## Main Comparison

The main comparison is original SwinJSCC versus UA-SwinJSCC on a full `SNR_true x SNR_hat` matrix. The primary claim is improved off-diagonal robustness with limited diagonal degradation.

## Required Before Submission

These experiments are not yet completed and must remain marked as TODO in the manuscript until real results exist.

| Item | Purpose | Status |
|---|---|---|
| Delta ablation | Verify whether `Delta=3 dB` is a stable choice and whether too-small or too-large uncertainty hurts matched/off-diagonal trade-off | [TODO-RUN] `Delta={0,1,3,6}` |
| Matched-only fine-tuning | Rule out the possibility that gains come only from extra fine-tuning rather than SNR uncertainty | [TODO-RUN] same checkpoint and epochs, but `SNR_hat=SNR_true` |
| Conservative SNR conditioning baseline | Compare against a simple robust heuristic that does not require UA training | [TODO-RUN] define conservative estimate rule before running |
| No-SA/no-Channel-ModNet baseline | Clarify whether the gain depends on the attention/modulation path | [TODO-RUN] only if compatible checkpoint exists |
| Repeated seeds | Estimate variance of aggregate gains | [TODO-RUN] at least 3 seeds if time allows |
| Qualitative reconstructions | Support numerical results with representative examples | [TODO-FIG] regenerate original and UA figures into separate directories |

## Limitations

The current protocol does not yet include Rayleigh channels, high-resolution image datasets, different CBR values, repeated random seeds, or completed ablations over `delta-train`. These must not be written as completed experiments.

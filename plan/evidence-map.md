# Evidence Map

## Source E1: Original SwinJSCC README

- Path: `README.md`
- Supports: SwinJSCC is the baseline project; the original system includes Channel ModNet and Rate ModNet; `multiple-snr` controls fixed or random SNR; C controls fixed or random rate.
- Allowed claim: original SwinJSCC adapts to SNR/rate through modulation modules.
- Boundary: README does not provide imperfect-CSI evaluation.

## Source E2: Network-level implementation

- Path: `net/network.py`
- Supports: local `SwinJSCC.forward` accepts `mod_SNR`; `chan_param` is used for physical channel forwarding; `mod_param` is used in Channel ModNet paths for `SwinJSCC_w/_SA` and `SwinJSCC_w/_SAandRA`.
- Allowed claim: this project decouples true channel SNR and estimated modulation SNR.

## Source E3: Training and evaluation implementation

- Path: `main.py`
- Supports: local arguments `--ua-train`, `--delta-train`, `--snr-min`, `--snr-max`; UA training samples `snr_true` and `snr_hat`; testing enumerates `SNR_true x SNR_hat` and saves mismatch CSV/visualizations.
- Allowed claim: this project implements uncertainty-aware SNR mismatch training and mismatch matrix evaluation.
- Reproducibility details supported by code: CIFAR10 batch size 256, Adam optimizer, learning rate `1e-4`, seed 42, and checkpoint initialization through `--checkpoint` or the default original CIFAR10/AWGN/C32 path.
- Boundary: exact fine-tuning epoch and command line for the current UA CSV still require `[TODO-CHECK]`.

## Source E3b: Channel implementation

- Path: `net/channel.py`
- Supports: `complex_normalize` uses `mean(x ** 2) * 2`; AWGN samples independent real and imaginary noise components with standard deviation `sqrt(1/(2*10^(SNR/10)))`.
- Allowed claim: after real-to-complex pairing, the average complex symbol power is normalized and the total complex noise power is `1/10^(SNR/10)`.

## Source E4: MS-SSIM evaluation

- Path: `eval_msssim_cpu.py`
- Supports: CPU-side MS-SSIM evaluation for the same mismatch matrix.
- Allowed claim: perceptual-quality robustness is evaluated with MS-SSIM and MS-SSIM(dB) using real local CSV outputs.

## Source E5: Original checkpoint results

- Path: `mismatch_results/original_cifar10_awgn_C32_msssim_cpu.csv`
- Supports: original model PSNR and MS-SSIM values over CIFAR10/AWGN/C32 mismatch matrix.

## Source E6: UA checkpoint results

- Path: `mismatch_results/ua_delta3_cifar10_awgn_C32_msssim_cpu.csv`
- Supports: UA model PSNR and MS-SSIM values over the same mismatch matrix.

## Derived Result Summary

- Off-diagonal average PSNR: original 29.8782 dB, UA 31.7947 dB, improvement 1.9164 dB.
- Diagonal average PSNR: original 34.9688 dB, UA 34.8771 dB, difference -0.0917 dB.
- Largest observed PSNR gain: +4.7989 dB at `SNR_true=1`, `SNR_hat=7`.
- Average off-diagonal MS-SSIM gain: +0.0290.

These derived values are computed from E5 and E6.

## Source E7: Communication Letters format

- Source: IEEE Communications Letters Policies and Guidelines, IEEE Communications Society.
- Supports: letter-style manuscript should be concise, no longer than five IEEE-format pages, include abstract, index terms, text, figures/tables, and references; abstract should be 75-100 words.
- Allowed claim: the rewritten draft follows a Communication Letters-inspired compact structure.

## Source E8: Model framework figure

- Path: `figures/model-framework.mmd`
- Supports: Fig. 1 in the draft, showing image input, encoder conditioned on estimated SNR, physical channel governed by true SNR, decoder conditioned on estimated SNR, and reconstruction.

## Source E9: Figure planning files

- Paths: `figures/heatmap-plan.md`, `figures/reconstruction-plan.md`
- Supports: planned Fig. 2 original mismatch heatmap, Fig. 3 UA gain heatmap, and Fig. 4 qualitative reconstruction comparison.
- Boundary: Fig. 2 and Fig. 3 have real CSV data but still need exported figures; Fig. 4 requires a visualization rerun to avoid checkpoint mixing.

## Current Evidence Gaps

- [TODO-RUN] Delta ablation: `Delta={0,1,3,6}`.
- [TODO-RUN] matched-only fine-tuning baseline.
- [TODO-RUN] conservative SNR conditioning baseline.
- [TODO-RUN] no-SA/no-Channel-ModNet baseline if compatible checkpoint exists.
- [TODO-RUN] repeated seeds.
- [TODO-FIG] vector heatmaps and qualitative reconstruction figure.
- [TODO-CHECK] exact UA fine-tuning epoch and command.

## Added Reference Evidence

- Deep JSCC image transmission: Bourtsoulatze, Kurka, and Gunduz, IEEE TCCN 2019, DOI 10.1109/TCCN.2019.2919300.
- Attention-adaptive JSCC: Xu et al., IEEE TCSVT 2022, DOI 10.1109/TCSVT.2021.3082521.
- NTSCC semantic communication: Dai et al., IEEE JSAC 2022, DOI 10.1109/JSAC.2022.3180802.
- DeepSC semantic communication: Xie et al., IEEE TSP 2021, DOI 10.1109/TSP.2021.3071210.
- SwinJSCC: Yang et al., IEEE TCCN 2024, DOI 10.1109/TCCN.2024.3424842.

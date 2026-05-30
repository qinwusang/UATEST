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

## Source E6c: Delta and baseline results

- Paths:
  - `mismatch_results/ua-d0-eval_*_EP5_msssim_cpu.csv`
  - `mismatch_results/ua-d1-eval_*_EP10_msssim_cpu.csv`
  - `mismatch_results/ua-d6-eval_*_EP10_msssim_cpu.csv`
  - `mismatch_results/random-hat-eval_*_EP10_msssim_cpu.csv`
  - `mismatch_results/rayleigh-original-eval_*_msssim_cpu.csv`
  - `mismatch_results/rayleigh-ua-d3-eval_*_EP10_msssim_cpu.csv`
- Supports: Delta-width trade-off, independent random SNR conditioning baseline, and a Rayleigh channel check.
- Boundary: the current manuscript uses the available D0 CSV as the Perfect-SNR FT baseline and does not state an epoch-count claim for D0 in the paper body.

## Source E6b: Robust variant checkpoint results

- Paths:
  - `mismatch_results/tail_ua_delta3_cifar10_awgn_C32_msssim_cpu.csv`
  - `mismatch_results/cons_ua_delta3_cifar10_awgn_C32_msssim_cpu.csv`
  - `mismatch_results/tail_cons_ua_delta3_cifar10_awgn_C32_msssim_cpu.csv`
- Supports: Tail-UA, Cons-UA, and Tail+Cons-UA PSNR/MS-SSIM values over the same CIFAR10/AWGN/C32 mismatch matrix.
- Allowed claim: robust variants provide complementary trade-offs; UA-Delta3 is best on average off-diagonal PSNR/MS-SSIM(dB), while Tail+Cons-UA is best on worst-case PSNR/MS-SSIM(dB).
- Boundary: these are single-seed CIFAR10/AWGN/C32 results, not broad channel or dataset conclusions.

## Derived Result Summary

- Off-diagonal average PSNR: original 29.8782 dB, UA 31.7947 dB, improvement 1.9164 dB.
- Diagonal average PSNR: original 34.9688 dB, UA 34.8771 dB, difference -0.0917 dB.
- Perfect-SNR D0 fine-tuning off-diagonal PSNR gain over original: +0.03 dB; matched PSNR change: +0.05 dB.
- UA-D3 bounded off-diagonal (`tau=3 dB`) PSNR gain over original: +1.09 dB.
- UA-D6 off-diagonal PSNR gain over original: +3.11 dB; matched PSNR change: -0.36 dB.
- Random-hat off-diagonal PSNR gain over original: +4.06 dB; matched PSNR change: -1.04 dB.
- Rayleigh UA-D3 off-diagonal PSNR gain over Rayleigh original: +1.50 dB.
- Largest observed PSNR gain: +4.7989 dB at `SNR_true=1`, `SNR_hat=7`.
- Average off-diagonal MS-SSIM(dB): original 15.9684 dB, UA 17.9652 dB, improvement 1.9969 dB.
- Tail+Cons-UA worst-case PSNR gain: +0.9892 dB.
- Tail+Cons-UA worst-case MS-SSIM(dB) gain: +0.7530 dB.

These derived values are computed from E5, E6, and E6b.

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
- Boundary: LaTeX Fig. 2--3 use generated data plots. LaTeX Fig. 4 now uses the only available `mismatch_vis_cifar10/` export as a qualitative reconstruction example, not as an Original-vs-UA method comparison. A method-separated qualitative comparison still requires rerunning visualizations into separate directories.

## Source E10: Robustness and domain-randomization positioning

- Sources: Tobin et al., IROS 2017, DOI 10.1109/IROS.2017.8202133; Volpi et al., NeurIPS 2018.
- Supports: the paper can position bounded SNR perturbation training as a communication-specific instance of training over nuisance variables or deployment shifts.
- Boundary: the manuscript should not claim to solve general domain generalization; the nuisance variable is specifically the mismatch between true physical SNR and estimated modulation SNR.

## Source E11: Adaptive/robust JSCC positioning

- Source: Bian, Shao, and Gunduz, DeepJSCC-l++, IEEE GLOBECOM 2023, DOI 10.1109/GLOBECOM54140.2023.10436878.
- Supports: prior DeepJSCC variants already study robustness and bandwidth-adaptive wireless image transmission; this manuscript should position its novelty as the explicit true-SNR/estimated-SNR mismatch formulation for SwinJSCC, not as general robust JSCC from scratch.
- Boundary: do not claim prior adaptive JSCC assumes no channel uncertainty in general; the specific gap is systematic decoupled `gamma`/`gamma_hat` training and evaluation for SwinJSCC Channel ModNet.

## Current Evidence Gaps

- [TODO-RUN] matched-only fine-tuning baseline.
- [TODO-RUN] conservative SNR conditioning baseline.
- [TODO-RUN] no-SA/no-Channel-ModNet baseline if compatible checkpoint exists.
- [TODO-RUN] repeated seeds.
- [TODO-FIG] method-separated Original-vs-UA qualitative reconstruction figure.

## Added Reference Evidence

- Deep JSCC image transmission: Bourtsoulatze, Kurka, and Gunduz, IEEE TCCN 2019, DOI 10.1109/TCCN.2019.2919300.
- Attention-adaptive JSCC: Xu et al., IEEE TCSVT 2022, DOI 10.1109/TCSVT.2021.3082521.
- NTSCC semantic communication: Dai et al., IEEE JSAC 2022, DOI 10.1109/JSAC.2022.3180802.
- DeepSC semantic communication: Xie et al., IEEE TSP 2021, DOI 10.1109/TSP.2021.3071210.
- SwinJSCC: Yang et al., IEEE TCCN 2025, vol. 11, no. 1, pp. 90--104, DOI 10.1109/TCCN.2024.3424842.
- DeepJSCC-l++: Bian, Shao, and Gunduz, IEEE GLOBECOM 2023, DOI 10.1109/GLOBECOM54140.2023.10436878.
- Domain randomization: Tobin et al., IEEE/RSJ IROS 2017, pp. 23--30, DOI 10.1109/IROS.2017.8202133.
- Domain generalization via adversarial augmentation: Volpi et al., NeurIPS 2018, vol. 31, pp. 5334--5344.

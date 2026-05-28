# Method-Experiment Traceability

| Contribution | Method module | Experiment | Table/Figure | Allowed claim | Evidence status |
|---|---|---|---|---|---|
| Decouple true SNR and estimated SNR | `net/network.py` `mod_SNR`; `main.py` mismatch evaluation | Original vs UA mismatch matrix | Current Table I, Fig. 2, Fig. 4 | Imperfect-CSI formulation better matches deployment with SNR estimation error | Supported on CIFAR10/AWGN/C=32 |
| UA bounded SNR-error training | `--ua-train`, `--delta-train`, `--snr-hat-mode bounded` | Delta ablation `Delta={0,1,3,6}` | Future Delta table | Delta=3 is justified by ablation rather than arbitrary choice | `Delta=3` complete; `0,1,6` pending |
| Tail-risk robustness | `--robust-train`, `--lambda-tail` | Tail-UA and Tail+Cons-UA | Current Table I; optional tail hyperparameter table | Tail term improves worst-case trade-off in some settings | Basic result complete; small hyperparameter check pending |
| Reconstruction consistency | `--robust-train`, `--lambda-cons` | Cons-UA and Tail+Cons-UA | Current Table I; optional consistency hyperparameter table | Consistency can preserve matched-SNR performance while retaining mismatch gains | Basic result complete; small hyperparameter check pending |
| Bounded CSI-error modeling is not generic random augmentation | `--snr-hat-mode independent` | Random SNR conditioning baseline | Future baseline table | Bounded perturbation is more appropriate than independent SNR randomization if it outperforms the baseline | Pending |
| Conservative fixed conditioning is not sufficient | `--snr-hat-mode fixed`, `--fixed-snr-hat` | Fixed SNR_hat=1 and 7 baselines | Future baseline table | A fixed conservative SNR policy does not replace UA if it loses average or high-SNR performance | Pending |
| Visual reconstruction quality | existing `main.py` CIFAR10 visualization output | Qualitative mismatch cases `(1,7)`, `(4,10)`, `(13,1)` | Future reconstruction figure | Numerical mismatch gains correspond to visible reconstruction improvements | Pending |
| Channel generalization | `--channel-type rayleigh` | Rayleigh Original vs UA if checkpoint exists | Future robustness table | Method extends beyond AWGN only if Rayleigh results support it | Pending, checkpoint-dependent |

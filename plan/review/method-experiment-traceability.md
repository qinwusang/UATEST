# Method-Experiment Traceability

| Contribution | Method module | Experiment | Table/Figure | Allowed claim | Evidence status |
|---|---|---|---|---|---|
| Imperfect-CSI formulation | Decouple `SNR_true` and `SNR_hat` | Full mismatch matrix | Table 1, Table 2 | The evaluation exposes the effect of SNR mismatch | Real local data |
| UA training | Bounded perturbation of estimated SNR | Original vs UA comparison | Table 1, Table 2 | UA improves average off-diagonal robustness on CIFAR10/AWGN/C32 | Real local data |
| Robustness evaluation protocol | `SNR_true x SNR_hat` grid | PSNR and MS-SSIM matrix | Table 1, Table 2 | Off-diagonal risk is a useful imperfect-CSI metric | Real local data |
| Broader semantic communication robustness | Same framework on Rayleigh/HR/rate-varying cases | Not completed | Future work only | No completed claim allowed | Missing |

# Table Schema

| Table | Purpose | Rows | Metrics | Data source | Replacement owner |
|---|---|---|---|---|---|
| Table 1 | Summarize diagonal and off-diagonal robustness | Original, UA, Delta | PSNR, MS-SSIM | `original_cifar10_awgn_C32_msssim_cpu.csv`, `ua_delta3_cifar10_awgn_C32_msssim_cpu.csv` | Author |
| Table 2 | Show representative mismatch cases | Selected SNR true/hat pairs | PSNR, delta PSNR | Same CSV files | Author |
| Table 3 | Required ablations and baseline plan | Delta ablation, matched-only fine-tuning, conservative SNR conditioning, no-SA/no-Channel-ModNet | Status and planned settings; future PSNR/MS-SSIM after runs | [TODO-RUN] no completed data yet | Author |
| Future Table C | Delta ablation results | `Delta={0,1,3,6}` | Diagonal PSNR/MS-SSIM, off-diagonal PSNR/MS-SSIM, trade-off | [TODO-RUN] | Author |
| Future Table D | Additional baseline comparison | Original, UA, matched-only fine-tuning, conservative SNR conditioning, no-SA/no-Channel-ModNet if available | Off-diagonal average, diagonal average, worst-case mismatch | [TODO-RUN] | Author |
| Future Table A | Rayleigh robustness | TBD | PSNR, MS-SSIM | Not available | Author |
| Future Table B | Rate adaptation ablation | TBD | PSNR, MS-SSIM, CBR | Not available | Author |

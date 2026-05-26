# Chapter Provenance

## chapters/00-full-draft.md

- Status: DONE_WITH_CONCERNS
- Writer: controller in current thread
- Reason separate chapter agents were not used: the user requested a single full-paper draft and did not request sub-agent delegation.
- Role in manuscript: complete compact SCI/conference-style draft for initial review.
- Required sources: `README.md`, `AGENT.md`, `net/network.py`, `main.py`, `eval_msssim_cpu.py`, `mismatch_results/original_cifar10_awgn_C32_msssim_cpu.csv`, `mismatch_results/ua_delta3_cifar10_awgn_C32_msssim_cpu.csv`.
- Argument chain: adaptive SwinJSCC assumes matched channel SNR and modulation SNR; practical CSI estimation creates mismatch; decoupling `SNR_true` and `SNR_hat` defines imperfect-CSI SwinJSCC; uncertainty-aware training approximates the imperfect-CSI risk; CIFAR10/AWGN/C32 results show improved off-diagonal robustness with small diagonal trade-off.
- Unresolved gaps: Rayleigh results, high-resolution image results, rate-adaptive ablation, repeated seeds, and uncertainty-width ablations are not available.
- Self-review: no unimplemented module is claimed; numerical claims are derived from local CSV files; broader robustness claims are limited to future work.

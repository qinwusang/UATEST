# Communication Letters Style Outline

## Title

Imperfect-CSI SwinJSCC: Robust Semantic Image Transmission under SNR Mismatch

## Abstract

75-100 words. State the perfect-CSI limitation, true-SNR/estimated-SNR decoupling, uncertainty-aware training, and the main CIFAR10/AWGN/C32 result.

## Index Terms

Semantic communications, deep joint source-channel coding, SwinJSCC, imperfect CSI, SNR mismatch.

## I. Introduction

Compact motivation: Deep JSCC and semantic image transmission; SwinJSCC adapts via Channel ModNet; practical CSI estimation creates SNR mismatch; contributions are imperfect-CSI formulation, UA training, and mismatch-matrix evaluation.

## II. System Model and Problem Formulation

Define image input, latent feature, power normalization, AWGN channel, decoder reconstruction, perfect-CSI risk, and imperfect-CSI risk.

## III. Imperfect-CSI SwinJSCC

Describe true SNR and estimated SNR decoupling; add UA training objective; include Lipschitz local-risk argument; cite Fig. 1 model framework.

## IV. Experimental Results

Use CIFAR10, AWGN, C=32, SNR set {1,4,7,10,13}. Report aggregate mismatch performance, representative mismatch cases, MS-SSIM trend, and limitations.

## V. Conclusion

Summarize the imperfect-CSI formulation and the observed robustness gain. State future work: Rayleigh, high-resolution images, variable CBR, repeated seeds, and uncertainty-width ablations.

## Format Rules

- English Communication Letters style.
- Abstract length: 75-100 words.
- Include Index Terms.
- Use IEEE numbered references.
- Include one model framework figure.
- Do not claim unimplemented modules or unsupported experiments.

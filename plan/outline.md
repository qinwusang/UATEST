# Communication Letters Style Outline

## Title

SNR-Mismatched SwinJSCC: Uncertainty-Aware Deep JSCC Under Imperfect SNR Estimation

## Abstract

75-100 words. State the perfect-SNR limitation, true-SNR/estimated-SNR decoupling, uncertainty-aware training, and the main CIFAR10/AWGN/C32 result.

## Index Terms

Deep joint source-channel coding, imperfect SNR estimation, semantic communications, SNR mismatch, SwinJSCC.

## I. Introduction

Compact motivation plus related-work positioning: Deep JSCC establishes end-to-end wireless image transmission; SwinJSCC adds Transformer/modulation capacity; robustness and domain randomization motivate training over deployment nuisance variables; this paper specializes that idea to SNR estimation mismatch.

## II. System Model and Problem Formulation

Define image input, latent feature, complex-symbol power normalization, AWGN channel, decoder reconstruction, perfect-SNR risk, and SNR-mismatch risk.

## III. Imperfect-CSI SwinJSCC

Describe true SNR and estimated SNR decoupling; add UA training objective; include bounded off-diagonal and full off-diagonal metrics; include Lipschitz local-risk argument; cite Fig. 1 model framework.

## IV. Experimental Results

Use CIFAR10, AWGN, C=32, SNR set {1,4,7,10,13}. Report heatmaps, trade-off table, combined Fig. 3 for metric-level and method-level robustness, qualitative reconstruction examples, auxiliary robust variants, Rayleigh preliminary check, and limitations.

## V. Conclusion

Summarize the imperfect-CSI formulation and the observed robustness gain. State future work: Rayleigh, high-resolution images, variable CBR, repeated seeds, and uncertainty-width ablations.

## Format Rules

- English Communication Letters style.
- Abstract length: 75-100 words.
- Include Index Terms.
- Use IEEE numbered references.
- Include one model framework figure.
- Do not claim unimplemented modules or unsupported experiments.

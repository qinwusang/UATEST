# Robust Imperfect-CSI Training Commands

This note records the next experiment group after the basic UA-SwinJSCC result.

## Common Setup

Run from the remote repository root:

```bash
cd /root/autodl-tmp/UATEST
```

Common arguments:

```bash
--training \
--trainset CIFAR10 \
--channel-type awgn \
--C 32 \
--multiple-snr 1,4,7,10,13 \
--model "SwinJSCC_w/_SA" \
--delta-train 3 \
--snr-min 1 \
--snr-max 13 \
--checkpoint "/root/autodl-tmp/SwinJSCC/SwinJSCC/checkpoint/SwinJSCC w- SA/SwinJSCC_w_SA_AWGN_CIFAR10_snr_psnr_C32.model"
```

## Baseline UA

```bash
python main.py \
  --training \
  --trainset CIFAR10 \
  --channel-type awgn \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --model "SwinJSCC_w/_SA" \
  --ua-train \
  --delta-train 3 \
  --snr-min 1 \
  --snr-max 13 \
  --checkpoint "/root/autodl-tmp/SwinJSCC/SwinJSCC/checkpoint/SwinJSCC w- SA/SwinJSCC_w_SA_AWGN_CIFAR10_snr_psnr_C32.model"
```

## Tail-UA

```bash
python main.py \
  --training \
  --trainset CIFAR10 \
  --channel-type awgn \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --model "SwinJSCC_w/_SA" \
  --robust-train \
  --num-mismatch-samples 3 \
  --tail-alpha 0.8 \
  --lambda-tail 0.5 \
  --lambda-cons 0 \
  --delta-train 3 \
  --snr-min 1 \
  --snr-max 13 \
  --checkpoint "/root/autodl-tmp/SwinJSCC/SwinJSCC/checkpoint/SwinJSCC w- SA/SwinJSCC_w_SA_AWGN_CIFAR10_snr_psnr_C32.model"
```

## Cons-UA

```bash
python main.py \
  --training \
  --trainset CIFAR10 \
  --channel-type awgn \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --model "SwinJSCC_w/_SA" \
  --robust-train \
  --num-mismatch-samples 3 \
  --tail-alpha 0.8 \
  --lambda-tail 0 \
  --lambda-cons 0.1 \
  --delta-train 3 \
  --snr-min 1 \
  --snr-max 13 \
  --checkpoint "/root/autodl-tmp/SwinJSCC/SwinJSCC/checkpoint/SwinJSCC w- SA/SwinJSCC_w_SA_AWGN_CIFAR10_snr_psnr_C32.model"
```

## Tail+Cons-UA

```bash
python main.py \
  --training \
  --trainset CIFAR10 \
  --channel-type awgn \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --model "SwinJSCC_w/_SA" \
  --robust-train \
  --num-mismatch-samples 3 \
  --tail-alpha 0.8 \
  --lambda-tail 0.5 \
  --lambda-cons 0.1 \
  --delta-train 3 \
  --snr-min 1 \
  --snr-max 13 \
  --checkpoint "/root/autodl-tmp/SwinJSCC/SwinJSCC/checkpoint/SwinJSCC w- SA/SwinJSCC_w_SA_AWGN_CIFAR10_snr_psnr_C32.model"
```

## Evaluation

After each training run, evaluate the selected checkpoint with:

```bash
python main.py \
  --trainset CIFAR10 \
  --channel-type awgn \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --model "SwinJSCC_w/_SA" \
  --checkpoint "<trained_checkpoint_path>"
```

Then compute CPU MS-SSIM if needed:

```bash
python eval_msssim_cpu.py --csv "<mismatch_csv_path>"
```

## Notes

- `--robust-train` does not require `--ua-train`; it samples mismatch SNRs internally.
- `--num-mismatch-samples 3` is the first recommended setting because it keeps GPU memory and runtime moderate.
- `--lambda-tail 0.5` and `--lambda-cons 0.1` are the first-pass settings for the paper experiment.
- The main paper comparison should report Original, UA, Tail-UA, Cons-UA, and Tail+Cons-UA.

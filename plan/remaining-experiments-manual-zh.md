# 剩余实验训练手册（不含重复 seed）

本文档只列出当前论文还值得补的实验。重复 seed 已按你的要求移除。实验按优先级排序，建议一组一组跑，不要同时开太多任务，避免 checkpoint 和结果文件混乱。

## 0. 先更新代码

远程服务器进入仓库：

```bash
cd /root/autodl-tmp/UATEST
git pull
```

本文档依赖新增参数：

```bash
--snr-hat-mode bounded|independent|fixed
--fixed-snr-hat <value>
```

含义：

```text
bounded     默认 UA：SNR_hat = clip(SNR_true + epsilon)
independent 随机 conditioning baseline：SNR_hat 从 SNR 集合独立采样
fixed       固定/保守 conditioning baseline：训练时 SNR_hat 固定为某个值
```

设置原版 checkpoint：

```bash
BASE_CKPT="/root/autodl-tmp/UATEST/checkpoint/SwinJSCC w- SA/SwinJSCC_w_SA_AWGN_CIFAR10_snr_psnr_C32.model"
ls "$BASE_CKPT"
```

如果 checkpoint 在旧目录，把路径改成：

```bash
BASE_CKPT="/root/autodl-tmp/SwinJSCC/SwinJSCC/checkpoint/SwinJSCC w- SA/SwinJSCC_w_SA_AWGN_CIFAR10_snr_psnr_C32.model"
ls "$BASE_CKPT"
```

统一基础设置：

```text
Dataset: CIFAR10
Channel: AWGN
Model: SwinJSCC_w/_SA
C: 32
SNR set: 1,4,7,10,13
Epoch: 10
Optimizer: Adam
Learning rate: 1e-4
Batch size: 256
```

每次训练后都要跑两种评估：

```text
1. main.py mismatch PSNR 评估：生成 mismatch CSV 和重构图目录
2. eval_msssim_cpu.py：生成同口径 PSNR / MS-SSIM / MS-SSIM(dB) CSV
```

论文主表优先使用 `eval_msssim_cpu.py` 的 CSV。

---

## 1. Delta 消融（最高优先级）

### 为什么必须做

当前主结果是 `Delta=3`。审稿人会问：为什么是 3 dB？如果没有消融，`Delta=3` 看起来像手调出来的。Delta 消融要证明：

```text
Delta=0: 只是继续 fine-tuning，不引入 CSI 误差
Delta=1: 小扰动
Delta=3: 当前主结果
Delta=6: 大扰动，观察是否损伤 matched-SNR
```

你已经有 `Delta=3` 结果，可以先补 `0,1,6`。如果想统一口径，也可以重新跑 `3`。

### 1.1 Delta=0

训练：

```bash
python main.py \
  --training \
  --trainset CIFAR10 \
  --model SwinJSCC_w/_SA \
  --channel-type awgn \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --ua-train \
  --snr-hat-mode bounded \
  --delta-train 0 \
  --snr-min 1 \
  --snr-max 13 \
  --max-epoch 10 \
  --exp-name ua-d0 \
  --checkpoint "$BASE_CKPT"
```

找 checkpoint：

```bash
UA_D0_CKPT=$(ls -t history/*ua-d0*/models/*_EP10.model | head -n 1)
echo "$UA_D0_CKPT"
```

评估：

```bash
python main.py \
  --trainset CIFAR10 \
  --model SwinJSCC_w/_SA \
  --channel-type awgn \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --exp-name ua-d0-eval \
  --checkpoint "$UA_D0_CKPT"
```

```bash
python eval_msssim_cpu.py \
  --trainset CIFAR10 \
  --model SwinJSCC_w/_SA \
  --channel-type awgn \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --exp-name ua-d0-eval \
  --checkpoint "$UA_D0_CKPT"
```

### 1.2 Delta=1

训练：

```bash
python main.py \
  --training \
  --trainset CIFAR10 \
  --model SwinJSCC_w/_SA \
  --channel-type awgn \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --ua-train \
  --snr-hat-mode bounded \
  --delta-train 1 \
  --snr-min 1 \
  --snr-max 13 \
  --max-epoch 10 \
  --exp-name ua-d1 \
  --checkpoint "$BASE_CKPT"
```

找 checkpoint：

```bash
UA_D1_CKPT=$(ls -t history/*ua-d1*/models/*_EP10.model | head -n 1)
echo "$UA_D1_CKPT"
```

评估：

```bash
python main.py \
  --trainset CIFAR10 \
  --model SwinJSCC_w/_SA \
  --channel-type awgn \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --exp-name ua-d1-eval \
  --checkpoint "$UA_D1_CKPT"
```

```bash
python eval_msssim_cpu.py \
  --trainset CIFAR10 \
  --model SwinJSCC_w/_SA \
  --channel-type awgn \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --exp-name ua-d1-eval \
  --checkpoint "$UA_D1_CKPT"
```

### 1.3 Delta=6

训练：

```bash
python main.py \
  --training \
  --trainset CIFAR10 \
  --model SwinJSCC_w/_SA \
  --channel-type awgn \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --ua-train \
  --snr-hat-mode bounded \
  --delta-train 6 \
  --snr-min 1 \
  --snr-max 13 \
  --max-epoch 10 \
  --exp-name ua-d6 \
  --checkpoint "$BASE_CKPT"
```

找 checkpoint：

```bash
UA_D6_CKPT=$(ls -t history/*ua-d6*/models/*_EP10.model | head -n 1)
echo "$UA_D6_CKPT"
```

评估：

```bash
python main.py \
  --trainset CIFAR10 \
  --model SwinJSCC_w/_SA \
  --channel-type awgn \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --exp-name ua-d6-eval \
  --checkpoint "$UA_D6_CKPT"
```

```bash
python eval_msssim_cpu.py \
  --trainset CIFAR10 \
  --model SwinJSCC_w/_SA \
  --channel-type awgn \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --exp-name ua-d6-eval \
  --checkpoint "$UA_D6_CKPT"
```

### Delta 消融进论文怎么用

最终表格建议：

```text
Method | Delta | Matched PSNR | Off-diag PSNR | Worst PSNR | Off-diag MS-SSIM(dB)
UA-D0
UA-D1
UA-D3
UA-D6
```

最想看到的结果是：

```text
Delta=0 不能明显提升 mismatch
Delta=1 有一定提升
Delta=3 平均表现最好
Delta=6 可能 worst-case 更好，但 matched-SNR 下降更多
```

---

## 2. 定性重构图（不一定要重新训练）

### 为什么要做

现在论文有 heatmap 和表，但缺少图像重构对比。通信/图像论文通常需要展示结果不是只在数值上好看。

推荐展示三组 mismatch：

```text
SNR_true=1,  SNR_hat=7
SNR_true=4,  SNR_hat=10
SNR_true=13, SNR_hat=1
```

每组展示：

```text
Original image | Original reconstruction | UA-Delta3 reconstruction | Tail+Cons-UA reconstruction
```

### 原版重构图

```bash
python main.py \
  --trainset CIFAR10 \
  --model SwinJSCC_w/_SA \
  --channel-type awgn \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --exp-name qual-original \
  --checkpoint "$BASE_CKPT"
```

### UA-Delta3 重构图

如果已有 UA checkpoint，直接设置：

```bash
UA_D3_CKPT=$(ls -t history/*ua-d3*/models/*_EP10.model | head -n 1)
echo "$UA_D3_CKPT"
```

再评估：

```bash
python main.py \
  --trainset CIFAR10 \
  --model SwinJSCC_w/_SA \
  --channel-type awgn \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --exp-name qual-ua-d3 \
  --checkpoint "$UA_D3_CKPT"
```

### Tail+Cons-UA 重构图

```bash
TAIL_CONS_CKPT=$(ls -t history/*tail-cons-ua-d3*/models/*_EP10.model | head -n 1)
echo "$TAIL_CONS_CKPT"
```

```bash
python main.py \
  --trainset CIFAR10 \
  --model SwinJSCC_w/_SA \
  --channel-type awgn \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --exp-name qual-tail-cons-ua-d3 \
  --checkpoint "$TAIL_CONS_CKPT"
```

### 去哪里找图

代码会生成类似目录：

```text
mismatch_vis_cifar10_<exp-tag>_ckpt-<checkpoint-name>/
```

每个 SNR pair 对应：

```text
true1_hat7_C32/batch0_original_grid.png
true1_hat7_C32/batch0_recon_grid.png
true4_hat10_C32/batch0_original_grid.png
true4_hat10_C32/batch0_recon_grid.png
true13_hat1_C32/batch0_original_grid.png
true13_hat1_C32/batch0_recon_grid.png
```

论文里不要直接塞整张 8xN grid，最好后续裁剪出同一张图像的对应 patch。当前先保存 grid，足够做素材。

---

## 3. 随机 SNR conditioning baseline

### 为什么要做

UA 的核心不是“随便随机 SNR”，而是“估计 SNR 围绕真实 SNR 有界扰动”。随机 SNR baseline 用来回答审稿人问题：

```text
这是不是普通 SNR augmentation？
```

这个 baseline 训练时：

```text
SNR_true 从 {1,4,7,10,13} 采样
SNR_hat 也从 {1,4,7,10,13} 独立采样
```

如果它不如 UA-Delta3，说明 bounded CSI-error modeling 更合理。

### 训练

```bash
python main.py \
  --training \
  --trainset CIFAR10 \
  --model SwinJSCC_w/_SA \
  --channel-type awgn \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --ua-train \
  --snr-hat-mode independent \
  --delta-train 3 \
  --snr-min 1 \
  --snr-max 13 \
  --max-epoch 10 \
  --exp-name random-hat \
  --checkpoint "$BASE_CKPT"
```

找 checkpoint：

```bash
RAND_HAT_CKPT=$(ls -t history/*random-hat*/models/*_EP10.model | head -n 1)
echo "$RAND_HAT_CKPT"
```

评估：

```bash
python main.py \
  --trainset CIFAR10 \
  --model SwinJSCC_w/_SA \
  --channel-type awgn \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --exp-name random-hat-eval \
  --checkpoint "$RAND_HAT_CKPT"
```

```bash
python eval_msssim_cpu.py \
  --trainset CIFAR10 \
  --model SwinJSCC_w/_SA \
  --channel-type awgn \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --exp-name random-hat-eval \
  --checkpoint "$RAND_HAT_CKPT"
```

---

## 4. 固定/保守 SNR conditioning baseline

### 为什么要做

另一个简单策略是不用估计值，直接固定一个保守 SNR。比如始终让 Channel ModNet 认为 SNR 是 1 dB，系统可能更保守，但可能损失高 SNR 性能。

建议跑两个：

```text
fixed SNR_hat = 1 dB   极保守
fixed SNR_hat = 7 dB   中间值
```

### 4.1 Fixed SNR_hat=1

训练：

```bash
python main.py \
  --training \
  --trainset CIFAR10 \
  --model SwinJSCC_w/_SA \
  --channel-type awgn \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --ua-train \
  --snr-hat-mode fixed \
  --fixed-snr-hat 1 \
  --delta-train 0 \
  --snr-min 1 \
  --snr-max 13 \
  --max-epoch 10 \
  --exp-name fixedhat1 \
  --checkpoint "$BASE_CKPT"
```

找 checkpoint：

```bash
FIXED1_CKPT=$(ls -t history/*fixedhat1*/models/*_EP10.model | head -n 1)
echo "$FIXED1_CKPT"
```

评估：

```bash
python main.py \
  --trainset CIFAR10 \
  --model SwinJSCC_w/_SA \
  --channel-type awgn \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --exp-name fixedhat1-eval \
  --checkpoint "$FIXED1_CKPT"
```

```bash
python eval_msssim_cpu.py \
  --trainset CIFAR10 \
  --model SwinJSCC_w/_SA \
  --channel-type awgn \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --exp-name fixedhat1-eval \
  --checkpoint "$FIXED1_CKPT"
```

### 4.2 Fixed SNR_hat=7

训练：

```bash
python main.py \
  --training \
  --trainset CIFAR10 \
  --model SwinJSCC_w/_SA \
  --channel-type awgn \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --ua-train \
  --snr-hat-mode fixed \
  --fixed-snr-hat 7 \
  --delta-train 0 \
  --snr-min 1 \
  --snr-max 13 \
  --max-epoch 10 \
  --exp-name fixedhat7 \
  --checkpoint "$BASE_CKPT"
```

找 checkpoint：

```bash
FIXED7_CKPT=$(ls -t history/*fixedhat7*/models/*_EP10.model | head -n 1)
echo "$FIXED7_CKPT"
```

评估：

```bash
python main.py \
  --trainset CIFAR10 \
  --model SwinJSCC_w/_SA \
  --channel-type awgn \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --exp-name fixedhat7-eval \
  --checkpoint "$FIXED7_CKPT"
```

```bash
python eval_msssim_cpu.py \
  --trainset CIFAR10 \
  --model SwinJSCC_w/_SA \
  --channel-type awgn \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --exp-name fixedhat7-eval \
  --checkpoint "$FIXED7_CKPT"
```

### 注意

固定 SNR baseline 的部署策略应主要看固定列，例如：

```text
fixedhat1 看 SNR_hat=1 这一列
fixedhat7 看 SNR_hat=7 这一列
```

不要简单拿完整 `SNR_true x SNR_hat` 矩阵的 off-diagonal 平均和 UA 比，因为固定策略实际不会在部署时切换到其它 `SNR_hat`。

---

## 5. Tail / Cons 超参数小消融

### 为什么做

你现在已有：

```text
Tail-UA: lambda_tail=0.5
Cons-UA: lambda_cons=0.1
Tail+Cons-UA: lambda_tail=0.5, lambda_cons=0.1
```

如果论文篇幅允许，补一个很小的超参数消融会更稳。优先跑两个极简实验，不要全网格：

```text
Tail stronger: lambda_tail=1.0, lambda_cons=0
Cons stronger: lambda_tail=0, lambda_cons=0.2
```

### 5.1 Tail stronger

```bash
python main.py \
  --training \
  --trainset CIFAR10 \
  --model SwinJSCC_w/_SA \
  --channel-type awgn \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --robust-train \
  --snr-hat-mode bounded \
  --delta-train 3 \
  --snr-min 1 \
  --snr-max 13 \
  --num-mismatch-samples 3 \
  --tail-alpha 0.8 \
  --lambda-tail 1.0 \
  --lambda-cons 0 \
  --max-epoch 10 \
  --exp-name tail-ua-d3-lt1 \
  --checkpoint "$BASE_CKPT"
```

```bash
TAIL_LT1_CKPT=$(ls -t history/*tail-ua-d3-lt1*/models/*_EP10.model | head -n 1)
echo "$TAIL_LT1_CKPT"
```

```bash
python eval_msssim_cpu.py \
  --trainset CIFAR10 \
  --model SwinJSCC_w/_SA \
  --channel-type awgn \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --exp-name tail-ua-d3-lt1-eval \
  --checkpoint "$TAIL_LT1_CKPT"
```

### 5.2 Cons stronger

```bash
python main.py \
  --training \
  --trainset CIFAR10 \
  --model SwinJSCC_w/_SA \
  --channel-type awgn \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --robust-train \
  --snr-hat-mode bounded \
  --delta-train 3 \
  --snr-min 1 \
  --snr-max 13 \
  --num-mismatch-samples 3 \
  --tail-alpha 0.8 \
  --lambda-tail 0 \
  --lambda-cons 0.2 \
  --max-epoch 10 \
  --exp-name cons-ua-d3-lc0p2 \
  --checkpoint "$BASE_CKPT"
```

```bash
CONS_LC02_CKPT=$(ls -t history/*cons-ua-d3-lc0p2*/models/*_EP10.model | head -n 1)
echo "$CONS_LC02_CKPT"
```

```bash
python eval_msssim_cpu.py \
  --trainset CIFAR10 \
  --model SwinJSCC_w/_SA \
  --channel-type awgn \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --exp-name cons-ua-d3-lc0p2-eval \
  --checkpoint "$CONS_LC02_CKPT"
```

这个小消融只用于说明趋势，不建议把论文主线变复杂。

---

## 6. Rayleigh 信道泛化

### 为什么做

AWGN 是最基础信道。Rayleigh 可以证明方法不只是在 AWGN 上有效。

### 先检查是否有 Rayleigh checkpoint

```bash
find checkpoint -iname "*rayleigh*C32*.model"
find checkpoint -iname "*Rayleigh*C32*.model"
```

如果没有 Rayleigh 原版 checkpoint，先不要跑这一组；用 AWGN checkpoint 直接测 Rayleigh 不公平，最多只能写成 stress test。

假设找到路径后设置：

```bash
RAY_CKPT="/path/to/rayleigh_cifar10_c32_checkpoint.model"
ls "$RAY_CKPT"
```

### Rayleigh Original

```bash
python main.py \
  --trainset CIFAR10 \
  --model SwinJSCC_w/_SA \
  --channel-type rayleigh \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --exp-name rayleigh-original-eval \
  --checkpoint "$RAY_CKPT"
```

```bash
python eval_msssim_cpu.py \
  --trainset CIFAR10 \
  --model SwinJSCC_w/_SA \
  --channel-type rayleigh \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --exp-name rayleigh-original-eval \
  --checkpoint "$RAY_CKPT"
```

### Rayleigh UA-Delta3

```bash
python main.py \
  --training \
  --trainset CIFAR10 \
  --model SwinJSCC_w/_SA \
  --channel-type rayleigh \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --ua-train \
  --snr-hat-mode bounded \
  --delta-train 3 \
  --snr-min 1 \
  --snr-max 13 \
  --max-epoch 10 \
  --exp-name rayleigh-ua-d3 \
  --checkpoint "$RAY_CKPT"
```

```bash
RAY_UA_CKPT=$(ls -t history/*rayleigh-ua-d3*/models/*_EP10.model | head -n 1)
echo "$RAY_UA_CKPT"
```

```bash
python eval_msssim_cpu.py \
  --trainset CIFAR10 \
  --model SwinJSCC_w/_SA \
  --channel-type rayleigh \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --exp-name rayleigh-ua-d3-eval \
  --checkpoint "$RAY_UA_CKPT"
```

---

## 7. 不同 C 值或高分辨率数据（有 checkpoint 再做）

### 为什么放最后

不同 C 值和高分辨率数据可以提升论文完整性，但成本更高，而且必须有匹配 checkpoint。C32 checkpoint 不能直接加载到 C16/C64 网络。

### 检查 C16/C64 checkpoint

```bash
find checkpoint -iname "*C16*.model"
find checkpoint -iname "*C64*.model"
```

如果有 C16 checkpoint，设置：

```bash
C16_CKPT="/path/to/C16_checkpoint.model"
```

然后按 UA-Delta3 跑：

```bash
python main.py \
  --training \
  --trainset CIFAR10 \
  --model SwinJSCC_w/_SA \
  --channel-type awgn \
  --C 16 \
  --multiple-snr 1,4,7,10,13 \
  --ua-train \
  --snr-hat-mode bounded \
  --delta-train 3 \
  --snr-min 1 \
  --snr-max 13 \
  --max-epoch 10 \
  --exp-name c16-ua-d3 \
  --checkpoint "$C16_CKPT"
```

C64 同理，把 `--C 16`、`C16_CKPT` 和 `exp-name` 改成 C64。

高分辨率 Kodak / CLIC 也必须先确认有 HR checkpoint 和数据路径。没有匹配 checkpoint 时，不建议作为当前 CL 主实验。

---

## 8. 最推荐的执行顺序

按这个顺序跑，论文收益最大：

```text
1. Delta=0
2. Delta=1
3. Delta=6
4. Qualitative reconstruction figure
5. Random SNR conditioning baseline
6. Fixed SNR_hat=1
7. Fixed SNR_hat=7
8. Tail stronger / Cons stronger 小消融
9. Rayleigh Original + UA（如果有 checkpoint）
10. C16/C64 或高分辨率（如果有 checkpoint）
```

如果时间不够，最低限度补：

```text
Delta=0, Delta=1, Delta=6, qualitative reconstruction figure
```

这四项能最直接提高论文说服力。

---

## 9. 结果文件交付给我时怎么发

每跑完一组，把下面两类 CSV 发回来：

```text
mismatch_results/*_mismatch.csv
mismatch_results/*_msssim_cpu.csv
```

优先发 `*_msssim_cpu.csv`，因为论文主表使用它。

如果是定性图，把对应目录下这几个文件发回来：

```text
true1_hat7_C32/batch0_original_grid.png
true1_hat7_C32/batch0_recon_grid.png
true4_hat10_C32/batch0_original_grid.png
true4_hat10_C32/batch0_recon_grid.png
true13_hat1_C32/batch0_original_grid.png
true13_hat1_C32/batch0_recon_grid.png
```

拿到结果后，我会继续更新 LaTeX 表格、heatmap、定性图和论文结论。

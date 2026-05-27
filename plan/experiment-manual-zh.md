# Imperfect-CSI SwinJSCC 实验手册

本文档给出当前论文版本需要补齐的完整实验路线、推荐命令和每组实验的意义。代码已经加入自动命名机制，训练、mismatch 评估、可视化和 CPU MS-SSIM 结果都会带上数据集、信道、模型、C、训练方法、扰动强度和 checkpoint 名称，原则上不需要再手动改文件名。

## 1. 实验目标

论文主线是验证：原版 SwinJSCC 默认 `SNR_hat = SNR_true`，而真实部署中估计 SNR 可能有误差。本项目把真实物理信道 SNR `SNR_true` 和调制网络使用的估计 SNR `SNR_hat` 解耦，并通过 uncertainty-aware training 提升 mismatch 条件下的鲁棒性。

因此实验必须回答四个问题：

1. 原版 checkpoint 在 `SNR_true x SNR_hat` mismatch 矩阵中是否明显脆弱。
2. UA 训练是否能提高 off-diagonal mismatch 区域的 PSNR / MS-SSIM。
3. 这种提升是否来自合理的 SNR 扰动建模，而不是偶然 fine-tuning。
4. 加入 tail-risk 和 consistency 约束后，是否能进一步改善 worst-case / average mismatch trade-off。

## 2. 自动命名规则

### 2.1 训练输出

训练时使用 `--exp-name` 指定一个短标签，例如 `ua-d3`、`tail-ua-d3`。训练目录会自动写成：

```bash
history/<timestamp>_<exp-name>_<dataset>_<channel>_<model>_C<C>_<method>/
```

模型文件会自动写成：

```bash
history/.../models/<timestamp>_<same-tag>_EP<epoch>.model
```

例子：

```text
history/20260527_153000_ua-d3_CIFAR10_awgn_SwinJSCC_w-SA_C32_ua_d3/
history/20260527_153000_ua-d3_CIFAR10_awgn_SwinJSCC_w-SA_C32_ua_d3/models/..._EP10.model
```

### 2.2 mismatch 评估输出

运行 `main.py` 的测试模式时，CSV 会自动写到：

```bash
mismatch_results/<experiment-tag>_ckpt-<checkpoint-name>_mismatch.csv
```

重构图会自动写到：

```bash
mismatch_vis_cifar10_<experiment-tag>_ckpt-<checkpoint-name>/
```

这可以避免 `mismatch_vis_cifar10/` 被不同 checkpoint 覆盖。

### 2.3 CPU MS-SSIM 输出

如果不手动传 `--output`，`eval_msssim_cpu.py` 会自动生成：

```bash
mismatch_results/<experiment-tag>_ckpt-<checkpoint-name>_msssim_cpu.csv
```

## 3. 统一实验设置

当前先只做论文最核心、最稳妥的一组设置：

```text
Dataset: CIFAR10
Channel: AWGN
Model: SwinJSCC_w/_SA
C: 32
SNR set: {1, 4, 7, 10, 13} dB
Delta: 3 dB
Optimizer: Adam
Learning rate: 1e-4
Batch size: 256
Seed: 42
Initialization: original CIFAR10 AWGN C32 checkpoint
```

推荐先训练 10 epoch。CIFAR10 下代码每 5 epoch 保存一次模型，并在保存时自动跑一次 mismatch 测试。若后续时间允许，再扩展到 20 或 30 epoch，用验证矩阵选择最好的 checkpoint。

## 4. 远程服务器准备

进入远程仓库：

```bash
cd /root/autodl-tmp/UATEST
git pull
```

设置原版 checkpoint 路径：

```bash
BASE_CKPT="/root/autodl-tmp/UATEST/checkpoint/SwinJSCC w- SA/SwinJSCC_w_SA_AWGN_CIFAR10_snr_psnr_C32.model"
```

如果你的 checkpoint 仍在旧目录，也可以改成：

```bash
BASE_CKPT="/root/autodl-tmp/SwinJSCC/SwinJSCC/checkpoint/SwinJSCC w- SA/SwinJSCC_w_SA_AWGN_CIFAR10_snr_psnr_C32.model"
```

先确认文件存在：

```bash
ls "$BASE_CKPT"
```

## 5. 实验 A：原版 checkpoint 的 mismatch 矩阵

### 命令

```bash
python main.py \
  --trainset CIFAR10 \
  --model SwinJSCC_w/_SA \
  --channel-type awgn \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --exp-name original-eval \
  --checkpoint "$BASE_CKPT"
```

```bash
python eval_msssim_cpu.py \
  --trainset CIFAR10 \
  --model SwinJSCC_w/_SA \
  --channel-type awgn \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --exp-name original-eval \
  --checkpoint "$BASE_CKPT"
```

### 为什么做

这是论文的 baseline，也是 Fig. 2 heatmap 的数据来源。它直接展示原版模型在 perfect CSI 之外的性能变化：对角线代表 `SNR_hat = SNR_true`，非对角线代表估计 SNR 与真实信道 SNR 不一致。

如果原版在非对角线区域明显下降，论文的问题动机才足够强。

## 6. 实验 B：UA-Delta3 主实验

### 训练命令

```bash
python main.py \
  --training \
  --trainset CIFAR10 \
  --model SwinJSCC_w/_SA \
  --channel-type awgn \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --ua-train \
  --delta-train 3 \
  --snr-min 1 \
  --snr-max 13 \
  --max-epoch 10 \
  --exp-name ua-d3 \
  --checkpoint "$BASE_CKPT"
```

### 找到最新 checkpoint

```bash
UA_CKPT=$(ls -t history/*ua-d3*/models/*_EP10.model | head -n 1)
echo "$UA_CKPT"
```

如果只训练到 5 epoch，则把 `EP10` 改成 `EP5`。

### 评估命令

```bash
python main.py \
  --trainset CIFAR10 \
  --model SwinJSCC_w/_SA \
  --channel-type awgn \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --exp-name ua-d3-eval \
  --checkpoint "$UA_CKPT"
```

```bash
python eval_msssim_cpu.py \
  --trainset CIFAR10 \
  --model SwinJSCC_w/_SA \
  --channel-type awgn \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --exp-name ua-d3-eval \
  --checkpoint "$UA_CKPT"
```

### 为什么做

这是当前论文的核心方法实验。训练时先采样真实信道 SNR：

```text
SNR_true in {1, 4, 7, 10, 13}
```

再采样估计误差：

```text
epsilon in [-3, 3]
SNR_hat = clip(SNR_true + epsilon, 1, 13)
```

物理信道仍使用 `SNR_true`，Encoder / Decoder 的 Channel ModNet 使用 `SNR_hat`。这比原版更接近 imperfect CSI 部署条件。

## 7. 实验 C：Tail-UA 鲁棒训练

### 训练命令

```bash
python main.py \
  --training \
  --trainset CIFAR10 \
  --model SwinJSCC_w/_SA \
  --channel-type awgn \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --robust-train \
  --delta-train 3 \
  --snr-min 1 \
  --snr-max 13 \
  --num-mismatch-samples 3 \
  --tail-alpha 0.8 \
  --lambda-tail 0.5 \
  --lambda-cons 0 \
  --max-epoch 10 \
  --exp-name tail-ua-d3 \
  --checkpoint "$BASE_CKPT"
```

### 找到最新 checkpoint

```bash
TAIL_CKPT=$(ls -t history/*tail-ua-d3*/models/*_EP10.model | head -n 1)
echo "$TAIL_CKPT"
```

### 评估命令

```bash
python main.py \
  --trainset CIFAR10 \
  --model SwinJSCC_w/_SA \
  --channel-type awgn \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --exp-name tail-ua-d3-eval \
  --checkpoint "$TAIL_CKPT"
```

```bash
python eval_msssim_cpu.py \
  --trainset CIFAR10 \
  --model SwinJSCC_w/_SA \
  --channel-type awgn \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --exp-name tail-ua-d3-eval \
  --checkpoint "$TAIL_CKPT"
```

### 为什么做

普通 UA 训练优化的是扰动分布下的平均风险。Tail-UA 进一步关注同一个 `SNR_true` 下多个 `SNR_hat` 样本中损失较高的部分，目标是改善 worst-case mismatch，而不只是改善平均指标。

论文中可以把它写成 tail-risk objective 或 CVaR-inspired robust training，但要注意措辞：这是工程上的鲁棒经验风险训练，不要声称它严格求解了所有不确定集合上的最优鲁棒解。

## 8. 实验 D：Consistency-UA 训练

### 训练命令

```bash
python main.py \
  --training \
  --trainset CIFAR10 \
  --model SwinJSCC_w/_SA \
  --channel-type awgn \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --robust-train \
  --delta-train 3 \
  --snr-min 1 \
  --snr-max 13 \
  --num-mismatch-samples 3 \
  --tail-alpha 0.8 \
  --lambda-tail 0 \
  --lambda-cons 0.1 \
  --max-epoch 10 \
  --exp-name cons-ua-d3 \
  --checkpoint "$BASE_CKPT"
```

### 找到最新 checkpoint

```bash
CONS_CKPT=$(ls -t history/*cons-ua-d3*/models/*_EP10.model | head -n 1)
echo "$CONS_CKPT"
```

### 评估命令

```bash
python main.py \
  --trainset CIFAR10 \
  --model SwinJSCC_w/_SA \
  --channel-type awgn \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --exp-name cons-ua-d3-eval \
  --checkpoint "$CONS_CKPT"
```

```bash
python eval_msssim_cpu.py \
  --trainset CIFAR10 \
  --model SwinJSCC_w/_SA \
  --channel-type awgn \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --exp-name cons-ua-d3-eval \
  --checkpoint "$CONS_CKPT"
```

### 为什么做

Consistency-UA 要求同一张图像在同一个真实信道 SNR 下，面对多个估计 SNR 输入时，重构结果不要剧烈漂移。它对应论文里“降低模型对单点 perfect-CSI conditioning 的依赖”的论点。

如果它能提高非对角线稳定性，同时对对角线损失较小，就可以作为第二个创新点：不仅做 SNR perturbation，还显式约束 mismatch reconstruction consistency。

## 9. 实验 E：Tail + Consistency 联合训练

### 训练命令

```bash
python main.py \
  --training \
  --trainset CIFAR10 \
  --model SwinJSCC_w/_SA \
  --channel-type awgn \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --robust-train \
  --delta-train 3 \
  --snr-min 1 \
  --snr-max 13 \
  --num-mismatch-samples 3 \
  --tail-alpha 0.8 \
  --lambda-tail 0.5 \
  --lambda-cons 0.1 \
  --max-epoch 10 \
  --exp-name tail-cons-ua-d3 \
  --checkpoint "$BASE_CKPT"
```

### 找到最新 checkpoint

```bash
TAIL_CONS_CKPT=$(ls -t history/*tail-cons-ua-d3*/models/*_EP10.model | head -n 1)
echo "$TAIL_CONS_CKPT"
```

### 评估命令

```bash
python main.py \
  --trainset CIFAR10 \
  --model SwinJSCC_w/_SA \
  --channel-type awgn \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --exp-name tail-cons-ua-d3-eval \
  --checkpoint "$TAIL_CONS_CKPT"
```

```bash
python eval_msssim_cpu.py \
  --trainset CIFAR10 \
  --model SwinJSCC_w/_SA \
  --channel-type awgn \
  --C 32 \
  --multiple-snr 1,4,7,10,13 \
  --exp-name tail-cons-ua-d3-eval \
  --checkpoint "$TAIL_CONS_CKPT"
```

### 为什么做

这是目前代码能支持的最强版本。它同时优化平均 mismatch、tail mismatch 和重构一致性。论文里可以把它作为增强方法或 ablation 中的最高配置。

如果效果优于 UA-Delta3，可以把文章从“只做 SNR mismatch 建模”增强为“imperfect-CSI risk formulation + tail/consistency robust training”。如果效果没有明显更好，也可以作为 negative ablation，说明简单 UA 已经足够有效。

## 10. 必须整理的结果表

每个模型都需要从 CSV 中统计以下指标：

```text
Matched PSNR: SNR_true = SNR_hat 的平均 PSNR
Off-diagonal PSNR: SNR_true != SNR_hat 的平均 PSNR
Worst PSNR: 整个 mismatch matrix 的最低 PSNR
Matched MS-SSIM(dB): 对角线平均 MS-SSIM(dB)
Off-diagonal MS-SSIM(dB): 非对角线平均 MS-SSIM(dB)
Worst MS-SSIM(dB): 整个矩阵最低 MS-SSIM(dB)
```

推荐论文表格：

```text
Method | Matched PSNR | Off-diag PSNR | Worst PSNR | Matched MS-SSIM(dB) | Off-diag MS-SSIM(dB) | Worst MS-SSIM(dB)
Original
UA-Delta3
Tail-UA
Consistency-UA
Tail+Consistency-UA
```

## 11. 必须生成的图

### Fig. 1：模型框图

展示两条 SNR 路径：

```text
SNR_true -> physical AWGN channel
SNR_hat  -> Encoder / Decoder Channel ModNet
```

这个图用于解释本项目相对原版 SwinJSCC 的核心修改。

### Fig. 2：Original mismatch heatmap

数据来自实验 A 的 PSNR CSV。用于证明 perfect-CSI checkpoint 在 off-diagonal mismatch 下存在脆弱性。

### Fig. 3：UA / robust method gain heatmap

推荐画：

```text
PSNR_gain = PSNR_method - PSNR_original
```

优先画 UA-Delta3。如果 Tail+Consistency 明显更好，再追加一张对比图或改成多子图。

### Fig. 4：重构图对比

推荐 case：

```text
SNR_true=1,  SNR_hat=7
SNR_true=4,  SNR_hat=10
SNR_true=13, SNR_hat=1
```

每组展示：

```text
Original image | Original SwinJSCC reconstruction | UA/robust reconstruction
```

重构图必须从不同自动命名目录中取，不能混用覆盖过的旧图。

## 12. 后续消融实验

如果时间允许，按优先级补：

### Delta 消融

```bash
--delta-train 0
--delta-train 1
--delta-train 3
--delta-train 6
```

目的：证明 `Delta=3` 不是随意选择，并观察扰动过大是否损伤 matched-SNR 性能。

### mismatch sample 数量消融

```bash
--num-mismatch-samples 1
--num-mismatch-samples 3
--num-mismatch-samples 5
```

目的：验证 Tail / Consistency 训练是否真的需要多个估计 SNR 样本。

### tail 权重消融

```bash
--lambda-tail 0.1
--lambda-tail 0.5
--lambda-tail 1.0
```

目的：观察 worst-case 鲁棒性和平均性能之间的 trade-off。

### consistency 权重消融

```bash
--lambda-cons 0.05
--lambda-cons 0.1
--lambda-cons 0.2
```

目的：观察重构稳定性约束是否过强，是否导致图像细节被过度平滑。

## 13. 写论文时的结论边界

可以严谨声称：

1. 本项目把 original SwinJSCC 的 perfect-CSI conditioning 改为 imperfect-CSI mismatch formulation。
2. UA 训练目标比 perfect-CSI 目标更匹配 imperfect-CSI 部署分布。
3. 在 CIFAR10 / AWGN / C=32 的已完成实验中，UA-Delta3 改善 off-diagonal mismatch 鲁棒性，同时 matched-SNR 损失较小。
4. Tail-risk 和 consistency 训练是当前代码新增的鲁棒增强方向，需要用完整实验矩阵确认最终收益。

不能声称：

1. 所有 SNR 点都逐点优于原版。
2. Rayleigh、Kodak、多 CBR 或真实无线信道已经验证。
3. 语义任务指标已经提升，因为当前指标仍是 PSNR 和 MS-SSIM。
4. 理论已经证明 UA 必然提升性能；理论只能证明训练目标与部署分布更一致。

## 14. 推荐执行顺序

先跑最关键的三组：

```text
1. Original eval
2. UA-Delta3 train + eval
3. Tail+Consistency-UA train + eval
```

如果第 3 组比 UA-Delta3 更好，再补 Tail-UA 和 Consistency-UA 分离消融。若第 3 组没有更好，论文主线仍保留 UA-Delta3，把 Tail / Consistency 写为探索性增强或 future work。

最后再补：

```text
1. heatmap
2. gain heatmap
3. reconstruction comparison
4. summary table
```

这样文章最短路径是闭环的：问题动机、方法、定量矩阵、可视化、复现实验信息都齐全。

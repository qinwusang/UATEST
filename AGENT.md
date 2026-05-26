# UA-SwinJSCC 论文论证型项目说明

## 核心结论

本项目相较原版 SwinJSCC 的主要创新不是替换 Swin Transformer 编解码主干，而是将原版隐含的 perfect CSI / perfect SNR input 假设改写为 imperfect CSI 下的 SNR mismatch 建模，并在此基础上引入 uncertainty-aware training 与 mismatch evaluation protocol。

原版 SwinJSCC 中，同一个 SNR 参数同时承担两个角色：

1. 物理信道参数：决定 AWGN/Rayleigh 信道中的噪声强度。
2. 模型认知参数：输入 Channel ModNet，决定编码器和解码器如何按信道状态调制特征。

这等价于假设系统永远知道真实信道状态，即估计 SNR 与真实 SNR 完全一致。本项目将二者显式拆分：

- `SNR_true` / `chan_param`：真实物理 SNR，用于信道加噪。
- `SNR_hat` / `mod_param`：估计 SNR，用于 Encoder/Decoder 的 Channel ModNet。

因此，本项目的创新可概括为：从 perfect-CSI SwinJSCC 推广到 imperfect-CSI SwinJSCC，使模型训练目标和评估协议更接近现实无线通信系统。

## 与原版 SwinJSCC 的关键差异

原版 SwinJSCC 根据 README 描述，提供四类模型：`SwinJSCC_w/o_SAandRA`、`SwinJSCC_w/_SA`、`SwinJSCC_w/_RA`、`SwinJSCC_w/_SAandRA`。其中 Channel ModNet 根据输入 SNR 自适应调制语义特征，Rate ModNet 根据目标传输率自适应选择特征维度。原版的核心假设是：输入给 Channel ModNet 的 SNR 就是真实信道 SNR。

本项目保留原有 SwinJSCC 主干和 ModNet 结构，主要增加以下机制：

- `net/network.py`：`SwinJSCC.forward` 新增 `mod_SNR`。`chan_param` 仍进入物理信道模块，`mod_param` 进入 Encoder/Decoder 的 Channel ModNet。
- `main.py`：新增 `--ua-train`、`--delta-train`、`--snr-min`、`--snr-max`，用于采样 SNR 估计误差。
- `main.py`：训练时采样 `SNR_true ~ multiple_snr`，并令 `SNR_hat = clip(SNR_true + eps)`，其中 `eps` 表示信道估计误差。
- `main.py`：测试时构造 `SNR_true x SNR_hat` mismatch 矩阵，用于观察真实信道与模型认知不一致时的性能变化。
- `eval_msssim_cpu.py`：增加 CPU 版 MS-SSIM mismatch 评估，避免 GPU/NVRTC 环境问题影响感知质量指标。

因此，本项目的创新强度应严谨表述为信道状态建模、训练目标和评估协议创新，而不是网络主干结构创新。

## 理论建模

设输入图像为 `x`，真实物理 SNR 为 `gamma`，SNR 估计误差为 `e`，估计 SNR 为：

```text
gamma_hat = gamma + e
```

设编码器为 `F_theta`，物理信道为 `T_gamma`，解码器为 `G_theta`，失真损失为 `L`。

原版 perfect-CSI SwinJSCC 优化的风险函数可以写为：

```text
R_perfect(theta)
  = E_{x,gamma}[
      L(G_theta(T_gamma(F_theta(x, gamma)), gamma), x)
    ]
```

这里同一个 `gamma` 同时进入编码器、信道和解码器。该目标只描述估计完全准确的理想情况。

现实 imperfect-CSI 部署风险应写为：

```text
R_imperfect(theta)
  = E_{x,gamma,e}[
      L(G_theta(T_gamma(F_theta(x, gamma + e)), gamma + e), x)
    ]
```

这里 `gamma` 决定真实信道噪声，`gamma + e` 决定模型内部调制策略。二者不一致正是实际系统中的 CSI estimation error / feedback delay / prediction error 所导致的 SNR mismatch。

## 理论证明

### 命题 1：perfect-CSI 目标是 imperfect-CSI 部署分布下的错设目标

若 `P(e != 0) > 0`，则 `R_perfect(theta)` 与 `R_imperfect(theta)` 对应不同的数据生成分布。

证明如下。`R_perfect` 等价于在退化误差分布 `P(e = 0) = 1` 下的风险：

```text
R_perfect(theta)
  = E_{x,gamma,e | e=0}[
      L(G_theta(T_gamma(F_theta(x, gamma + e)), gamma + e), x)
    ]
```

而真实部署条件满足 `P(e != 0) > 0`。除非损失函数对 SNR 估计误差完全不敏感，即对几乎所有 `x, gamma, e` 都有

```text
L(G_theta(T_gamma(F_theta(x, gamma + e)), gamma + e), x)
=
L(G_theta(T_gamma(F_theta(x, gamma)), gamma), x)
```

否则两个期望一般不相等。因此，原版目标仅覆盖 perfect CSI 的特殊情形，在 imperfect CSI 场景中属于目标分布错设。

### 命题 2：UA 训练是 imperfect-CSI 风险的 Monte Carlo / ERM 近似

本项目训练时从 `multiple_snr` 中采样 `SNR_true`，再采样误差 `eps`，构造 `SNR_hat = clip(SNR_true + eps)`。该过程对应从经验分布中采样 `(x, gamma, e)`，并最小化：

```text
R_UA(theta)
  = E_{x,gamma,e ~ P_train}[
      L(G_theta(T_gamma(F_theta(x, gamma + e)), gamma + e), x)
    ]
```

当 `P_train(e)` 与部署误差分布 `P_deploy(e)` 一致，或至少其支撑集覆盖部署误差范围时，`R_UA` 是 `R_imperfect` 的一致经验近似。由经验风险最小化原则，训练样本数增加时，经验均值收敛到对应分布下的期望风险。

因此，相比只在 `e=0` 上训练的 `R_perfect`，UA 训练在统计目标上更匹配 imperfect-CSI 部署条件。

### 命题 3：SNR mismatch 风险随估计误差上界增长，UA 训练等价于局部风险平滑

假设复合重建函数

```text
H_theta(x, gamma, e)
  = G_theta(T_gamma(F_theta(x, gamma + e)), gamma + e)
```

在 `e` 的局部邻域内 Lipschitz 连续，且损失函数 `L` 对重建结果 Lipschitz 连续。则存在常数 `K_theta >= 0`，使得：

```text
|L(H_theta(x, gamma, e), x) - L(H_theta(x, gamma, 0), x)|
  <= K_theta |e|
```

这说明 perfect-CSI 训练只控制 `e=0` 单点上的风险，而部署误差增大时，性能偏离的上界随 `|e|` 增长。

UA 训练不再只优化单点风险，而是优化误差邻域上的平均风险：

```text
E_{e in U}[
  L(H_theta(x, gamma, e), x)
]
```

其中 `U` 是由 `delta-train` 与裁剪区间 `[snr-min, snr-max]` 定义的误差邻域。该目标在数学上相当于对 SNR 条件风险做局部平滑，使模型参数不只适配 `gamma_hat = gamma`，而是适配一族可能的 `gamma_hat`。这会降低模型对 perfect-CSI 单点输入的依赖，提高对估计误差的鲁棒性。

## 创新是否合理

该创新是合理的，原因有三点：

1. 物理合理性：实际无线系统只能获得估计 CSI，而非真实 CSI。将真实 SNR 与估计 SNR 分离，符合通信系统中的信道估计误差、反馈延迟和移动性问题。
2. 优化合理性：原版训练目标对应 `e=0` 的退化分布；UA 训练目标对应 `e` 非退化的现实分布。因此 UA 训练不是任意扰动，而是对部署风险的更准确建模。
3. 评估合理性：`SNR_true x SNR_hat` mismatch 矩阵能直接刻画模型在不同真实信道和不同估计偏差下的表现，比只测试对角线 `SNR_true = SNR_hat` 更能反映 imperfect-CSI 场景。

需要严谨限定的是：上述理论不能保证 UA 模型在所有 SNR、所有误差幅度、所有模型容量和所有训练轮数下逐点优于原版。理论上能严格证明的是：当部署环境存在 SNR 估计误差时，UA 目标函数比 perfect-CSI 目标函数更匹配部署分布。具体 PSNR / MS-SSIM 是否逐点提升，仍需通过实验矩阵验证。

## 本地实验证据定位

本项目已有以下结果文件，可作为论文中实验部分或附录证据：

- `mismatch_results/original_cifar10_awgn_C32_msssim_cpu.csv`
- `mismatch_results/ua_delta3_cifar10_awgn_C32_msssim_cpu.csv`
- `mismatch_results/CIFAR10_awgn_SwinJSCC_w-_SA_C32_mismatch.csv`

这些文件用于支持以下实验叙述：

- 原版模型在 `SNR_true != SNR_hat` 时会出现明显性能下降，尤其是低真实 SNR 被高估时。
- UA 训练后的模型在部分 mismatch 区域能缓解性能下降。
- 是否整体优于原版，应按完整矩阵、平均 mismatch 风险、最坏情况风险和对角线性能损失共同评价。

## 建议论文表述

推荐将本项目创新点表述为：

```text
We identify that the adaptive Channel ModNet in SwinJSCC implicitly assumes perfect SNR feedback, where the SNR used for semantic modulation is identical to the true physical channel SNR. To relax this idealized assumption, we formulate an imperfect-CSI SwinJSCC setting by decoupling the true channel SNR and the estimated modulation SNR. Based on this formulation, we introduce uncertainty-aware training over bounded SNR estimation errors and evaluate robustness using a full true-SNR by estimated-SNR mismatch matrix.
```

中文表述为：

```text
本文指出原版 SwinJSCC 的 Channel ModNet 隐含依赖 perfect CSI 假设，即用于语义特征调制的 SNR 与真实物理信道 SNR 完全一致。为放宽该理想化假设，本文将真实信道 SNR 与估计调制 SNR 显式解耦，建立 imperfect-CSI SwinJSCC 问题设置，并通过有界 SNR 估计误差上的不确定性感知训练提升模型在 SNR mismatch 下的鲁棒性。
```

## 后续工作边界

若继续增强论文贡献，应优先补充以下方向：

- 在 AWGN 与 Rayleigh 信道上分别报告 mismatch 矩阵。
- 报告 average mismatch risk、worst-case mismatch risk 和 diagonal performance loss。
- 比较不同 `delta-train` 的鲁棒性-对角线性能折中。
- 若要声称结构创新，需要进一步设计显式 uncertainty encoder、confidence-aware ModNet 或 distributional CSI input，而不仅是训练目标与评估协议改变。

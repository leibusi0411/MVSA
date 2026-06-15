# 无监督训练：数学推导

## 1. 符号约定

| 符号 | 维度 | 含义 |
|------|------|------|
| $g$ | $[B, D]$ | 冻结 CLIP 对原图 224×224 提取的全局 CLS 特征（L2 归一化） |
| $f$ | $[B, D]$ | STN 多视角融合特征（L2 归一化） |
| $\mathbf{v} = [v_1, \dots, v_N]$ | $[B, N, D]$ | N 个 STN 局部视角特征（各 L2 归一化） |
| $\bar{v} = \frac{1}{N}\sum_i v_i$ | $[B, D]$ | N 个局部视角特征的平均 |
| $\mathbf{T}$ | $[D, C]$ | 预计算的 C 个类别文本特征（L2 归一化） |

**Logits 计算**（未经温度缩放）：

$$
\begin{aligned}
z^g &= g \cdot \mathbf{T} \in \mathbb{R}^{B \times C} \quad &\text{全局 logits} \\
z^f &= f \cdot \mathbf{T} \in \mathbb{R}^{B \times C} \quad &\text{融合 logits} \\
z^v &= \mathbf{v} \cdot \mathbf{T} \in \mathbb{R}^{B \times N \times C} \quad &\text{多视角 logits}
\end{aligned}
$$

---

## 2. 阶段一：Warmup（epoch 0 ~ warmup_epochs-1）

让尚未训练的 STN 定位网络和融合模块从**冻结 CLIP 的全局知识**中快速学习。CLIP 编码器始终冻结。

### Teacher 分布（$\tau_t = 0.05$）

$$
P_g = \text{softmax}(z^g / \tau_t) \in \mathbb{R}^{B \times C}
$$

### Local 分支（$\tau_s^w = 0.09$）

每个局部视角向 CLIP 全局分布对齐。student 分布 $Q^v = \text{softmax}(z^v / \tau_s^w) \in \mathbb{R}^{B \times N \times C}$：

$$
\mathcal{L}_{\text{local}} = \text{KL}(P_g \parallel Q^v)
= \frac{1}{B \cdot N}\sum_{b=1}^{B}\sum_{i=1}^{N}\sum_{c=1}^{C} P_{g}(b,c) \cdot \log\frac{P_{g}(b,c)}{Q^v_{b,i,c}}
$$

即逐视角 KL 后取平均。$P_g(b,c)$ 为第 $b$ 个样本对第 $c$ 类的 teacher 概率，$Q^v_{b,i,c}$ 为第 $b$ 个样本第 $i$ 个视角对第 $c$ 类的 student 概率。

### Fused 分支（$\tau_s^w = 0.09$）

融合特征向 CLIP 全局分布对齐。student 分布 $Q^f = \text{softmax}(z^f / \tau_s^w) \in \mathbb{R}^{B \times C}$：

$$
\mathcal{L}_{\text{fused}} = \text{KL}(P_g \parallel Q^f)
= \frac{1}{B}\sum_{b=1}^{B}\sum_{c=1}^{C} P_{g}(b,c) \cdot \log\frac{P_{g}(b,c)}{Q^f_{b,c}}
$$

### 阶段一总损失

$$
\boxed{\mathcal{L}_{\text{stage1}} = \lambda_l \cdot \mathcal{L}_{\text{local}} + \lambda_f \cdot \mathcal{L}_{\text{fused}}}
$$

其中 $\lambda_l = \lambda_f = 1.0$。

---

## 3. 阶段二：对称一致性（epoch ≥ warmup_epochs）

阶段一结束后，fused 和 local 互相学习：

- **fused → local**：融合特征综合了全图 + N 个裁剪块的信息，比单个 local view 的认知更完整，教定位网络
- **local → fused**：N 个局部视角聚焦不同区域，平均后提供多视角共识，教融合模块
- **CLIP 锚定**：打破 fused ↔ local 闭环，防止两端坍缩到平凡解

### 3.1 Fused → Local（教定位网络，$\tau_d = 0.15$）

$$
P_f = \text{softmax}(z^f / \tau_d), \qquad
Q^v = \text{softmax}(z^v / \tau_d)
$$

$$
\mathcal{L}_{f \to l} = \text{KL}(P_f \parallel Q^v)
= \frac{1}{B \cdot N}\sum_{b=1}^{B}\sum_{i=1}^{N}\sum_{c=1}^{C} P_{f}(b,c) \cdot \log\frac{P_{f}(b,c)}{Q^v_{b,i,c}}
$$

### 3.2 Local → Fused（教融合模块，$\tau_d = 0.15$）

$$
\bar{z}^v = \frac{1}{N}\sum_{i=1}^{N} z^v_i, \qquad
P_v = \text{softmax}(\bar{z}^v / \tau_d), \qquad
Q^f = \text{softmax}(z^f / \tau_d)
$$

$$
\mathcal{L}_{l \to f} = \text{KL}(P_v \parallel Q^f)
= \frac{1}{B}\sum_{b=1}^{B}\sum_{c=1}^{C} P_{v}(b,c) \cdot \log\frac{P_{v}(b,c)}{Q^f_{b,c}}
$$

### 3.3 CLIP 锚定（防坍缩，$\tau_t = 0.05$，$\tau_s^w = 0.09$）

$$
P_g = \text{softmax}(z^g / \tau_t), \qquad
Q^f = \text{softmax}(z^f / \tau_s^w)
$$

$$
\mathcal{L}_{\text{clip}} = \text{KL}(P_g \parallel Q^f)
= \frac{1}{B}\sum_{b=1}^{B}\sum_{c=1}^{C} P_{g}(b,c) \cdot \log\frac{P_{g}(b,c)}{Q^f_{b,c}}
$$

### 3.4 阶段二总损失

$$
\boxed{\mathcal{L}_{\text{stage2}} = \mathcal{L}_{f \to l} + \mathcal{L}_{l \to f} + \lambda_{\text{clip}} \cdot \mathcal{L}_{\text{clip}}}
$$

其中 $\lambda_{\text{clip}} = 1.0$。

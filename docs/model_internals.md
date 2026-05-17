# 模型详解

这份文档讲清楚 Table 1 里每个模型在干什么、数据是怎么进去的，重点讲明白 **MetaTSF 系列**、**Chronos-Bolt ZS**、还有 **VGLG + KD（蒸馏）**。

---

## 1. 先说"时序预测"是个啥任务

简单一句话：**给你过去 96 小时的数据，让你预测未来 96 小时**（或者 192、336、720 小时）。

举例：ETTh1 数据集是某变压器的 7 个监测指标（油温、负载等），每小时一条数据。任务就是：
- 给你 **过去 4 天** 的 7 个指标
- 预测 **未来 4 天**（或一周、两周、一个月）每个指标的值

我们用 **MSE（均方误差）** 来打分，越低越好。

---

## 2. 一个 batch 长啥样

每个 batch 是 4 个 tensor，但其实就 2 个核心：

| 名字 | shape | 是啥 |
|---|---|---|
| `seq_x` | `(B, 96, N)` | **历史**：B 个样本，每个 96 步，每步 N 个变量 |
| `seq_y` | `(B, 96+pred, N)` | **真值**：前 96 步重叠历史，后 pred_len 步是要预测的未来 |
| `seq_x_mark` | `(B, 96, 4或5)` | 历史的时间戳特征（"现在是周几、几点钟"） |
| `seq_y_mark` | `(B, 96+pred, 4或5)` | 真值的时间戳特征 |

`B = batch_size = 32`，`N = 变量数`（ETT=7, Traffic=862）。

**重点**：模型只用 `seq_x` 做输入，输出 `(B, pred_len, N)` 跟 `seq_y` 的后 pred_len 步比 MSE。

举具体数字：ETTh1, pred_len=96：
- 输入：`(32, 96, 7)` — 32 个样本，每个看过去 96 小时的 7 个指标
- 输出：`(32, 96, 7)` — 预测未来 96 小时的 7 个指标
- loss：均方误差，越小越好

**归一化**：所有数据按训练集的均值和标准差做 z-score（变成均值 0 方差 1）。所以 MSE 这个数是在 z-score 空间，不能直接换算成"实际温度差几度"。

---

## 3. 八个 baseline 一句话简介

| 模型 | 它在干啥 |
|---|---|
| **DLinear** | 最简单：把时序拆成"趋势 + 周期"，各过一个 Linear，加起来。一个证明"复杂模型未必比简单 Linear 强"的 baseline |
| **LSTM** | 经典 RNN。每个变量独立喂进 LSTM，最后一步的 hidden state 投影成预测 |
| **GRU** | 跟 LSTM 一模一样，只是 cell 换成 GRU（更轻量） |
| **SegRNN** | 现代 RNN：把序列切成 segment，GRU 学每个 segment，再自回归生成未来 segment |
| **TimeMixer** | 纯 MLP：把序列下采样到多个尺度（96 → 48 → 24 → 12），每个尺度做"趋势/周期"分解，再融合 |
| **ModernTCN** | 纯 CNN：用很大的 conv kernel（31），加 grouped ConvFFN 混合维度和变量 |
| **iTransformer** | 把每个**变量当一个 token**（不是每个时间步），用 attention 学变量之间的关系 |
| **PatchTST** | 把每个变量切成**很多 patch**（每 16 个时间步一个 patch），patch 之间做 attention，channel-independent |

---

## 4. MetaTSF 是啥（我们的方法）

### 4.1 一个朴素的问题

你看上面这 8 个 baseline，有用 MLP 的（TimeMixer）、有用 CNN 的（ModernTCN）、有用 Transformer 的（PatchTST、iTransformer）。论文里都说"我用了 XX 所以表现好"。

但有个问题：**这些模型的"其他部分"也不一样**（归一化方式不一样、深度不一样、宽度不一样、训练步数不一样）。所以你比较"MLP vs CNN vs Transformer 谁好"的时候，**其实在比一堆别的东西**，不是真的在比 mixer 本身。

**MetaTSF 干的事**：搭一个**完全固定**的"骨架"，只换中间最关键的那个"混合器（TokenMixer）"。这样比较才公平。

### 4.2 MetaTSF 的骨架

每个 metatsf 模型走的流程：

```
输入: (32, 96, 7)           [32 个样本，96 步历史，7 个变量]
   ↓
RevIN 归一化               [每个样本独立 z-score，预测完会反归一]
   ↓
Linear: 96 → 128            [把"时间维"投影成 128 维隐藏向量]
   ↓
   ↓  ──── MetaTSF Block ────
   ↓   ┌─────────────────────────┐
   ↓   │ LayerNorm                │
   ↓   │ TokenMixer  ← 这里可以换 │  ← MLP / Conv / Attn / VGLG 之一
   ↓   │ + residual                │
   ↓   │ LayerNorm                │
   ↓   │ ChannelMLP                │  ← 所有变体共用，混合 7 个变量
   ↓   │ + residual                │
   ↓   └─────────────────────────┘
   ↓  （重复 2 次）
   ↓
Linear: 128 → 96            [投影回未来 96 步]
   ↓
RevIN 反归一化
   ↓
输出: (32, 96, 7)            [预测未来 96 步]
```

**整个流程里只有 `TokenMixer` 那一行不一样**。其他全相同。这才叫公平比较。

### 4.3 四个 TokenMixer 的核心区别

每个 TokenMixer 都要接受 `(B, L, N)` 输出 `(B, L, N)`。区别在"怎么混合"：

#### MLP Mixer
"沿时间维做个 MLP"。简单粗暴。

```python
def forward(x):  # x: (B, L, N)
    # 把 N（变量）当 batch 维，每个变量独立过 MLP
    h = Linear(L, 2L)(x.transpose) → GELU → Linear(2L, L)(...)
    return h.transpose
```

#### Conv Mixer
"沿时间维做个大核 depthwise conv"。每个变量独立卷。

```python
def forward(x):
    h = DepthwiseConv1d(N, kernel=31)(x)   # 每变量独立用 31 长度卷积核
    h = GELU(h)
    h = Conv1d(N, N, kernel=1)(h)          # 1x1 conv 跨变量轻混
    return h
```

#### Attn Mixer
"把每个变量当一个 token，做 attention"。iTransformer 风格。

```python
def forward(x):  # x: (B, L, N)
    h = x.transpose  # (B, N, L) -- N 个 token，每个长 L
    h, _ = MultiheadAttention(embed_dim=L)(h, h, h)
    return h.transpose
```

#### VGLG Mixer (**我们的方法**)

**核心思想**：不同变量有不同的"性格"。
- 有的变量很有"局部 pattern"（比如电力负荷一天一个周期）→ 用 conv 抓局部
- 有的变量很平滑、有长期趋势（比如温度）→ 用 low-rank 线性抓长程
- **让模型自己决定每个变量怎么混合**

```python
def forward(x):  # x: (B, L, N)
    # 路径 1：Local — 每变量独立的大核 conv
    h_local  = DepthwiseConv1d(N, kernel=31)(x)
    
    # 路径 2：Global — 低秩线性投影 (L → 8 → L)
    h_global = W2(W1(x))   # W1: L→8, W2: 8→L
    
    # Gate：看每个变量的 4 个统计量，自动学一个 0~1 的开关
    g = VariateGate(x)     # shape (B, 1, N)，每个变量一个 gate
    
    # 加权融合
    return g * h_local + (1 - g) * h_global
```

**Gate 是怎么学的**：
1. 对每个变量提取 4 个特征：均值、标准差、lag-1 自相关、主频幅值
2. 这 4 个数过一个小 MLP，sigmoid 输出 `g ∈ (0, 1)`
3. g 接近 1 → 走 conv 路径（local 主导）
4. g 接近 0 → 走 low-rank 路径（global 主导）

整个 gate 也是端到端学的，模型自己决定每个变量更倾向哪条路径。

---

## 5. Chronos-Bolt ZS — Amazon 的"万能预测模型"

### 5.1 什么是 Foundation Model

类似 GPT-4 在 NLP 里的角色：在**巨量数据**上预训练一次，然后什么任务都能直接做（**不用 fine-tune**）。

Chronos-Bolt-Base 是 Amazon 训的时序版本：
- **205M 参数**（很大，是 MetaTSF-VGLG 30k 的 6800 倍）
- 在 30 亿个时序"token"上预训练
- T5 架构（encoder-decoder）

### 5.2 ZS 是啥意思

**ZS = Zero-Shot**，意思是"看都没看过这个数据集就直接预测"。不训练、不 fine-tune，**只跑 inference**。

```python
pipe = ChronosBoltPipeline.from_pretrained("amazon/chronos-bolt-base")
predictions = pipe.predict_quantiles(inputs=context, prediction_length=96)
# 直接得到预测，没有任何训练
```

### 5.3 单变量怎么处理多变量

Chronos 是**单变量**模型（一次只看一条序列）。我们的数据是多变量（7-862 个变量），怎么办？

**答**：循环 N 次。每个变量独立喂进 Chronos，得到那个变量的预测，最后拼起来。

实际实现上为了快，我们把 `(B, L, N)` 摊平成 `(B*N, L)` 一次性丢进去：

```python
flat = context.permute(0,2,1).reshape(B*N, L)  # 把所有变量塞进 batch
chunks = split_into_chunks_of(flat, 256)        # 防 OOM 分块
for chunk in chunks:
    _, mean = pipe.predict_quantiles(inputs=chunk, prediction_length=pred_len)
    outs.append(mean)
return torch.cat(outs).reshape(B, N, pred_len).permute(0,2,1)
```

### 5.4 ZS 表现如何

- 在 **没见过类似数据** 的数据集上（如 ETTm1）：比训练过的小模型差很多（MSE 1.03 vs 0.41）
- 在 **见过类似数据** 的数据集上（如 Electricity）：跟训练过的小模型差不多甚至更好（MSE 0.218 vs 0.221）

---

## 6. KD — 知识蒸馏到底在干啥

### 6.1 直觉

想象你在准备考试：
- **学生**：你自己，记不住所有公式，但你聪明、灵活
- **老师**：一本超厚的教科书，无所不知，但你没法把整本书背下来

KD 就是"让学生模仿老师的解题思路"：你不光看正确答案（ground truth），还**看老师是怎么思考的**（teacher 的预测过程）。即使你最终用的是自己的方法，老师的提示能让你少走弯路。

在 ML 里：
- **学生 (Student)**：小模型，参数少、推理快、适合部署。我们的 MetaTSF-VGLG（30k 参数）
- **老师 (Teacher)**：大模型，预测好但慢、笨重。我们的 Chronos-Bolt-Base（205M 参数）

学生训练时**多看一份"老师的预测"作为辅助监督**。

### 6.2 经典 KD（分类任务）

最早 Hinton 在 2015 年提出 KD 是为图像分类用的：

```
loss = α · 分类损失（学生输出 vs 真实标签）
     + (1-α) · 蒸馏损失（学生输出分布 vs 老师输出分布）
```

学生不光要预测对类别，还要让"它有多确信"跟老师一致。

### 6.3 时序预测的 KD（我们的版本）

时序没有"类别概率分布"。怎么让学生模仿老师？我们用 **3 种不同角度** 比较学生和老师的预测：

| 损失项 | 公式 | 在比什么 |
|---|---|---|
| **MSE 主任务** | `MSE(s_pred, target)` | 学生预测 vs 真值（必须的） |
| **trend_kd** | FFT 前 8 个低频系数的 MSE | 比"长期趋势"是否一致 |
| **freq_kd** | 全频谱幅值的 MSE | 比"哪些频率分量重要"是否一致 |
| **diff_kd** | 一阶差分（相邻步的变化）的 MSE | 比"局部斜率方向"是否一致 |

最终 loss：
```python
loss = 1.0   * MSE_主任务
     + 0.0002 * trend_kd     # 低频对齐
     + 0.001  * freq_kd      # 频谱对齐
     + 1.0    * diff_kd      # 差分对齐
```

**为什么 lambda 差异这么大？**

因为 3 种 KD 损失的**数值量级**完全不同：
- MSE：~0.3
- trend_kd（FFT 系数 MSE）：~165（FFT 把能量集中到少数系数）
- freq_kd（全频谱）：~24
- diff_kd（相邻差）：~0.03（差分小）

我们手动校准 lambda 让每项贡献约等于 MSE 的 10%。否则 trend_kd 会主导整个 loss，学生就一心模仿老师，完全不顾真值了。

### 6.4 完整两阶段训练流程

#### 阶段 1：缓存老师预测（**只跑一次**）

让 Chronos 在所有训练样本上预测一遍，存到磁盘：

```python
for sample_x in training_set:
    teacher_pred = chronos.predict(sample_x, pred_len=96)
    save_to_disk(teacher_pred)   # cache/teacher/<dataset>_h96_train.pt
```

- ETT 数据集小：cache ~700 MB（fp32 够用）
- Traffic 数据集大：cache ~30 GB（必须用 fp16 + numpy mmap 才能装下）

为什么要缓存？因为蒸馏训练要跑 10 epoch，每个 epoch 全部样本过一遍，**老师推理一次就够了**，不用每 epoch 重跑。

#### 阶段 2：学生训练（每 epoch）

```python
for epoch in 1..10:
    for batch in train_loader:
        seq_x, seq_y, marks, teacher_pred = batch  # 从缓存读老师预测
        
        student_pred = MetaTSF_VGLG(seq_x, marks)
        
        target = seq_y[:, -pred_len:, :]
        loss = 1.0 * MSE(student_pred, target) \
             + 0.0002 * trend_kd(student_pred, teacher_pred) \
             + 0.001  * freq_kd(student_pred, teacher_pred) \
             + 1.0    * diff_kd(student_pred, teacher_pred)
        
        loss.backward()
        optimizer.step()
```

**Warmup**：前 1 epoch 只用 MSE，让学生先有个合理初始化。否则 KD 会一开始就把随机初始化的学生拉去模仿老师，结果学生连基础任务都学不会。

### 6.5 实测结果与有趣发现

| 数据集 | 老师 (Chronos ZS) | 学生 (VGLG no KD) | 学生 + KD | 结论 |
|---|---:|---:|---:|---|
| ETTm1 | **1.033** | 0.406 | 0.429 | **KD 反而变差 5%** |
| ETTm2 | **0.352** | 0.287 | 0.291 | **KD 略差 1%** |
| Electricity | 0.218 | 0.221 | (跑中) | 预期可能略好 |

**关键 insight**：**老师必须比学生强，KD 才有用**。在 ETT 上 Chronos 比学生差 2.5 倍，KD 等于"让一个研究生跟着一个初中生学"，结果当然变差。

这正是 roadmap 想验证的问题：**"架构创新（VGLG 设计）vs 蒸馏（KD），哪个更重要？"**

我们的初步答案：**在数据集分布不在 foundation model 训练分布内时（ETT 类小数据），架构创新远比蒸馏重要**。在数据集分布在 foundation 训练分布内时（Electricity 类工业数据），可能蒸馏有用。这是 paper 里一个很有意思的对比点。

---

## 7. 总结一张图

| 你想要的对比 | 应该看 Table 1 哪一行/列 |
|---|---|
| "我用 MLP / Conv / Transformer / VGLG，哪个 mixer 最好" | **MetaTSF-MLP / Conv / Attn / VGLG 4 行（同 backbone 公平对比）** |
| "我的小模型能不能打过 SOTA 大模型" | MetaTSF-VGLG vs PatchTST / iTransformer / TimeMixer |
| "Chronos 这种 foundation model 在我数据上 zero-shot 行不行" | _Chronos-Bolt_ ZS 行 |
| "如果用 Chronos 蒸馏，我的小模型会变更好吗" | **MetaTSF-VGLG + KD vs MetaTSF-VGLG** |

---

## 附录：代码文件位置（看哪改哪）

| 想看的东西 | 文件 |
|---|---|
| 数据加载 | [src/data/datasets.py](src/data/datasets.py) |
| 通用 trainer 循环 | [src/train/trainer.py](src/train/trainer.py) |
| 8 个 baseline | [src/models/baselines/*.py](src/models/baselines/) |
| MetaTSF 骨架 | [src/models/metatsf/backbone.py](src/models/metatsf/backbone.py) + [block.py](src/models/metatsf/block.py) |
| 4 个 TokenMixer | [src/models/metatsf/mixers/](src/models/metatsf/mixers/) |
| Chronos 老师 | [src/models/teacher.py](src/models/teacher.py) |
| 3 个 KD loss | [src/losses/distill.py](src/losses/distill.py) |
| 蒸馏 trainer（多了 KD 项） | [src/train/distill_trainer.py](src/train/distill_trainer.py) |
| 老师缓存脚本 | [scripts/cache_teacher_predictions.py](scripts/cache_teacher_predictions.py) |

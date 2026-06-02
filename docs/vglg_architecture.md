# VGLG 架构图 — 与代码 1:1 对应

> **VGLG** = **V**ariate-**G**ated **L**ocal-**G**lobal TokenMixer
> 嵌在共享的 MetaTSF 骨干里，只换 TokenMixer 这一块，跟 MLP/Conv/Attn 三个对照组做公平对比。

输入张量约定：`(B, L, N)` — `B` 批次、`L=seq_len=96`、`N` 变量数（数据集决定）。
输出 `(B, H, N)` — `H=pred_len ∈ {96, 192, 336, 720}`。

最终调优配方（[`metatsf_tuned2` commit 0200a57](../scripts/run_metatsf_tuned2.sh)）：30 epoch · lr=1e-3 · cosine LR · patience=8 · weight_decay=1e-3 · `d_model=128` · `n_layers=2` · `dropout=0.1`。Avg(7) MSE = **0.359**（排名 #7/12）。

---

## Tier 1 — 全模型 forward pass

> 文件 [src/models/metatsf/backbone.py:64-76](../src/models/metatsf/backbone.py#L64-L76)

```mermaid
flowchart TD
    IN(["<b>x_enc</b><br/>(B, L=96, N)"]):::data
    REVIN1["<b>RevIN.norm</b><br/>per-(instance,variate)<br/>z-score<br/><sub>layers/revin.py</sub>"]:::norm
    IPROJ["<b>input_proj</b><br/>Linear(L→d_model)<br/>along time axis<br/><sub>backbone.py:51</sub>"]:::learn
    BL1["<b>MetaTSFBlock #1</b><br/>(B, d_model, N)<br/><sub>block.py</sub>"]:::block
    BL2["<b>MetaTSFBlock #2</b><br/>(B, d_model, N)<br/><sub>block.py</sub>"]:::block
    OPROJ["<b>output_proj</b><br/>Linear(d_model→H)<br/>along time axis<br/><sub>backbone.py:62</sub>"]:::learn
    REVIN2["<b>RevIN.denorm</b><br/>restore μ, σ<br/><sub>layers/revin.py</sub>"]:::norm
    OUT(["<b>y_pred</b><br/>(B, H, N)"]):::data

    IN --> REVIN1
    REVIN1 -- "(B, L, N)" --> IPROJ
    IPROJ -- "(B, d_model, N)" --> BL1
    BL1 -- "(B, d_model, N)" --> BL2
    BL2 -- "(B, d_model, N)" --> OPROJ
    OPROJ -- "(B, H, N)" --> REVIN2
    REVIN2 --> OUT

    classDef data fill:#FFE4B5,stroke:#B8860B,stroke-width:2px,color:#000
    classDef norm fill:#E0FFFF,stroke:#008B8B,stroke-width:2px,color:#000
    classDef learn fill:#D8BFD8,stroke:#8B008B,stroke-width:2px,color:#000
    classDef block fill:#FFB6C1,stroke:#C71585,stroke-width:3px,color:#000
```

---

## Tier 2 — MetaTSFBlock（每一层）

> 文件 [src/models/metatsf/block.py:30-45](../src/models/metatsf/block.py#L30-L45)
> 结构：**Norm → TokenMixer → +residual → Norm → ChannelMLP → +residual**
> 这就是 MetaFormer 的标准 abstraction（Yu et al., CVPR 2022 Oral），所有 4 个 mixer 公用这套骨架。

```mermaid
flowchart TD
    X(["<b>x</b> in<br/>(B, d_model, N)"]):::data

    subgraph S1 [" ① TokenMixer 子层 (time-axis mixing) "]
        N1["<b>LayerNorm</b><br/>over N<br/><sub>block.py:36</sub>"]:::norm
        MIX["<b>VGLGMixer</b><br/>📦 ours<br/><sub>vglg_mixer.py</sub>"]:::ours
        ADD1((+)):::math
        N1 --> MIX --> ADD1
    end

    subgraph S2 [" ② ChannelMLP 子层 (variate-axis mixing) "]
        N2["<b>LayerNorm</b><br/>over N<br/><sub>block.py:38</sub>"]:::norm
        CMLP["<b>ChannelMLP</b><br/>Linear(N→2N) → GELU<br/>→ Dropout → Linear(2N→N)<br/><sub>block.py:12-27</sub>"]:::learn
        ADD2((+)):::math
        N2 --> CMLP --> ADD2
    end

    Y(["<b>x</b> out<br/>(B, d_model, N)"]):::data

    X --> N1
    X -.->|residual| ADD1
    ADD1 --> N2
    ADD1 -.->|residual| ADD2
    ADD2 --> Y

    classDef data fill:#FFE4B5,stroke:#B8860B,stroke-width:2px,color:#000
    classDef norm fill:#E0FFFF,stroke:#008B8B,stroke-width:2px,color:#000
    classDef learn fill:#D8BFD8,stroke:#8B008B,stroke-width:2px,color:#000
    classDef ours fill:#FF69B4,stroke:#8B0000,stroke-width:4px,color:#FFF
    classDef math fill:#F0E68C,stroke:#B8860B,stroke-width:2px,color:#000
```

> **设计要点**：MetaTSFBlock 把"沿时间轴混"（TokenMixer）和"沿变量轴混"（ChannelMLP）解耦成两个子层 — 各自前面挂 LayerNorm + 后面带残差。把 TokenMixer 换成 MLP/Conv/Attn/VGLG 就是 4 个 baseline 的全部差异，骨干一字不改。

---

## Tier 3 — VGLGMixer 核心（**我们的方法**）

> 文件 [src/models/metatsf/mixers/vglg_mixer.py:56-106](../src/models/metatsf/mixers/vglg_mixer.py#L56-L106)
> 三路并行：**Local conv（变量独立） + Global low-rank（长程混合） + Gate（按变量决定融合比例）**

```mermaid
flowchart TD
    X(["<b>x</b><br/>(B, L, N)<br/><i>L = d_model = 128</i>"]):::data

    subgraph LOC ["🔵 Local 路 — 短程, 变量独立"]
        L1["Transpose<br/>(B, L, N) → (B, N, L)"]:::math
        L2["<b>Conv1d depthwise</b><br/>k=31, groups=N<br/>padding=k//2<br/><sub>vglg_mixer.py:78-82</sub>"]:::learn
        L3["Transpose<br/>(B, N, L) → (B, L, N)"]:::math
        L1 --> L2 --> L3
    end

    subgraph GLO ["🟢 Global 路 — 长程, 低秩共享"]
        G1["Transpose<br/>(B, L, N) → (B, N, L)"]:::math
        G2["<b>W1: Linear(L→rank=8)</b><br/>降维瓶颈, no-bias<br/><sub>vglg_mixer.py:83</sub>"]:::learn
        G3["<b>W2: Linear(rank=8→L)</b><br/>升回时间, no-bias<br/><sub>vglg_mixer.py:84</sub>"]:::learn
        G4["Transpose<br/>(B, N, L) → (B, L, N)"]:::math
        G1 --> G2 --> G3 --> G4
    end

    subgraph GATE ["🌟 Gate 路 — 按变量决定混合比例"]
        VG["<b>VariateGate</b><br/>详见 Tier 4<br/><sub>vglg_mixer.py:16-53</sub>"]:::gate
        VG -.-> GOUT["g ∈ (0,1)<br/>(B, 1, N)<br/>每个变量一个标量"]:::math
    end

    HL(["<b>h_local</b><br/>(B, L, N)"]):::data
    HG(["<b>h_global</b><br/>(B, L, N)"]):::data
    G_NODE(["<b>g</b><br/>(B, 1, N)"]):::data

    MUL1(["× <b>g</b>"]):::math
    MUL2(["× (<b>1-g</b>)"]):::math
    SUM((+)):::math
    DROP["<b>Dropout(0.1)</b><br/><sub>vglg_mixer.py:92</sub>"]:::norm
    OUT(["<b>out</b><br/>(B, L, N)<br/>= g·local + (1-g)·global"]):::data

    X --> L1
    X --> G1
    X --> VG
    L3 --> HL --> MUL1
    G4 --> HG --> MUL2
    GOUT --> G_NODE
    G_NODE --> MUL1
    G_NODE --> MUL2
    MUL1 --> SUM
    MUL2 --> SUM
    SUM --> DROP --> OUT

    classDef data fill:#FFE4B5,stroke:#B8860B,stroke-width:2px,color:#000
    classDef norm fill:#E0FFFF,stroke:#008B8B,stroke-width:2px,color:#000
    classDef learn fill:#D8BFD8,stroke:#8B008B,stroke-width:2px,color:#000
    classDef math fill:#F0E68C,stroke:#B8860B,stroke-width:2px,color:#000
    classDef gate fill:#FF1493,stroke:#8B0000,stroke-width:4px,color:#FFF

    linkStyle 12 stroke:#FF1493,stroke-width:3px
    linkStyle 13 stroke:#FF1493,stroke-width:3px
```

### 公式（一行版）

```
out = g ⊙ Conv1d_depthwise_k31(x)  +  (1-g) ⊙ W₂(W₁(x))
       └──── Local（HF / 短程） ────┘    └─ Global（LF / 长程） ─┘
       g = VariateGate(x) ∈ (0,1)^N         按变量自适应
```

> **设计直觉**：能不能用一个 mixer 同时把"短窗口的局部抖动（如季节性）"和"长窗口的整体走势（如趋势）"都覆盖？
> — Local 走 depthwise conv k=31（每个变量独立的 1D 卷积，参数极少）；Global 走低秩瓶颈 L→8→L（鼓励学到全局慢变化）；每个变量按自己的统计特征决定信"局部"还是信"全局"。
> 消融（`gate_mode` 字段）：`fixed_1.0`=只用 Local；`fixed_0.0`=只用 Global；`fixed_0.5`=固定 50/50；`learned`=默认。

---

## Tier 4 — VariateGate（4 统计量 → 标量门）

> 文件 [src/models/metatsf/mixers/vglg_mixer.py:16-53](../src/models/metatsf/mixers/vglg_mixer.py#L16-L53)
> 输入 `(B, L, N)`，输出 `g ∈ (0,1)^(B, 1, N)`。
> 不引入新可学时间参数 — 仅看每条序列自己的 4 个统计特征。

```mermaid
flowchart TD
    X(["<b>x</b><br/>(B, L, N)"]):::data

    subgraph STATS ["💎 compute_stats — 4 个时序特征 (沿时间轴)"]
        S1["<b>① mean</b><br/>μ = x.mean(dim=1)<br/>整体水平"]:::stat
        S2["<b>② std</b><br/>σ = x.std(dim=1)<br/>波动大小"]:::stat
        S3["<b>③ lag-1 autocorr</b><br/>Σ(x_c[1:]·x_c[:-1])<br/>÷ Σ(x_c²)<br/>相邻时刻相关性"]:::stat
        S4["<b>④ dom freq energy</b><br/>max|FFT[1:]| / Σ|FFT|<br/>主频集中度"]:::stat
    end

    CAT["<b>concat</b><br/>(B, 4, N) → transpose → (B, N, 4)"]:::math
    LN["<b>LayerNorm(4)</b><br/>稳定 MLP 输入<br/><sub>vglg_mixer.py:21</sub>"]:::norm
    MLP1["<b>Linear(4 → 16)</b><br/><sub>gate_hidden=16</sub>"]:::learn
    GELU["GELU"]:::math
    MLP2["<b>Linear(16 → 1)</b>"]:::learn
    SIG["<b>σ (sigmoid)</b><br/>→ (0, 1)"]:::math
    OUT(["<b>g</b><br/>(B, 1, N)<br/>每变量一个门控值"]):::data

    X --> S1
    X --> S2
    X --> S3
    X --> S4
    S1 --> CAT
    S2 --> CAT
    S3 --> CAT
    S4 --> CAT
    CAT --> LN --> MLP1 --> GELU --> MLP2 --> SIG --> OUT

    classDef data fill:#FFE4B5,stroke:#B8860B,stroke-width:2px,color:#000
    classDef norm fill:#E0FFFF,stroke:#008B8B,stroke-width:2px,color:#000
    classDef learn fill:#D8BFD8,stroke:#8B008B,stroke-width:2px,color:#000
    classDef math fill:#F0E68C,stroke:#B8860B,stroke-width:2px,color:#000
    classDef stat fill:#98FB98,stroke:#2E8B57,stroke-width:3px,color:#000
```

### 4 个统计量的物理含义

| 统计量 | 公式 | 想表达什么 | 高值 → gate 倾向 |
|---|---|---|---|
| **① mean** μ | `x.mean(dim=1)` | 序列整体水平 | 中性，仅作 MLP 输入归一化锚点 |
| **② std** σ | `x.std(dim=1, unbiased=False)` | 波动大小 | 大 → 噪声多 → 可能偏好 Global（平滑） |
| **③ lag-1 autocorr** | `Σ(x_c[t]·x_c[t-1]) / Σ x_c²` | 相邻时刻相关性，越接近 1 越平滑 | 接近 1 → 平滑序列 → 偏好 Local（卷积够用） |
| **④ dom freq energy** | `max\|FFT[1:]\| / Σ\|FFT\|` | 主频在全频谱里的占比，越大越像规则周期 | 大 → 强周期 → 偏好 Local（k=31 卷积捕获季节性） |

> **为什么 4 个就够？** 这些是"决定该用短程还是长程"的最少充分量。再加更多（峰度、偏度、HF/LF 能量比、…）会让 gate MLP 参数膨胀但带来的判别力递减。

---

## 关键超参数（[`configs/model/metatsf_vglg.yaml`](../configs/model/metatsf_vglg.yaml)）

| 字段 | 值 | 含义 |
|---|---|---|
| `d_model` | 128 | 输入投影后的"时间"维 |
| `n_layers` | 2 | MetaTSFBlock 堆几层 |
| `dropout` | 0.1 | Block + Mixer 都用 |
| `channel_mlp_mult` | 2 | ChannelMLP 隐藏层宽 = 2N |
| `revin / affine` | true / true | RevIN 带可学习仿射 |
| `mixer.kernel_size` | **31** | Local depthwise conv 核大小 |
| `mixer.rank` | **8** | Global 低秩瓶颈 |
| `mixer.gate_hidden` | 16 | VariateGate MLP 隐藏层 |
| `mixer.gate_mode` | learned | 消融用：`fixed_0.0` / `fixed_0.5` / `fixed_1.0` |
| `mixer.gate_entropy_reg` | 0.0 | 可选 entropy reg（默认关，保留为消融用） |

训练配方（[`scripts/run_metatsf_tuned2.sh`](../scripts/run_metatsf_tuned2.sh)）：
`30 epoch · lr=1e-3 · cosine · patience=8 · weight_decay=1e-3 · batch_size=32`

---

## 参数量预算（h=96，ETTh1 N=7 为例）

| 模块 | 形状 | 参数量 |
|---|---|---|
| RevIN affine | 2·N | 14 |
| input_proj | L·d_model = 96·128 | 12,288 |
| MetaTSFBlock × 2: | | |
| ┊ LayerNorm × 2 | 2·2·N | 28 |
| ┊ VGLGMixer.local_conv (depthwise k=31, groups=N) | N·k + N | 224 |
| ┊ VGLGMixer.W1 (no bias) | d_model·rank = 128·8 | 1,024 |
| ┊ VGLGMixer.W2 (no bias) | rank·d_model = 8·128 | 1,024 |
| ┊ VGLGMixer.gate (LN(4) + 4·16 + 16·1 + 17) | — | 113 |
| ┊ ChannelMLP (N→2N→N) | 2·(2N·N + N) ≈ | 126 |
| × 2 layers → | | ≈ 5,038 |
| output_proj | d_model·H = 128·96 | 12,288 |
| **总计** | | **≈ 30 K** |

> **跟 Chronos-Bolt 老师对比**：VGLG ≈ **30K** 参数；Chronos-Bolt-Base ≈ **205M** 参数 — **约 6,800 × 小**。这就是为什么蒸馏（KD）有意义：把大模型在 foundation training distribution 上学到的归纳偏置压到 30K 的学生里。

---

## 代码 ↔ 图节点 速查表

| 图里的节点 | 对应代码 |
|---|---|
| Tier 1: RevIN | [`src/models/layers/revin.py`](../src/models/layers/revin.py) |
| Tier 1: input_proj / output_proj | [`src/models/metatsf/backbone.py:51,62`](../src/models/metatsf/backbone.py#L51) |
| Tier 1: 整体 forward | [`backbone.py:64-76`](../src/models/metatsf/backbone.py#L64-L76) |
| Tier 2: MetaTSFBlock | [`block.py:30-45`](../src/models/metatsf/block.py#L30-L45) |
| Tier 2: ChannelMLP | [`block.py:12-27`](../src/models/metatsf/block.py#L12-L27) |
| Tier 3: Local 路（Conv1d） | [`vglg_mixer.py:78-82`](../src/models/metatsf/mixers/vglg_mixer.py#L78-L82) |
| Tier 3: Global 路（W₁ / W₂） | [`vglg_mixer.py:83-84`](../src/models/metatsf/mixers/vglg_mixer.py#L83-L84) |
| Tier 3: 融合 g·local + (1-g)·global | [`vglg_mixer.py:105`](../src/models/metatsf/mixers/vglg_mixer.py#L105) |
| Tier 4: compute_stats（4 统计量） | [`vglg_mixer.py:29-47`](../src/models/metatsf/mixers/vglg_mixer.py#L29-L47) |
| Tier 4: VariateGate MLP | [`vglg_mixer.py:23-27`](../src/models/metatsf/mixers/vglg_mixer.py#L23-L27) |
| Gate 可视化 hook | [`vglg_mixer.py:94, 104`](../src/models/metatsf/mixers/vglg_mixer.py#L94) (`self._last_gate`) |
| Optional gate entropy 正则 | [`vglg_mixer.py:108-117`](../src/models/metatsf/mixers/vglg_mixer.py#L108-L117) |

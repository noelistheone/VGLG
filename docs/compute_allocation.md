# 算力分工

每个人跑分到的数据集 / 消融变体，**全 12 模型 × 4 horizon × 3 seed**。

## Table 1（主实验，8 数据集）

| 编号 | 设备 | 数据集 |
|---:|---|---|
| P1 | RTX 4090 | ETTm1, ETTm2 |
| P2 | 3× A6000 | Electricity, Traffic |
| P3 | MacBook i5 | 不跑（监控 + 写作） |
| P4 | MacBook M4 | ETTh1, f1weather |
| P5 | RTX 5060 | ETTh2, Weather |

## Table 2（消融，10 变体 × ETTh1+Weather+Electricity × 3 horizon × 3 seed）

| 编号 | 变体 |
|---:|---|
| P1 | full, fixed gate g=0.5, local only, global only, no RevIN |
| P2 | rank=4, rank=16, rank=32 |
| P5 | kernel=15, kernel=51 |
| P4 | 不跑消融，做 Figure 1/2/3 |

## 命令模板

```bash
python -m src.train.trainer model=<m> data=<d> train.pred_len=<h> seed=<s> tag=main
```

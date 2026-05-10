"""Self-contained Transformer encoder used by iTransformer / PatchTST baselines.

Adapted from THUML Time-Series-Library: layers/Transformer_EncDec.py and
layers/SelfAttention_Family.py. We strip everything we don't need (decoder,
ProbAttention, etc.) so the file is short and dependency-free.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FullAttention(nn.Module):
    def __init__(self, mask_flag: bool = False, attention_dropout: float = 0.1, output_attention: bool = False):
        super().__init__()
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
        _, S, _, _ = values.shape
        scale = 1.0 / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            mask = torch.triu(torch.ones(L, S, dtype=torch.bool, device=queries.device), diagonal=1)
            scores.masked_fill_(mask, float("-inf"))

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values).contiguous()
        return V, (A if self.output_attention else None)


class AttentionLayer(nn.Module):
    def __init__(self, attention: FullAttention, d_model: int, n_heads: int):
        super().__init__()
        d_keys = d_model // n_heads
        d_values = d_model // n_heads
        self.inner = attention
        self.q_proj = nn.Linear(d_model, d_keys * n_heads)
        self.k_proj = nn.Linear(d_model, d_keys * n_heads)
        self.v_proj = nn.Linear(d_model, d_values * n_heads)
        self.out_proj = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, q, k, v, attn_mask=None):
        B, L, _ = q.shape
        _, S, _ = k.shape
        H = self.n_heads
        q = self.q_proj(q).view(B, L, H, -1)
        k = self.k_proj(k).view(B, S, H, -1)
        v = self.v_proj(v).view(B, S, H, -1)
        out, attn = self.inner(q, k, v, attn_mask)
        return self.out_proj(out.view(B, L, -1)), attn


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model: int, d_ff: int | None = None,
                 dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer: nn.Module | None = None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        for layer in self.attn_layers:
            x, attn = layer(x, attn_mask=attn_mask)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, : x.size(1)]

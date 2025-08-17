from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [L, D]
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, L, D]

    def forward(self, x):
        # x: [B, L, D]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: [B, L, D]
        B, L, _ = x.size()
        q = self.w_q(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # [B,H,L,Dh]
        k = self.w_k(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = self.w_v(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B,H,L,L]
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # [B,H,L,Dh]
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.w_o(out), attn  # return weights for introspection

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int = 4*128, dropout: float = 0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, attn = self.mha(self.ln1(x), mask)
        x = x + self.dropout(attn_out)
        ff_out = self.ff(self.ln2(x))
        x = x + self.dropout(ff_out)
        return x, attn

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, n_layers: int = 2, n_heads: int = 4, max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff=4*d_model, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, ids, attn_mask=None, return_attn=False):
        # ids: [B, L]
        x = self.tok(ids)
        x = self.pe(x)
        all_attn = []
        for layer in self.layers:
            x, attn = layer(x, attn_mask)
            if return_attn:
                all_attn.append(attn)  # [B,H,L,L]
        x = self.ln(x)
        logits = self.head(x)  # [B, L, V]
        if return_attn:
            return logits, all_attn
        return logits

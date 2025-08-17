from __future__ import annotations
import torch
from torch.utils.data import Dataset
import numpy as np

class ToyMaskedLanguageDataset(Dataset):
    """
    Synthetic token sequences (integers). We randomly mask a few tokens and train to predict them.
    """
    def __init__(self, vocab_size=200, seq_len=32, size=20000, mask_ratio=0.15, seed=42):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.mask_id = vocab_size - 1
        self.size = size
        self.mask_ratio = mask_ratio

        self.data = rng.integers(1, vocab_size-1, size=(size, seq_len), endpoint=False)
        # reserve last id as [MASK]
        self.data = self.data.astype(np.int64)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = self.data[idx].copy()
        y = x.copy()

        # mask some positions
        mask = np.zeros_like(x, dtype=bool)
        n_mask = max(1, int(self.seq_len * self.mask_ratio))
        pos = np.random.choice(self.seq_len, size=n_mask, replace=False)
        mask[pos] = True
        x[mask] = self.mask_id

        # attention mask (no padding here)
        attn_mask = np.ones((self.seq_len, self.seq_len), dtype=np.int64)

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(attn_mask, dtype=torch.long),
            torch.tensor(mask, dtype=torch.bool),
        )

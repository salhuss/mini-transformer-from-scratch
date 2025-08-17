from __future__ import annotations
import typer
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from .model import TinyTransformer
from .data import ToyMaskedLanguageDataset

app = typer.Typer(help="Train tiny transformer on a masked-token toy task.")

@app.command()
def main(
    steps: int = typer.Option(1000),
    d_model: int = typer.Option(128),
    n_heads: int = typer.Option(4),
    n_layers: int = typer.Option(2),
    seq_len: int = typer.Option(32),
    vocab_size: int = typer.Option(200),
    batch_size: int = typer.Option(64),
    lr: float = typer.Option(2e-3),
    out_dir: str = typer.Option("runs"),
    device: str = typer.Option("cpu"),
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)
    ds = ToyMaskedLanguageDataset(vocab_size=vocab_size, seq_len=seq_len, size=50000)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = TinyTransformer(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads, max_len=seq_len).to(dev)
    opt = AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    it = iter(dl)
    pbar = tqdm(range(steps))
    model.train()
    for _ in pbar:
        try:
            xb, yb, attn_m, mask = next(it)
        except StopIteration:
            it = iter(dl)
            xb, yb, attn_m, mask = next(it)

        xb, yb, attn_m, mask = xb.to(dev), yb.to(dev), attn_m.to(dev), mask.to(dev)
        logits = model(xb, attn_mask=attn_m)
        # Only compute loss on masked positions
        logits_m = logits[mask]
        targets_m = yb[mask]
        loss = loss_fn(logits_m, targets_m)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        pbar.set_description(f"loss={loss.item():.3f}")

    torch.save({"model": model.state_dict(), "cfg": dict(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, n_layers=n_layers, seq_len=seq_len)}, f"{out_dir}/last.pt")
    print(f"✅ saved {out_dir}/last.pt")

if __name__ == "__main__":
    app()

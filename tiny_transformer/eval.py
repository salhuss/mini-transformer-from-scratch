from __future__ import annotations
import typer, torch
from .model import TinyTransformer
from .data import ToyMaskedLanguageDataset

app = typer.Typer(help="Quick evaluation probes.")

@app.command()
def main(ckpt: str = typer.Option("runs/last.pt"), n: int = 3, device: str = typer.Option("cpu")):
    ck = torch.load(ckpt, map_location=device)
    cfg = ck["cfg"]
    model = TinyTransformer(**{k: cfg[k] for k in ["vocab_size","d_model","n_layers","n_heads","seq_len"]})
    model.load_state_dict(ck["model"])
    model.eval()

    ds = ToyMaskedLanguageDataset(vocab_size=cfg["vocab_size"], seq_len=cfg["seq_len"], size=1000)
    for i in range(n):
        xb, yb, attn_m, mask = ds[i]
        with torch.no_grad():
            logits, attn = model(xb.unsqueeze(0), attn_m.unsqueeze(0), return_attn=True)
        pred = logits.argmax(-1).squeeze(0)
        print(f"\nSample {i+1}")
        print(" input :", xb.tolist())
        print(" target:", yb.tolist())
        print(" pred  :", pred.tolist())
        mpos = [i for i,b in enumerate(mask.tolist()) if b]
        print(" masked positions:", mpos)
        # show attention size info
        if attn:
            h = attn[0].shape[1]
            print(f" attn heads layer0: {h}")

if __name__ == "__main__":
    app()

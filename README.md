# Mini Transformer (From Scratch)

Tiny, well-commented Transformer **encoder** in PyTorch:
- Scaled dot-product attention
- Multi-head attention
- Add & Norm, Feed-Forward, Positional Encoding
- Trains on a toy **masked-language** task over synthetic tokens

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train tiny model (CPU-friendly)
python -m tiny_transformer.train --steps 800 --d-model 128 --n-heads 4 --n-layers 2
# Eval a few samples
python -m tiny_transformer.eval --ckpt runs/last.pt

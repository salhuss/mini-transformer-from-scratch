# test.py
import torch
from model import MiniTransformer

def main():
    vocab_size = 100
    d_model = 32
    num_heads = 4
    num_layers = 2
    max_len = 20

    # load model
    model = MiniTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        max_len=max_len,
    )

    # random input sequence of token IDs
    x = torch.randint(0, vocab_size, (1, max_len))

    print("Input token IDs:")
    print(x)

    # forward pass
    with torch.no_grad():
        out = model(x)

    print("\nOutput logits shape:", out.shape)
    print("Sample logits for first token:\n", out[0, 0, :5])  # first 5 logits

    # predicted tokens
    preds = out.argmax(dim=-1)
    print("\nPredicted token IDs:")
    print(preds)

if __name__ == "__main__":
    main()

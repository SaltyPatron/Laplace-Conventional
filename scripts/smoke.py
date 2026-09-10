#!/usr/bin/env python3
from __future__ import annotations

import torch

from laplace_conventional.model import ConventionalTransformer, ModelConfig
from laplace_conventional.tokenizer import ByteTokenizer


def main() -> None:
    torch.manual_seed(7)
    tok = ByteTokenizer()
    cfg = ModelConfig(vocab_size=tok.vocab_size, max_seq_len=64, n_layer=2, n_head=4, d_model=64, d_ff=256, gradient_checkpointing=False)
    model = ConventionalTransformer(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    records = [tok.graph_record("king", "is a", "monarch"), tok.graph_record("queen", "is a", "monarch"), tok.graph_record("rook", "is a", "chess piece"), tok.graph_record("bishop", "is a", "chess piece")]
    losses = []
    for step in range(24):
        ids = records[step % len(records)]
        ids = (ids + [tok.eos_id] * 65)[:65]
        x = torch.tensor([ids[:-1]], dtype=torch.long)
        y = torch.tensor([ids[1:]], dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        loss = model(x, y)["loss"]
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    print({"first_loss": losses[0], "last_loss": losses[-1], "decreased": losses[-1] < losses[0]})
    if not losses[-1] < losses[0]:
        raise SystemExit("loss did not decrease")


if __name__ == "__main__":
    main()

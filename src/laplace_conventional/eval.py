from __future__ import annotations

import argparse
import json
import math
import os

import torch

from .data import iter_realized_entities, iter_graph_examples, pack_blocks, batch_blocks
from .model import ConventionalTransformer, ModelConfig
from .tokenizer import ByteTokenizer


def load(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ConventionalTransformer(ModelConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dsn", default=os.environ.get("LAPLACE_PG_DSN"))
    ap.add_argument("--lm-batches", type=int, default=100)
    ap.add_argument("--graph-examples", type=int, default=100)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = ByteTokenizer()
    model = load(args.checkpoint, device)
    text_stream = (tok.encode(x.text, bos=True, eos=True) for x in iter_realized_entities(dsn=args.dsn, split="validation"))
    batches = batch_blocks(pack_blocks(text_stream, block_size=model.cfg.max_seq_len, pad_id=tok.pad_id), batch_size=1)
    losses = []
    for i, (x, y) in enumerate(batches):
        if i >= args.lm_batches:
            break
        losses.append(float(model(x.to(device), y.to(device))["loss"]))
    ppl = math.exp(sum(losses) / max(1, len(losses))) if losses else float("nan")
    exact = total = 0
    for ex in iter_graph_examples(dsn=args.dsn, split="validation"):
        prompt_ids = tok.graph_query(ex.subject, ex.relation)
        out = model.generate(torch.tensor([prompt_ids], device=device), max_new_tokens=128, temperature=0.7, top_k=32, eos_id=tok.eos_id)
        ids = out[0, len(prompt_ids):].cpu().tolist()
        if tok.eos_id in ids:
            ids = ids[:ids.index(tok.eos_id)]
        exact += int(tok.decode(ids).strip() == ex.object.strip())
        total += 1
        if total >= args.graph_examples:
            break
    print(json.dumps({"validation_perplexity": ppl, "graph_exact_match": exact / max(1, total), "graph_examples": total}))


if __name__ == "__main__":
    main()

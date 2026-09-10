from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .train import TokenShardDataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--dataset-report", required=True)
    ap.add_argument("--context", type=int, required=True)
    ap.add_argument("--max-batches", type=int, default=1000)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    report = json.loads(Path(args.dataset_report).read_text(encoding="utf-8"))
    ds = TokenShardDataset(Path(args.data), "validation", args.context, report["dtype"])
    loader = DataLoader(ds, batch_size=1)
    nll = 0.0
    tokens = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            count = batch["labels"].numel()
            nll += float(out.loss) * count
            tokens += count
    print(json.dumps({"validation_tokens": tokens, "nll_per_token": nll / max(tokens, 1), "perplexity": float(np.exp(min(nll / max(tokens, 1), 50.0)))}, indent=2))

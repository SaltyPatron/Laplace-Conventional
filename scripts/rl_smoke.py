#!/usr/bin/env python3
from __future__ import annotations

import torch
from torch.nn import functional as F

from laplace_conventional.model import ConventionalTransformer, ModelConfig
from laplace_conventional.rl import _sample_action, _sequence_logps_values, _reward
from laplace_conventional.tokenizer import ByteTokenizer


def main() -> None:
    torch.manual_seed(11)
    tok = ByteTokenizer()
    model = ConventionalTransformer(ModelConfig(vocab_size=tok.vocab_size, max_seq_len=64, n_layer=2, n_head=4, d_model=64, d_ff=256, gradient_checkpointing=False))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    prompt = torch.tensor([tok.graph_query("king", "is a")], dtype=torch.long)
    with torch.no_grad():
        _, actions, old_logps, old_values = _sample_action(model, prompt, tok.eos_id, max_new=8, temperature=1.0)
    reward = _reward(tok, actions, "monarch")
    returns = torch.tensor(reward).expand_as(old_values)
    advantage = (returns - old_values).detach()
    logps, values = _sequence_logps_values(model, prompt, actions)
    ratio = torch.exp(logps - old_logps)
    policy_loss = -torch.minimum(ratio * advantage, torch.clamp(ratio, 0.8, 1.2) * advantage).mean()
    value_loss = F.mse_loss(values, returns)
    loss = policy_loss + 0.5 * value_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    receipt = {"reward": reward, "policy_loss": float(policy_loss.detach()), "value_loss": float(value_loss.detach()), "grad_norm": float(grad_norm.detach()), "actions": actions.shape[1]}
    print(receipt)
    if not torch.isfinite(torch.tensor(receipt["grad_norm"])) or receipt["grad_norm"] <= 0:
        raise SystemExit("policy-gradient update produced no finite gradient")


if __name__ == "__main__":
    main()

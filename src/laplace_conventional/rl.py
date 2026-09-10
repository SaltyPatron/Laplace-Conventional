from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.nn import functional as F

from .data import iter_graph_examples
from .model import ConventionalTransformer, ModelConfig
from .tokenizer import ByteTokenizer


def _load_model(path: str, device: torch.device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ConventionalTransformer(ModelConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    return model, ckpt


def _sample_action(model, prompt: torch.Tensor, eos_id: int, max_new: int, temperature: float):
    generated = prompt
    action_tokens, old_logps, values = [], [], []
    for _ in range(max_new):
        out = model(generated[:, -model.cfg.max_seq_len :])
        logits = out["logits"][:, -1, :] / max(temperature, 1e-6)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        action_tokens.append(action)
        old_logps.append(dist.log_prob(action))
        values.append(out["values"][:, -1])
        generated = torch.cat((generated, action[:, None]), dim=1)
        if int(action.item()) == eos_id:
            break
    return generated, torch.stack(action_tokens, dim=1), torch.stack(old_logps, dim=1), torch.stack(values, dim=1)


def _sequence_logps_values(model, prompt, actions):
    seq = prompt
    logps, values = [], []
    for i in range(actions.size(1)):
        out = model(seq[:, -model.cfg.max_seq_len :])
        logits = out["logits"][:, -1, :]
        lp = F.log_softmax(logits, dim=-1).gather(1, actions[:, i:i+1]).squeeze(1)
        logps.append(lp)
        values.append(out["values"][:, -1])
        seq = torch.cat((seq, actions[:, i:i+1]), dim=1)
    return torch.stack(logps, dim=1), torch.stack(values, dim=1)


def _reward(tok: ByteTokenizer, actions: torch.Tensor, target: str) -> float:
    ids = actions[0].detach().cpu().tolist()
    if tok.eos_id in ids:
        ids = ids[:ids.index(tok.eos_id)]
    return 1.0 if tok.decode(ids).strip() == target.strip() else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Clipped actor-critic policy-gradient training on substrate graph completion.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dsn", default=os.environ.get("LAPLACE_PG_DSN"))
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--learning-rate", type=float, default=1e-6)
    ap.add_argument("--ppo-epochs", type=int, default=2)
    ap.add_argument("--clip-ratio", type=float, default=0.2)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = _load_model(args.checkpoint, device)
    model.train()
    tok = ByteTokenizer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    examples = iter_graph_examples(dsn=args.dsn, split="train")
    for step in range(args.steps):
        ex = next(examples)
        prompt_ids = tok.graph_query(ex.subject, ex.relation)
        prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            _, actions, old_logps, old_values = _sample_action(model, prompt, tok.eos_id, args.max_new_tokens, args.temperature)
        reward = _reward(tok, actions, ex.object)
        returns = torch.tensor(reward, device=device).expand_as(old_values)
        advantage = (returns - old_values).detach()
        for _ in range(args.ppo_epochs):
            logps, values = _sequence_logps_values(model, prompt, actions)
            ratio = torch.exp(logps - old_logps)
            policy_loss = -torch.minimum(ratio * advantage, torch.clamp(ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio) * advantage).mean()
            value_loss = F.mse_loss(values, returns)
            entropy_proxy = -logps.mean()
            loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy_proxy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        print(json.dumps({"event": "rl", "step": step + 1, "reward": reward, "policy_loss": float(policy_loss.detach()), "value_loss": float(value_loss.detach())}), flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt["model"] = model.state_dict()
    ckpt["rl_steps"] = int(ckpt.get("rl_steps", 0)) + args.steps
    torch.save(ckpt, out)


if __name__ == "__main__":
    main()

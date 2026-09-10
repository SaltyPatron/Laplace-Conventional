from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class ModelConfig:
    vocab_size: int = 265
    max_seq_len: int = 2048
    n_layer: int = 12
    n_head: int = 12
    d_model: int = 768
    d_ff: int = 3072
    dropout: float = 0.0
    gradient_checkpointing: bool = True

    def validate(self) -> None:
        if self.d_model % self.n_head:
            raise ValueError("d_model must be divisible by n_head")
        if self.max_seq_len < 2:
            raise ValueError("max_seq_len must be >= 2")


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(cfg.d_ff, cfg.d_model, bias=False),
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x)))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class ConventionalTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.value_head = nn.Linear(cfg.d_model, 1)
        self.lm_head.weight = self.tok.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def config_dict(self) -> dict:
        return asdict(self.cfg)

    def hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, t = input_ids.shape
        if t > self.cfg.max_seq_len:
            raise ValueError(f"sequence {t} exceeds max_seq_len={self.cfg.max_seq_len}")
        pos = torch.arange(t, device=input_ids.device)
        x = self.tok(input_ids) + self.pos(pos)[None, :, :]
        for block in self.blocks:
            if self.training and self.cfg.gradient_checkpointing:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.norm(x)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        h = self.hidden(input_ids)
        logits = self.lm_head(h)
        out = {"logits": logits, "values": self.value_head(h).squeeze(-1)}
        if labels is not None:
            out["loss"] = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return out

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, *, max_new_tokens: int, temperature: float = 1.0, top_k: int = 0, eos_id: int | None = None) -> torch.Tensor:
        for _ in range(max_new_tokens):
            x = input_ids[:, -self.cfg.max_seq_len :]
            logits = self(x)["logits"][:, -1, :] / max(temperature, 1e-6)
            if top_k > 0:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            input_ids = torch.cat((input_ids, nxt), dim=1)
            if eos_id is not None and bool((nxt == eos_id).all()):
                break
        return input_ids


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

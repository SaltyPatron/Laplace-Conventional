from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExecutionPlan:
    backend: str
    precision: str
    micro_batch_size: int
    parameter_count: int
    gpu_total_bytes: int
    gpu_budget_bytes: int
    ram_total_bytes: int
    ram_budget_bytes: int
    native_parameter_state_bytes: int
    activation_reserve_bytes: int
    native_estimated_gpu_bytes: int
    zero3_layer_working_set_bytes: int
    zero3_estimated_cpu_bytes: int
    native_fits: bool
    zero3_cpu_offload_fits: bool
    scope_preserved: bool
    reason: str


def _model_parameter_count(model_cfg: dict) -> int:
    value = int(model_cfg.get("parameter_estimate", 0))
    if value > 0:
        return value
    vocab = int(model_cfg["vocab_size"])
    hidden = int(model_cfg["hidden_size"])
    layers = int(model_cfg["num_hidden_layers"])
    ffn = int(model_cfg["intermediate_size"])
    return vocab * hidden + layers * (4 * hidden * hidden + 3 * hidden * ffn)


def _largest_layer_parameters(model_cfg: dict) -> int:
    value = int(model_cfg.get("largest_layer_parameter_estimate", 0))
    if value > 0:
        return value
    vocab = int(model_cfg["vocab_size"])
    hidden = int(model_cfg["hidden_size"])
    ffn = int(model_cfg["intermediate_size"])
    block = 4 * hidden * hidden + 3 * hidden * ffn
    embedding = vocab * hidden
    return max(block, embedding)


def derive_execution_plan(training_config: dict, hardware: dict, *, execution_policy: dict | None = None) -> dict:
    policy = dict(execution_policy or {})
    model_cfg = training_config["model"]
    params = _model_parameter_count(model_cfg)

    backend_policy = str(policy.get("backend", "auto"))
    if backend_policy not in {"auto", "native", "deepspeed_zero3_cpu_offload"}:
        raise ValueError(f"unsupported execution backend: {backend_policy}")

    precision = str(policy.get("precision", "fp32"))
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError(f"unsupported precision: {precision}")

    micro_batch = int(policy.get("micro_batch_size", 1))
    if micro_batch < 1:
        raise ValueError("micro_batch_size must be positive")

    gpu = hardware.get("gpu") or {}
    gpu_total = int(gpu.get("total_memory", 0))
    ram_total = int(hardware.get("ram_bytes", 0))
    gpu_fraction = float(policy.get("gpu_memory_fraction", 0.88))
    ram_fraction = float(policy.get("ram_memory_fraction", 0.75))
    if not 0.0 < gpu_fraction <= 1.0:
        raise ValueError("gpu_memory_fraction must be in (0,1]")
    if not 0.0 < ram_fraction <= 1.0:
        raise ValueError("ram_memory_fraction must be in (0,1]")

    gpu_budget = int(gpu_total * gpu_fraction)
    ram_budget = int(ram_total * ram_fraction)

    # Conservative full-training state estimate: weights/master copy, gradients,
    # and Adam first/second moments. This is deliberately not a throughput claim.
    state_bytes_per_parameter = int(policy.get("native_state_bytes_per_parameter", 16))
    native_state = params * state_bytes_per_parameter

    activation_fraction = float(policy.get("activation_reserve_fraction", 0.20))
    if not 0.0 <= activation_fraction < 1.0:
        raise ValueError("activation_reserve_fraction must be in [0,1)")
    activation_reserve = int(gpu_total * activation_fraction)
    native_estimated = native_state + activation_reserve
    native_fits = bool(gpu_total and native_estimated <= gpu_budget)

    largest_layer = _largest_layer_parameters(model_cfg)
    zero3_multiplier = float(policy.get("zero3_layer_working_set_multiplier", 3.0))
    zero3_layer_gpu = int(largest_layer * 4 * zero3_multiplier + activation_reserve)
    zero3_cpu_overhead = float(policy.get("zero3_cpu_state_overhead", 1.20))
    zero3_cpu = int(native_state * zero3_cpu_overhead)
    zero3_fits = bool(gpu_total and ram_total and zero3_layer_gpu <= gpu_budget and zero3_cpu <= ram_budget)

    if backend_policy == "native":
        backend = "native"
        reason = "explicit native backend policy"
    elif backend_policy == "deepspeed_zero3_cpu_offload":
        backend = "deepspeed_zero3_cpu_offload"
        reason = "explicit DeepSpeed ZeRO-3 CPU-offload backend policy"
    elif native_fits:
        backend = "native"
        reason = "full model/optimizer state estimate fits configured GPU budget"
    elif zero3_fits:
        backend = "deepspeed_zero3_cpu_offload"
        reason = "native state does not fit GPU budget; ZeRO-3 CPU offload fits measured GPU/RAM budgets"
    else:
        backend = "unavailable"
        reason = "derived model does not fit either native GPU state or configured ZeRO-3 CPU-offload budgets"

    if backend == "native" and not native_fits:
        reason += "; preflight will refuse launch rather than shrink the model"
    if backend == "deepspeed_zero3_cpu_offload" and not zero3_fits:
        reason += "; preflight will refuse launch rather than shrink the model"

    plan = ExecutionPlan(
        backend=backend,
        precision=precision,
        micro_batch_size=micro_batch,
        parameter_count=params,
        gpu_total_bytes=gpu_total,
        gpu_budget_bytes=gpu_budget,
        ram_total_bytes=ram_total,
        ram_budget_bytes=ram_budget,
        native_parameter_state_bytes=native_state,
        activation_reserve_bytes=activation_reserve,
        native_estimated_gpu_bytes=native_estimated,
        zero3_layer_working_set_bytes=zero3_layer_gpu,
        zero3_estimated_cpu_bytes=zero3_cpu,
        native_fits=native_fits,
        zero3_cpu_offload_fits=zero3_fits,
        scope_preserved=True,
        reason=reason,
    )
    return asdict(plan)


def validate_execution_plan(plan: dict) -> None:
    backend = plan.get("backend")
    if backend == "native" and not plan.get("native_fits"):
        raise RuntimeError(plan.get("reason", "native execution plan does not fit"))
    if backend == "deepspeed_zero3_cpu_offload" and not plan.get("zero3_cpu_offload_fits"):
        raise RuntimeError(plan.get("reason", "ZeRO-3 execution plan does not fit"))
    if backend not in {"native", "deepspeed_zero3_cpu_offload"}:
        raise RuntimeError(plan.get("reason", "no executable backend"))
    if not plan.get("scope_preserved"):
        raise RuntimeError("execution plan changed model scope")


def write_execution_plan(training_config_path: Path, hardware_path: Path, output: Path, *, execution_policy: dict | None = None) -> dict:
    training = json.loads(training_config_path.read_text(encoding="utf-8"))
    hardware = json.loads(hardware_path.read_text(encoding="utf-8"))
    plan = derive_execution_plan(training, hardware, execution_policy=execution_policy)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    return plan

from __future__ import annotations


def accelerator_for_plan(plan: dict, *, gradient_accumulation_steps: int, gradient_clip: float):
    from accelerate import Accelerator

    precision = str(plan["precision"])
    mixed_precision = {"fp32": "no", "fp16": "fp16", "bf16": "bf16"}[precision]
    backend = str(plan["backend"])
    if backend == "native":
        return Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
        )
    if backend == "deepspeed_zero3_cpu_offload":
        try:
            from accelerate import DeepSpeedPlugin
            import deepspeed  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "execution plan requires DeepSpeed ZeRO-3 CPU offload; run scripts/setup.sh with offload support"
            ) from exc
        plugin = DeepSpeedPlugin(
            zero_stage=3,
            offload_optimizer_device="cpu",
            offload_param_device="cpu",
            zero3_init_flag=True,
            zero3_save_16bit_model=precision != "fp32",
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_clipping=gradient_clip,
        )
        return Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            deepspeed_plugin=plugin,
        )
    raise RuntimeError(f"unsupported execution backend: {backend}")

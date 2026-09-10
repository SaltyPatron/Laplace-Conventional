from laplace_conventional.execution import derive_execution_plan, validate_execution_plan


def _training(params=100_000_000):
    return {
        "model": {
            "parameter_estimate": params,
            "vocab_size": 32000,
            "hidden_size": 768,
            "intermediate_size": 2048,
            "num_hidden_layers": 12,
        }
    }


def _hardware(gpu=11 << 30, ram=128 << 30):
    return {"ram_bytes": ram, "gpu": {"total_memory": gpu}}


def test_native_plan_does_not_resize_model():
    config = _training(100_000_000)
    plan = derive_execution_plan(config, _hardware())
    assert plan["parameter_count"] == config["model"]["parameter_estimate"]
    assert plan["scope_preserved"] is True
    assert plan["backend"] == "native"
    validate_execution_plan(plan)


def test_auto_uses_cpu_offload_without_reducing_scope():
    config = _training(700_000_000)
    plan = derive_execution_plan(config, _hardware(ram=128 << 30))
    assert plan["parameter_count"] == 700_000_000
    assert plan["native_fits"] is False
    assert plan["zero3_cpu_offload_fits"] is True
    assert plan["backend"] == "deepspeed_zero3_cpu_offload"
    assert plan["scope_preserved"] is True
    validate_execution_plan(plan)


def test_impossible_plan_refuses_instead_of_shrinking():
    config = _training(7_000_000_000)
    plan = derive_execution_plan(config, _hardware(ram=16 << 30))
    assert plan["parameter_count"] == 7_000_000_000
    assert plan["backend"] == "unavailable"
    assert plan["scope_preserved"] is True
    try:
        validate_execution_plan(plan)
    except RuntimeError:
        pass
    else:
        raise AssertionError("unexecutable plan must be rejected")

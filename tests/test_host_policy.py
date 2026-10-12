import builtins
import os
import socket
from unittest.mock import patch

import pytest

from laplace_conventional.hardware import probe
from laplace_conventional.host_policy import gpu_allowed, require_gpu_host
from laplace_conventional.runtime import accelerator_for_plan


@pytest.mark.parametrize("declared", ["", "another-host", "*"])
def test_ineligible_inventory_never_imports_gpu_runtime(declared):
    original = builtins.__import__
    gpu_imports = []

    def guarded(name, *args, **kwargs):
        if name in {"torch", "accelerate", "deepspeed"}:
            gpu_imports.append(name)
            raise AssertionError("Ineligible inventory reached a GPU runtime")
        return original(name, *args, **kwargs)

    with patch.dict(os.environ, {"LAPLACE_GPU_HOST": declared}), patch("builtins.__import__", guarded):
        assert not gpu_allowed()
        result = probe()
        assert result["gpu_allowed"] is False
        assert result["cuda"] is False
        assert "gpu" not in result
        assert not gpu_imports
        with pytest.raises(RuntimeError, match="before hardware access"):
            require_gpu_host()


@pytest.mark.parametrize("backend", ["native", "deepspeed_zero3_cpu_offload"])
def test_transferred_gpu_plan_cannot_initialize_accelerator(backend):
    with patch.dict(os.environ, {"LAPLACE_GPU_HOST": "another-host"}):
        with pytest.raises(RuntimeError, match="before hardware access"):
            accelerator_for_plan({"backend": backend, "gpu_total_bytes": 8 << 30},
                                 gradient_accumulation_steps=1, gradient_clip=1)


def test_explicit_local_eligibility_does_not_probe_hardware():
    with patch.dict(os.environ, {"LAPLACE_GPU_HOST": socket.gethostname().split(".")[0].upper()}):
        assert gpu_allowed()
        require_gpu_host()

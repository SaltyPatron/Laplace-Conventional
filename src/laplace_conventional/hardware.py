from __future__ import annotations

import json
import platform
from pathlib import Path

import psutil


def probe() -> dict:
    result = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": psutil.cpu_count(logical=True),
        "ram_bytes": psutil.virtual_memory().total,
        "cuda": False,
    }
    try:
        import torch
        result["torch"] = torch.__version__
        result["cuda"] = torch.cuda.is_available()
        if result["cuda"]:
            props = torch.cuda.get_device_properties(0)
            result["gpu"] = {
                "name": props.name,
                "total_memory": props.total_memory,
                "compute_capability": list(torch.cuda.get_device_capability(0)),
                "multi_processor_count": props.multi_processor_count,
            }
    except Exception as exc:
        result["torch_probe_error"] = f"{type(exc).__name__}: {exc}"
    return result


def write_probe(output: Path) -> dict:
    data = probe()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return data

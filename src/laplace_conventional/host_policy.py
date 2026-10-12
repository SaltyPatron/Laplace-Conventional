"""GPU eligibility is declared independently of detected hardware or model size."""
from __future__ import annotations

import os
import socket


def gpu_allowed() -> bool:
    declared = os.environ.get("LAPLACE_GPU_HOST", "").strip().casefold()
    return bool(declared) and declared == socket.gethostname().split(".")[0].casefold()


def require_gpu_host() -> None:
    if not gpu_allowed():
        raise RuntimeError(
            "GPU operation refused before hardware access: LAPLACE_GPU_HOST must "
            "explicitly name this eligible host. CPU-only hosts must leave it unset."
        )


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require", action="store_true")
    args = parser.parse_args()
    if args.require:
        require_gpu_host()
    print(json.dumps({"host": socket.gethostname(), "gpu_allowed": gpu_allowed()}))

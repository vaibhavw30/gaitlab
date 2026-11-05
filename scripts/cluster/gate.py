"""rf_gate: check the GPU facts docs/CLUSTER.md marks [unverified] before anything else uses a GPU.

Run inside the rf_gate job (scripts/cluster/rf_gate.sbatch):
1. the node's driver supports CUDA 13 (the locked torch and tensorrt-cu13 are CUDA 13 builds);
2. torch is a CUDA build, sees the GPU, and the device is a GH200 (plus one bf16 matmul);
3. ``import tensorrt`` and ``import torch_tensorrt`` succeed.
Prints one ``GATE <check> PASS|FAIL <detail>`` line per check, some ``INFO`` lines, and exits
non-zero if any check fails. No timing is printed.
"""

from __future__ import annotations

import importlib
import math
import platform
import re
import socket
import subprocess
import sys

results: list[bool] = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"GATE {name:<22} {'PASS' if ok else 'FAIL'}  {detail}", flush=True)
    results.append(ok)


def info(name: str, detail: object) -> None:
    print(f"INFO {name:<22} {detail}", flush=True)


def driver_and_max_cuda() -> tuple[str | None, str | None]:
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None
    m = re.search(r"Driver Version:\s*([\d.]+).*?CUDA Version:\s*([\d.]+)", out, re.S)
    return (m.group(1), m.group(2)) if m else (None, None)


def main() -> int:
    info("python", platform.python_version())
    info("machine", platform.machine())
    info("glibc", "-".join(platform.libc_ver()))

    driver, max_cuda = driver_and_max_cuda()
    major = int(max_cuda.split(".")[0]) if max_cuda else 0
    check("driver_supports_cuda13", major >= 13, f"driver {driver}, max CUDA {max_cuda}")

    try:
        import torch
    except Exception as e:  # report and carry on to the TensorRT checks
        check("torch_import", False, f"{type(e).__name__}: {e}")
    else:
        avail = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if avail else None
        check("torch_cuda_build", torch.version.cuda is not None,
              f"torch {torch.__version__}, built for CUDA {torch.version.cuda}")  # fmt: skip
        check("torch_cuda_available", avail, f"device_count {torch.cuda.device_count()}")
        check("device_is_gh200", bool(name) and "GH200" in name, f"{name}")
        if avail:
            x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
            ok = math.isfinite(float((x @ x).float().sum()))
            check("cuda_bf16_matmul", ok,
                  f"bf16 supported {torch.cuda.is_bf16_supported()}, "
                  f"capability {torch.cuda.get_device_capability(0)}, "
                  f"cudnn {torch.backends.cudnn.version()}")  # fmt: skip

    for mod in ("tensorrt", "torch_tensorrt"):
        try:
            m = importlib.import_module(mod)
            check(f"import_{mod}", True, str(getattr(m, "__version__", "?")))
        except Exception as e:
            check(f"import_{mod}", False, f"{type(e).__name__}: {e}")

    try:  # expected to fail: compute nodes have no internet (jobs must not fetch)
        socket.create_connection(("pypi.org", 443), timeout=5).close()
        info("internet", "pypi.org:443 reachable")
    except OSError as e:
        info("internet", f"none ({type(e).__name__})")

    passed = all(results)
    print(f"GATE RESULT {'PASS' if passed else 'FAIL'}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

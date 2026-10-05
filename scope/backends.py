"""Explicit local inference devices and available host resource measurements."""

import platform
import resource


def hardware_report():
    report = {"platform": platform.platform(), "machine": platform.machine(),
              "cpu_percent": None, "rss_bytes": None, "accelerator_utilization": None}
    try:
        import psutil
        report.update(memory_bytes=psutil.virtual_memory().total,
                      cpu_percent=psutil.cpu_percent(), rss_bytes=psutil.Process().memory_info().rss)
    except ImportError:
        report["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    try:
        import torch
        report.update(torch=torch.__version__, mps_available=torch.backends.mps.is_available(),
                      cuda_available=torch.cuda.is_available(), torch_threads=torch.get_num_threads())
    except ImportError:
        report.update(mps_available=None, cuda_available=None)
    return report


def resolve_device(requested="cpu"):
    import torch
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if requested not in ("cpu", "mps", "cuda"):
        raise ValueError("Device must be cpu, mps, cuda, or auto")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; select --device cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select --device cpu")
    return requested


def synchronize(device):
    import torch
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()

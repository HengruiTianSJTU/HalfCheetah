"""Capture an allowlisted system report; never dump environment variables."""
import importlib.metadata
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

report = {"timestamp": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(), "platform": platform.platform()}
for package in ["torch", "gymnasium", "mujoco", "stable-baselines3", "numpy"]:
    try:
        report[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        report[package] = None
for name, command in {"gpu": ["nvidia-smi"], "cpu": ["lscpu"], "ram": ["free", "-h"], "disk": ["df", "-h", "."], "git": ["git", "--version"], "ffmpeg": ["ffmpeg", "-version"]}.items():
    report[name] = subprocess.run(command, capture_output=True, text=True, timeout=30).stdout if shutil.which(command[0]) else "unavailable"
try:
    import torch
    report["cuda_available"] = torch.cuda.is_available()
    report["torch_cuda"] = torch.version.cuda
except ImportError:
    pass
Path("results/infrastructure").mkdir(parents=True, exist_ok=True)
Path("results/infrastructure/system.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))

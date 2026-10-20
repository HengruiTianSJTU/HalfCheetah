# HalfCheetah PPO course project

Target: Gymnasium HalfCheetah-v5, a continuous control benchmark for the intelligent wireless networking course. PPO concepts will be related to power control and resource allocation; this environment is not a wireless network simulation.

**Status: infrastructure preparation; no training results yet.** AutoDL TCP connectivity must be restored before hardware benchmarking and formal experiments. No performance claims are currently made.

## Environment setup

On the designated AutoDL node, preserve the supplied PyTorch/CUDA installation:

```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python scripts/diagnostics.py
python scripts/smoke.py
MUJOCO_GL=egl python scripts/smoke.py --video
```

Dependency ranges will be supplemented by a resolved lock after installation and validation on the compute node. Diagnostics record only allowlisted system details. Run commands from the repository root. Random smoke defaults to seed 42 and 1200 environment steps.

## Artifact policy

Commit source, configs, compact metrics, final figures and documentation. Checkpoints, videos, raw TensorBoard logs, virtual environments and credentials are excluded. Final video delivery and checkpoint download instructions will be added after successful training. Do not store SSH credentials in this repository.

See [execution plan](docs/EXECUTION.md) for experiment and delivery criteria.

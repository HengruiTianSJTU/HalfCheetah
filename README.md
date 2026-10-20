# HalfCheetah PPO course project

Target: Gymnasium HalfCheetah-v5, a continuous control benchmark for the intelligent wireless networking course. PPO concepts will be related to power control and resource allocation; this environment is not a wireless network simulation.

**Status: smoke, numerical tests and checkpoint/evaluation integration passed on AutoDL; reference benchmarking in progress.** Formal multi-seed results are not yet available.

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

Diagnostics record only allowlisted system details. Run commands from the repository root. Random smoke defaults to seed 42 and 1200 environment steps. CPU-only use is supported; EGL rendering requires working system EGL libraries. AutoDL uses the existing SOCKS network proxy from the jump host; its compute is independent of the PJLab worker.

## Training, evaluation and experiment matrix

```bash
python -m pytest -q
python -m halfcheetah.train --config configs/qa.yaml --name qa_pipeline
python -m halfcheetah.train --config configs/baseline.yaml
python scripts/evaluate.py results/hc_lr0.0003_clip0.2_seed42/checkpoint.pt --episodes 20
python scripts/run_experiments.py --workers 3 --timesteps 2000000
python scripts/finalize.py
```

Formal runs refuse to overwrite existing directories. The matrix uses learning rates 1e-4/3e-4/1e-3, clip ranges 0.1/0.2/0.3 with the baseline shared between the two ablations, seeds 42/43/44, plus an aggressive 0.03 learning-rate run: 16 unique experiments. Pilot benchmarking determines the final compute settings and budget; the actual saved configs are authoritative. Use tmux for long commands on AutoDL.

Training and deterministic evaluation are separate. Checkpoints include frozen observation statistics; evaluation reports raw returns. Final policy selection uses the median final-validation baseline seed, followed by 20 independent test episodes (20000–20019). Current checkpoints support policy reuse, not exact simulator/RNG continuation.

## Curves, video and offline tutorial

```bash
python scripts/report.py
MUJOCO_GL=egl python scripts/evaluate.py results/hc_lr0.0003_clip0.2_seed42/checkpoint.pt --episodes 1 --seed 20000 --video results/videos/halfcheetah_running.mp4 --output results/video_evaluation.csv
```

Open `tutorial/index.html` directly in a browser after generating the report. The HTML embeds real experiment data and uses local JavaScript/SVG with no CDN or Python server. Interactive controls cover PPO clipping, ablations, seeds, training/evaluation selection, hover and zoom. The video path is relative to the repository. `scripts/finalize.py` generates the video from the predeclared representative model rather than necessarily seed 42.

Optional browser QA needs Playwright and Chromium: `python -m pip install playwright`, `python -m playwright install --with-deps chromium`, then `python scripts/check_tutorial.py`.

Technical materials: [PPO implementation and defense notes](docs/ALGORITHM.md), [environment, Hopper and wireless-network comparison](docs/ENVIRONMENT.md).

## Artifact policy

Commit source, configs, compact metrics, final figures and documentation. Checkpoints, videos, raw TensorBoard logs, virtual environments and credentials are excluded. Final video delivery and checkpoint download instructions will be added after successful training. Do not store SSH credentials in this repository.

See [execution plan](docs/EXECUTION.md) for experiment and delivery criteria.

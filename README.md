# HalfCheetah PPO course project

Target: Gymnasium HalfCheetah-v5, a continuous control benchmark for the intelligent wireless networking course. PPO concepts will be related to power control and resource allocation; this environment is not a wireless network simulation.

**Results:** 24 completed custom-PPO experiments, 2,000,896 transitions each. The representative baseline model (lr=3e-4, clip=0.2, seed=45) achieves **3614.6 ±692.5 raw episodic return** across 20 untouched test seeds and **3.94 m/s** mean forward velocity. Baseline validation across five training seeds is **3791.6 ±1874.1** (training-seed SD). These are different measures of variability.

Learning rate 3e-4 has the highest observed group mean and evaluation AUC; 1e-3 is substantially poorer. Clip 0.1 and 0.2 have similar group means relative to their large seed variability, while 0.3 is lower here. A real transient collapse and an aggressive lr=0.03 run are preserved. See [full recorded results](docs/RESULTS.md), [defense guide](docs/DEFENSE.md), and the offline [interactive tutorial](tutorial/index.html).

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

For the exact Linux/CUDA dependency versions, install `python -m pip install -r requirements.lock.txt` followed by `python -m pip install --no-deps -e .`. The lock reuses AutoDL's existing `torch==2.5.1+cu124`. On other platforms use the portable project definition and install the appropriate PyTorch build.

## Measured machine benchmark

Tesla T4 16GiB, 8-vCPU / 56GiB container quota, Python 3.12.3, PyTorch 2.5.1+cu124. SB3 PPO used 100,352 actual transitions per case, a 2048-transition rollout batch and one torch thread.

| Device / environments / transport | Steps/s | Training wall time |
|---|---:|---:|
| CPU / 1 / synchronous | 442 | 227 s |
| CPU / 8 / synchronous | 677 | 148 s |
| CPU / 8 / subprocess | 1119 | 90 s |
| CUDA / 8 / synchronous | 600 | 167 s |

See [all eight measured cases](results/benchmark_summary.csv) for CPU/GPU utilization, memory, raw-environment throughput and evaluation. Faster single-job subprocess transport consumes more CPU: the custom PPO pilots measured ~580 synchronous versus ~377 asynchronous transitions per total CPU-second. The full matrix therefore runs eight independent one-thread CPU jobs, each with eight synchronous environments. This improves aggregate CPU use; it does not mean the T4 is unable to train PPO. A few short CPU integration/pilot checks ran on spare cores during reference benchmarking; timings describe this node's measured workload, not a universal hardware ranking.

## Training, evaluation and experiment matrix

```bash
python -m pytest -q
python -m halfcheetah.train --config configs/qa.yaml --name qa_pipeline
python -m halfcheetah.train --config configs/baseline.yaml --name reproduce_baseline_seed42
python scripts/evaluate.py results/reproduce_baseline_seed42/checkpoint.pt --episodes 20
```

Formal runs refuse to overwrite existing directories. A fresh clone already contains reference metrics and completed metadata, so run the full matrix in a code-only workspace; otherwise resumable orchestration will correctly skip those reference directories, which do not include checkpoints:

```bash
python scripts/prepare_reproduction.py
cd .local/reproduction
python -m pip install --no-deps -e .
python scripts/diagnostics.py
python scripts/smoke.py --video
python scripts/run_benchmarks.py
python scripts/benchmark_report.py
python scripts/benchmark_custom.py
python scripts/run_experiments.py --workers 8 --timesteps 2000000
python scripts/finalize.py
```

Use the installed environment above; the complete benchmark recipe targets the CUDA-enabled AutoDL instance. For another reproduction workspace pass `--name reproduction2` to the preparation script. No committed reference metric is overwritten. The initial matrix uses learning rates 1e-4/3e-4/1e-3, clip ranges 0.1/0.2/0.3 with the baseline shared between the two ablations, seeds 42/43/44, plus an aggressive 0.03 learning-rate run: 16 unique experiments. Before final evaluation, `finalize.py` expands any normal group whose initial final-return seed SD/abs(mean) exceeds 0.30 to five seeds (adds 45/46), using the same 2M-step budget. The saved initial and extension plans are authoritative. Use tmux for long commands on AutoDL.

Training and deterministic evaluation are separate. Checkpoints include frozen observation statistics; evaluation reports raw returns. Final selection first checks the median baseline model on 20 separate selection episodes (11000–11019). If it fails the declared stability rule, all normal candidates are validated and a stable policy is selected by its lower return percentile. Only then are 20 untouched test episodes (30000–30019) used. `final_motion.csv` independently audits forward displacement, velocity and reward decomposition on those same test seeds. Current checkpoints support policy reuse, not exact simulator/RNG continuation.

## Curves, video and offline tutorial

```bash
python scripts/report.py
MUJOCO_GL=egl python scripts/evaluate.py results/reproduce_baseline_seed42/checkpoint.pt --episodes 1 --seed 30000 --video results/videos/reproduced_running.mp4 --output results/reproduced_video_evaluation.csv
```

Open `tutorial/index.html` directly in a browser after generating the report. The HTML embeds real experiment data and uses local JavaScript/SVG with no CDN or Python server. Interactive controls cover PPO clipping, ablations, seeds, training/evaluation selection, hover and zoom. The video path is relative to the repository. `scripts/finalize.py` generates the video from the predeclared representative model rather than necessarily seed 42.

Optional browser QA needs Playwright and Chromium: `python -m pip install playwright`, `python -m playwright install --with-deps chromium`, then `python scripts/check_tutorial.py`.

Technical materials: [results chapter](docs/RESULTS.md), [presentation and defense guide](docs/DEFENSE.md), [PPO implementation and defense notes](docs/ALGORITHM.md), [environment, Hopper and wireless-network comparison](docs/ENVIRONMENT.md).

The [experiment protocol](docs/EXPERIMENT_PROTOCOL.md) defines budgets, seed aggregation, held-out evaluation and model selection. `python scripts/inspect_environments.py` records installed HalfCheetah/Hopper definitions. After final results are committed, `python scripts/fresh_qa.py` checks a new checkout and a new project dependency venv while reusing the supplied base PyTorch/CUDA.

## Artifact policy

Commit source, configs, compact metrics, final figures and documentation. Checkpoints, videos, raw TensorBoard logs, virtual environments and credentials are excluded. The complete local delivery includes `results/videos/halfcheetah_running.mp4` and `results/hc_lr0.0003_clip0.2_seed45/checkpoint.pt`; SHA-256 identifiers are recorded in `results/final_evaluation_summary.json`. GitHub contains the small final frame/poster, metrics and standalone Tutorial, but excludes model/video binaries. A fresh clone can inspect all curves offline immediately; to play the final video, copy the local delivery's `results/videos/` beside the cloned results, or reproduce training and run `finalize.py`. No public binary download is currently hosted. Do not store SSH credentials in this repository.

See [execution plan](docs/EXECUTION.md) for experiment and delivery criteria.

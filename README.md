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

Diagnostics record only allowlisted system details. Run commands from the repository root. Random smoke defaults to seed 42 and 1200 environment steps. CPU-only use is supported; EGL rendering requires working system EGL libraries.

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

Final validation: the [QA record](docs/QA.md) documents a fresh checkout/new project-dependency environment, all six tests, EGL smoke, training/evaluation, byte-identical plot/Tutorial regeneration, offline browser controls and artifact audits.

## Supplemental entropy-coefficient experiment

The [entropy ablation](docs/ENTROPY_ABLATION.md) fixes lr=3e-4 and clip=0.2 and
compares entropy coefficients 0 / 0.001 / 0.005 / 0.01 across seeds 42–46. It adds
15 CUDA training runs and reuses five compatible baselines from a separate
2026-10-30 reproduction; it does not change the original 24-run study above.
Its reproduced baseline is **3359.4 ±2067.7**, while coefficient 0.001 gives
**3422.1 ±1118.9** (training-seed SD). The small mean improvement, lower learning
curve AUC and mixed paired-seed outcomes do not establish a universally better setting.

Open the offline [entropy Tutorial](tutorial/entropy.html), inspect the
[group results](results/entropy_sweep_20261030/group_summary.csv), or regenerate
them with `python scripts/report_entropy.py` (no weights required). Run a new
matrix with `python scripts/run_entropy_experiments.py --device cpu --workers 1`;
see the supplement for CUDA, explicit baseline reuse and short-check commands.

## Artifact policy

Source, configurations, compact metrics, figures and documentation are committed.
The **4090 GPU 实现结果** now also publishes selected experiment directories' weights
and videos through **Git LFS**. The original CPU reference study's binary delivery
remains separate; its `results/videos/halfcheetah_running.mp4` and seed-45 policy
are not the 4090 reproduction's selected policy.

### 4090 GPU 实现结果：全部正式实验权重与视频

- [Video gallery (open locally)](results/video/index.html) and [download/use instructions](results/video/README.md).
- [26 learning-rate/clip checkpoints and their own logs](results/reproduction_4090_20261030/).
- [20 entropy checkpoint paths and logs](results/entropy_sweep_20261030/): 15 new models plus five reused controls.
- [Manifest linking weights, configurations and videos](results/video/4090_gpu/manifest.json).

These are **41 distinct trained policies across 46 checkpoint paths**. All formal
training used RTX 4090 CUDA; deterministic video inference uses CPU and EGL renders
the enlarged floor. The 26-run narrative and the entropy supplement are labeled
separately. New videos do not change any model-selection or final-test result.

Install Git LFS, then download the published objects from the repository root:

```bash
git lfs install
git lfs pull
```

Without Git LFS objects, CSV-based reports still regenerate; a small text pointer
cannot be loaded as a checkpoint or played as a video. Only the named 4090 artifact
directories are exempted from the binary ignore rules. Other checkpoints/videos,
raw console/TensorBoard logs, virtual environments and credentials remain excluded.

See [execution plan](docs/EXECUTION.md) for experiment and delivery criteria.

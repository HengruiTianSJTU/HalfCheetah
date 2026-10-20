# Execution plan and acceptance criteria

Initial source: main at 888c6a2. Repository contained only a Python .gitignore; no existing implementation was replaced.

1. Restore AutoDL connectivity; benchmark bidirectional file transfer and capture system diagnostics.
2. Install isolated environment without replacing supplied torch/CUDA; freeze resolved dependencies.
3. Pass seed-42 random smoke and EGL video validation.
4. SB3 reference: CPU/CUDA, 1/4/8 environments, approximately 100k steps each; measure throughput, elapsed time and utilization.
5. Implement and test PPO actor/critic, rollout, time-limit bootstrapping, GAE, clipped loss, updates, normalization, checkpointing and deterministic evaluation.
6. Pilot baseline to determine adequate training duration, then a shared-budget design: learning rates 1e-4/3e-4/1e-3 and clips 0.1/0.2/0.3, seeds 42/43/44 (15 unique runs using the shared baseline). Extend duration/seeds if noise prevents interpretation.
7. Preserve a naturally poor run or add an aggressive learning-rate experiment. Evaluate multiple episodes at fixed intervals on held-out seeds and final policy on additional seeds.
8. Generate all figures from recorded CSV metrics, final video, standalone interactive tutorial, algorithm/environment/wireless comparison and defense notes.
9. Fresh-install command validation, numerical tests, render/video checks, offline tutorial interaction checks, secret/size audit, clean Git and pushed commits.

Every formal run must record config, Git revision and dirty flag, UTC timestamp, seed, dependency versions, device, environment parameters, elapsed time and separate training/evaluation metrics. Never infer evaluation return from training reward. Preserve failed runs and report variance rather than a selected lucky episode.

Network: direct TCP timed out, but the jump host's existing SOCKS5 proxy successfully connected and authenticated AutoDL. Use SSH ProxyCommand, never store the SSH password. The PJLab worker wrapper and its compute resources are not used. An 8MiB incompressible round trip passed a byte-for-byte check (upload 0.84MiB/s; download 0.62MiB/s). This is adequate for code/metrics synchronization.

AutoDL reports Python 3.12.3, PyTorch 2.5.1+cu124, an idle Tesla T4 16GiB, cgroup CPU quota 8 and RAM quota 56GiB. Host-level lscpu/free show larger shared hardware; use container quotas for planning.

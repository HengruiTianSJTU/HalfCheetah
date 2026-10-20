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

Current blocker: direct TCP connection to the supplied AutoDL endpoint timed out before authentication. No remote diagnostics, transfer test or training has been performed.

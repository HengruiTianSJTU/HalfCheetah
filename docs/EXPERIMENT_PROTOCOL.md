# 实验设计与可复现性

## 固定设计

正常组采用 one-factor-at-a-time：lr ∈ {1e-4, 3e-4, 1e-3}、固定 clip=0.2；以及 clip ∈ {0.1,0.2,0.3}、固定 lr=3e-4。每组 seeds=42/43/44，共 15 个不同 run；共享 baseline 不重复计数。这不是完整 3×3 factorial 设计，不能推断 lr 与 clip 的交互作用。

另加 lr=0.03、clip=0.2、seed=42 的激进配置，保留真实日志作失败分析；如果它并不差，需要继续寻找有理论意义且实际较差的实验，不能预先把配置标签当成失败证据。

每个正式 run 请求 2,000,000 environment transitions；完整 rollout 对齐到 2,000,896。这是环境步数而非 gradient steps，也不是每个 vector environment 分别运行 200 万步。一个 batch 是 256×8=2048 transitions，10 epochs，每个 minibatch 64。lr 固定，不使用衰减 schedule。设备、采样后端与并发数由实际 benchmark 决定，见 `results/compute_decision.json`。

100k pilot 用于软件与训练 sanity check，不计入正式矩阵；SB3 的 100k 实验用于参考和机器吞吐比较，不能与 2M 自研结果直接声称算法性能优劣。

## Evaluation 与选择偏差

训练 episode 的原始 reward 由 `episodes.csv` 保存；`metrics.csv` 的 training_return_mean 是最近 100 个已结束 episode 的均值。训练包含随机高斯探索。

每约 100k transitions 用确定性均值动作，在 seeds=10000..10004 上各跑一回合。每个训练 run 的评估都冻结 observation statistics，不更新 reward statistics，不把训练中归一化过的奖励当成报告回报。`eval.csv` 记录每个回合的原始累计回报和长度。

报告的跨 seed SD 由每个训练 seed 的末次 evaluation 均值计算（sample SD，ddof=1），不是把 15 个 episode 假装成 15 次独立训练。AUC 是 evaluation mean 对环境 transitions 积分后除以预算区间。连续观察多个 evaluation checkpoint 会有选择偏差，所以最初预设采用 baseline 末次 validation 均值的中位数模型；以下补充独立稳定性检查，不依据最终 test 改选模型。

预先声明的噪声补测规则：仅使用初始 seeds 42/43/44 的末次 validation 均值；若某个正常配置的 sample SD / max(|mean|,1) > 0.30，追加 seeds 45、46，保持训练预算与 PPO 超参数不变，将该配置扩为 5 seeds。规则、初始统计与追加清单保存为 `results/extension_plan.json`，先完成补测再做最终 test；不使用 test 结果决定补测。只有两个补测任务时可采用已验证轨迹一致的 async transport，以利用空闲 CPU；这不改变有序 rollout、batch 或 PPO 更新。

自然失败 checkpoint 说明确定性策略还需要更充分的初始状态验证，因此在最终训练和 test 结束前补充以下选择规则：先取所有 baseline seeds（3 或 5 个）末次 validation 的中位数模型，在独立 selection seeds=11000..11019 上评估。若所有20回合回报>1000 且 episode CV≤0.30，保留该模型；否则在所有正常候选上使用相同 selection seeds，筛选满足相同稳定性条件者，按回报第10百分位最高、再按均值选模型。若无候选通过则继续调查，不进入最终 test。这些门槛是本项目工程标准，不是 Gymnasium 官方 solved 标准。

规则保存为 `results/model_selection_protocol.json`，实际候选回报与选择结果保存为 `selection_eval.csv` / `model_selection.json`。选定后才使用完全不同的 seeds=30000..30019 做20回合最终 test；另用固定 seed 30000 拍视频，不搜索好看的 seed。最终 episode SD 描述同一模型对初始状态的敏感性，训练 seed SD 描述训练随机性，两者分开报告。

3 个或 5 个训练 seeds 都是有限样本。若补测后仍波动大，应明确限制结论，不声称已经分辨出稳定超参数排名。不能仅比较三组单次最高分，也不能把不显著的排序包装为普遍最优。若出现探索策略与确定性策略明显不一致，应结合动作、视频与 value/KL 日志调查。

## 记录与断点

每个 run 保存 resolved config、UTC 时间、Git revision、tracked dirty flag、所有 `src/*.py` 的 SHA-256、Python/torch/Gymnasium/MuJoCo/NumPy/SB3 版本、wall-clock，以及 training/evaluation CSV。配置和原始 CSV 是报告数字的依据。依赖闭包来自实际安装环境，保存为 `requirements.lock.txt`；完整 AutoDL 基础软件清单不作为可移植 requirements。

首批正式 run 启动时，系统诊断和报告脚本仍有 tracked 修改，因此 metadata 如实记录 `git_dirty=true`。训练算法文件并未修改：最终 QA 同时核对当前 `src/*.py`、run 保存的 SHA-256 和所记录 Git commit 中的对应 blob，三者必须一致。此标记不能简单省略；独立 source hash 验证用于证明实际训练算法可从 Git 恢复。

Checkpoint 内含网络、normalizer 和优化器；它不包含 MuJoCo 内部 state、所有 RNG 及 vector workers 状态，所以当前实现不支持精确恢复到同一轨迹。要复现实验应从固定 seed 重新训练。CPU/CUDA 及依赖版本差异可经由连续动力学放大，不能承诺跨硬件逐位一致。

训练由 AutoDL tmux 承载，SSH 断线不会中断。GitHub 保存代码、config、compact metrics 和最终图表；checkpoint、原始日志、视频与缓存不直接入 Git。最终视频和模型在本地交付，并记录 hash 以验证它们与评估的对应关系。

最终 test seed 范围在最终候选选择之前修订为 30000..30019，避免与短程软件 QA 曾使用的 20000/20001 重叠；修订前未评估任何最终候选的 test 回报。原始 experiment_plan.json 作为历史记录保留，model_selection_protocol.json 为修订后的执行协议。

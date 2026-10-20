# 课程展示与答辩提纲

本页组织展示顺序；具体数字以自动生成的 [RESULTS.md](RESULTS.md) 和 CSV 为准。

## 建议的 8–10 分钟展示

1. **任务与结果（1 分钟）**：播放固定测试 seed 的 running video，展示 20 回合累计回报和速度，说明这是连续控制 benchmark。
2. **MDP 建模（1 分钟）**：17 维 observation、6 维 action、速度减控制惩罚；指出 observation 不等于完整 simulator state，1000 步是 truncation。
3. **PPO（2 分钟）**：按 actor → rollout → critic/GAE → old/new likelihood ratio → clipped objective → minibatch updates 的顺序解释。用 Tutorial 调节 advantage 的正负和 epsilon。
4. **实验设计（1 分钟）**：固定其余参数，分别比较 lr 与 clip；初始 3 seeds，按预声明噪声规则为四组补至 5 seeds，最终 24 runs。训练、曲线 validation、选择 validation、最终 test 使用不同随机种子。
5. **结果与失败（2 分钟）**：展示 lr/clip 曲线和 seed variability。说明 baseline 组均值领先，但跨 seed 波动大，clip=0.1 与0.2不能宣称显著差异。展示自然退化 checkpoint 与大 lr 失败；前者后来恢复，不是永久失败。
6. **工程与联系（1–2 分钟）**：展示 CPU/CUDA 实测、版本与配置追踪、离线 Tutorial。解释如何把连续策略迁移到无线功率控制，并说明尚未进行无线仿真。

## 数字的出处

| 答辩主张 | 可核查文件 |
|---|---|
| 最终独立回报及每回合长度 | `results/final_evaluation.csv`、`final_evaluation_summary.json` |
| 速度、位移、控制惩罚 | `results/final_motion.csv` |
| 每组均值、跨训练 seed SD、AUC | `results/summary.csv`、`summary.json` |
| 是否按噪声规则补测 | `results/extension_plan.json` |
| 为什么展示该模型 | `results/model_selection_protocol.json`、`model_selection.json` |
| 自然失败发生与后续恢复 | `results/failures/natural_baseline_seed44/snapshot.json`、源实验 `eval.csv` |
| 性能开销 | `results/benchmark_summary.csv`、`suite_timing.json`、`extension_timing.json` |
| 实现是否与训练源码一致 | 各实验 `metadata.json` 中 Git revision 与 source SHA-256 |
| 新环境验证与离线交互 | `results/qa/fresh_environment.json`、`tutorial.json`、`artifacts.json` |

## 容易被追问的边界

**为什么不展示最高回报 seed？** 展示模型先按 baseline 末次曲线均值取中位数，再在独立 20 回合 selection 上检查稳定性；最终 test 不用于改选模型。目标是展示代表性结果。

**3614.6 ±692.5 与 baseline 3791.6 ±1874.1 为什么不同？** 前者是同一选定模型在20个新初始状态上的 episode SD；后者是5次独立训练各自5回合 validation 均值之间的训练 seed SD。两者回答不同问题。最终 test 中最低回报985.4，1/20回合不超过selection门槛1000；完整保留这一结果，不因test结果更换模型。

**PPO clipping 为什么仍会失败？** 它裁剪 surrogate 对策略比率的奖励，不是硬约束 KL 或策略参数；多 epoch、Adam、共享采样分布和动作方差变化仍会带来大更新。用 approx-KL、clip fraction 与实际独立评估联合诊断，不能仅凭某个指标断定唯一原因。

**训练回报很好为什么确定性评估会退化？** 训练使用带噪声高斯动作、滚动窗口和持续改变的策略；评估使用冻结的均值动作与独立初始状态。自然失败 checkpoint 正说明必须保留独立评估。

**Hopper 动作更少是否更容易？** 不能这样推断。它存在平衡和健康 termination，失败会缩短轨迹；本项目只比较环境定义，没有实测 Hopper 学习难度排名。

**无线实验结果在哪里？** 本项目验证的是连续控制算法与实验方法。功率控制、资源分配是迁移设计示例，不能把 HalfCheetah 回报解释为无线吞吐量改进。

**这是否证明 PPO 收敛或超参数全局最优？** 没有。只有一个环境、固定2M预算和有限 seeds，且是单因素消融。较大的 seed SD 限制了结论的强度；本项目报告观察结果，不报告统计显著性或理论收敛。

**怎样复现？** 从 README 的新环境步骤开始，先运行6个数值/检查点测试与 smoke，再运行独立命名的训练；完整矩阵使用 code-only reproduction workspace。保留 checkpoint 的统计量用于评估。当前不支持精确断点续训；从固定 seed 重跑，跨依赖/硬件不保证逐位相同。

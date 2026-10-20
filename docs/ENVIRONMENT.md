# 环境与智能无线组网联系

默认 HalfCheetah-v5 的 observation 是 17 维：8 个位置/角度，加 9 个线速度/角速度。绝对水平位置被排除，使策略可复用于跑道不同位置；完整 simulator state 比 observation 更广，不能混为一谈。6 维动作分别控制前后腿的 thigh、shin、foot，范围 [-1,1]。

MuJoCo 默认有 9 维 qpos 和 9 维 qvel，合计 18 个运动学状态分量；给 agent 的观测去掉了绝对 x，成为 17 维。完整仿真内部还有 time、求解器等状态。由于任务对水平平移不变，去掉绝对 x 不意味着必须依靠它才能决定动作。最终位移审计读取 simulator 的 x 仅用于分析，不把它输入策略。

默认 reward = (x_after − x_before)/0.05 − 0.1 Σ aᵢ²。前一项鼓励向前速度，后一项惩罚过大的控制输入。此平方动作惩罚不是精确的机械能测量。一个 1000 步 episode 对应约 50 秒模拟时间。HalfCheetah 默认不会因为跌倒而 terminated，1000 步是 TimeLimit truncation。

| 默认环境 | Observation | Action | Episode 结束特点 |
|---|---:|---:|---|
| HalfCheetah-v5 | 17 | 6 | 无健康状态 termination；1000 步 truncation |
| Hopper-v5 | 11 | 3 | 高度、角度或其他状态不健康时可提前 termination；也有 1000 步 time limit |

Hopper 是单腿平衡与跳跃，HalfCheetah 需要协调两腿的更多关节。Hopper 维度较低，但摔倒会使有效轨迹缩短、探索面临生存约束，因此不能仅按维度断言训练更容易。HalfCheetah 能获得更长的探索轨迹，也可能学到非自然但高回报的动作；最终需要视频观察。这里没有完成 Hopper 训练，难度差异是机制分析，不是本项目的实测排名。

Hopper 默认健康高度下界 0.7、躯干角度范围 (-0.2,0.2)，并有健康奖励；HalfCheetah 没有此健康奖励。两者的模拟步长也不同：0.008 秒对 0.05 秒，所以 1000 步的物理时间不同，奖励不能直接横比。`scripts/inspect_environments.py` 从实际安装的两个环境记录维度、dt 和默认参数，输出 `results/infrastructure/environments.json`，不需要训练 Hopper。

可靠定义与参数见 [Gymnasium HalfCheetah 文档](https://gymnasium.farama.org/environments/mujoco/half_cheetah/) 和 [Gymnasium Hopper 文档](https://gymnasium.farama.org/environments/mujoco/hopper/)。实际安装版本会由 diagnostics 和实验 metadata 记录。

## 为什么与《智能无线组网》相关

HalfCheetah 是连续控制算法 benchmark，本项目没有把它当作无线网络实验。可迁移的是 MDP 建模、连续策略、受约束动作、稳定策略更新及多 seed 评估方法。

例如功率控制：状态可含信道增益、干扰和队列；动作是各链路发射功率；reward 可组合吞吐量、能耗和时延。资源分配可以输出连续带宽份额，连续网络参数优化可以调整阈值或波束参数。这些只是潜在应用设计，不是已经验证的网络性能结论。

迁移时还必须重新处理部分可观测性、非平稳多用户竞争、通信开销、硬 QoS/功率约束、离线到在线分布偏移。HalfCheetah 的 action clipping 不能直接替代网络的可行域投影或安全约束机制。

# PPO 实现与答辩笔记

代码阅读顺序：`ppo.py` → `train.py` → `evaluate.py`。SB3 只负责参考实验，自研训练器自己的网络、轨迹采样、GAE、目标函数和优化循环都可以直接审阅。

## 决策与网络

Actor 把 17 维观测映射到 6 维高斯均值；另有 6 个可学习 log standard deviation。Actor 与 Critic 各用两层 64 单元 Tanh 网络。Critic 输出标量 V(s)。训练采样高斯动作，评估使用均值。环境动作限制在 [-1,1]；日志概率针对采样的原始高斯动作计算，环境接收其裁剪值。这和在裁剪值上重新计算高斯概率不同，后者会给 PPO 比率造成错误。饱和动作仍是这一策略参数化的局限。

Observation 使用运行均值和方差标准化，裁剪至 ±10；reward 除以折扣回报的运行标准差以改善 critic 的数值尺度。训练指标始终记录原始环境回报。评估冻结 observation 统计，不做 reward 标准化。Checkpoint 保存网络、优化器与统计量；跨硬件/依赖版本不保证逐位一致。

## 从轨迹到更新

每个 rollout 包含 `n_steps × n_envs` 个 transition。时间差分残差为 δₜ = rₜ + γV(sₜ₊₁) − V(sₜ)。GAE 反向递推 Aₜ = δₜ + γλAₜ₊₁，target return = Aₜ + V(sₜ)。λ 控制 bias/variance，γ 控制未来奖励权重。

真正 terminated 时 next value 置零。达到 1000 步的 truncated 仍使用最终观测的 V 做 bootstrap；递推在两种结束边界都停止，避免把 reset 后的新 episode 混入上一条轨迹。数学测试专门覆盖这个区别。

令 ρₜ = exp(log πθ(aₜ|sₜ) − log πold(aₜ|sₜ))，PPO 最大化 min(ρₜAₜ, clip(ρₜ,1−ε,1+ε)Aₜ)。正优势限制过大的概率增幅，负优势限制过大的概率减幅。它不直接硬限制 KL，也不保证每次更新都改善回报。

每个 minibatch 对优势去均值、除标准差。损失为负的 clipped surrogate 加 `value_coef × 0.5 × MSE(V,return)`，再减 `entropy_coef × entropy`。默认 entropy coefficient 为零，但高斯采样仍提供探索。Adam epsilon 为 1e-5，梯度范数裁剪为 0.5；多轮 epochs 重用同一个 rollout。不会用更新后的策略重新采样旧轨迹动作。

## 如何解释消融

Learning rate 控制参数更新幅度；过小可能在给定步数内尚未收敛，过大可能破坏已学策略。Clip range 控制 surrogate 对概率比变化的敏感度；更小不等价于一定更好、更大也不等价于更快学习。结论必须限定在固定 rollout、batch size、epochs、normalization、训练预算和 seeds 的设计内。

同时看全程曲线、末次 held-out evaluation、跨 seed 方差和失败轨迹。训练曲线包含探索，评估使用确定性均值动作；两者差异不能简单解释为 bug。跨 episode 的标准差描述同一策略对初始状态的敏感性，跨训练 seed 的标准差描述训练随机性，两者不可混用。

## 答辩常见问题

- 为什么 on-policy 仍能重用数据？PPO 在有限 epochs 内通过 old-policy 概率比和 clipping 控制偏离；它不是通用 off-policy replay。
- 为什么要 critic？它提供 baseline 和 bootstrap，以降低策略梯度估计方差。
- 为什么不以训练中的最高回报选模型？单个探索 episode 噪声很大；使用独立 evaluation、多 seeds 和预先声明的选择规则。
- 为什么 CPU 可能更快？小 MLP 加上 CPU MuJoCo 的每步交互容易被 GPU kernel launch 和数据传输开销支配，必须实测。
- Checkpoint 能否精确恢复训练？当前保存可复用策略和优化器，但没有保存 MuJoCo 内部状态及所有 RNG 状态，不支持中断后逐位相同的续训；复现从初始 seed 重新运行。

PPO 原始方法来源：[Schulman et al., 2017](https://arxiv.org/abs/1707.06347)。实现中 normalization、网络及日志选择以上述代码为准，不能把 SB3 内部实现细节直接当作自研算法。

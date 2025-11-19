GRPO 或 PPO 并不直接优化 $J(\theta)$，而是优化一个 surrogate objective：

$$
\mathcal{L}_{\text{GRPO}}(\theta) = \mathbb{E}_\tau \left[ \min \left( r_\theta(\tau) A(\tau),\ \text{clip}(r_\theta(\tau), 1 - \epsilon, 1 + \epsilon) A(\tau) \right) \right]
$$





## ProRL——延时强化学习（Prolonged Reinforcement Learning）**

目标是解决传统强化学习在长期训练中面临的两大挑战：

1. **熵塌缩（Entropy Collapse）**：输出分布过早集中，限制探索；
2. **训练不稳定性（Instability）**：策略偏移过大导致训练震荡。


 为解决问题有以下三个策略：
 ProRL 为解决 **熵塌缩（entropy collapse）** 引入了以下多项机制：

1. **提高采样温度（high rollout temperature）**：增加初始输出熵，鼓励早期探索。
2. **解耦截断边界（decoupled clip bounds）**：设置更大的上界 $\epsilon_{\text{high}}$，提升低概率 token 的选中率，保持输出多样性。
3. **KL 散度正则项（KL divergence penalty）**：约束当前策略 $\pi_\theta$ 不偏离参考策略 $\pi_{\text{ref}}$，防止策略快速收敛到单一解。
4. **动态采样（dynamic sampling）**：过滤掉始终正确或错误的 prompt，集中训练具有学习价值的中等难度样本。
5. **周期性重置参考策略（reference policy reset）**：定期将 $\pi_{\text{ref}}$ 更新为当前策略快照，防止 KL 项主导训练损失，促进持续优化。

---

### **2.1 强化学习基础：GRPO**

ProRL 以 **GRPO（Group Relative Policy Optimization）** 作为基础强化学习算法。相较于 PPO，GRPO **不依赖 value function（值函数）**，而是利用 group-based reward 均值和标准差构造优势值（advantage）：

* **目标函数**为：

  $$
  \mathcal{L}_{\text{GRPO}}(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \min \left( r_\theta(\tau) A(\tau),\ \text{clip}(r_\theta(\tau), 1 - \epsilon,\ 1 + \epsilon) A(\tau) \right) \right]
  $$

  其中：

  $$
  r_\theta(\tau) = \frac{\pi_\theta(\tau)}{\pi_{\text{old}}(\tau)} \quad \text{是概率比值}
  $$

* **优势值估计方式**为：

  $$
  A(\tau) = \frac{R_\tau - \text{mean}(\{R_i\}_{i \in G(\tau)})}{\text{std}(\{R_i\}_{i \in G(\tau)})}
  $$

  不需要 critic 网络，仅依赖 group reward。

---

### **2.2 熵塌缩的挑战与应对**

**问题描述**：
长时间强化学习容易出现熵塌缩（entropy collapse），即模型输出越来越确定，导致熵值下降，探索能力急剧退化。这会造成：

* 模型只输出高概率路径；
* 训练信号缺失，导致停滞；
* 推理任务中多样解被抑制。

**常规手段**：
提升采样温度（temperature）可暂缓熵塌缩，但无法从根本上解决。

---

### **2.3 DAPO 机制：解耦截断 + 动态采样**

ProRL 借鉴了 DAPO \[4] 中的两项策略：

#### ✅ 解耦截断（Decoupled Clipping）：

将原本对称的 PPO clip 上下界，分别设定为：

$$
\text{clip}(r_\theta(\tau),\ 1 - \epsilon_{\text{low}},\ 1 + \epsilon_{\text{high}})
$$

设置较大的 $\epsilon_{\text{high}}$ 使得罕见 token 更容易提升概率，从而扩大探索范围，延缓熵下降。

#### ✅ 动态采样（Dynamic Sampling）：

在训练过程中，过滤掉准确率为 1 或 0 的 prompt，因为这些数据无法提供有效的训练信号。聚焦于“中等难度样本”可增加输出多样性与学习梯度。

---

### **2.3.1 KL 正则化与参考策略重置**

#### ✅ **KL 正则项（KL Divergence Penalty）**

引入 KL 散度惩罚项以稳定训练：

$$
\mathcal{L}_{\text{KL-RL}}(\theta) = \mathcal{L}_{\text{GRPO}}(\theta) - \beta \cdot D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}})
$$

* 限制当前策略与参考策略之间的偏移；
* 保持熵水平，抑制过拟合与训练震荡。

#### 📌 与现有观点的不同：

一些研究建议移除 KL 惩罚项，认为其阻碍了策略自由探索。但 ProRL 以一个已具备推理能力的模型为起点（DeepSeek-R1-Distill-Qwen-1.5B），认为此时仍然 **需要 KL 项保持稳定性与熵值**。

#### ✅ **参考策略重置（Reference Policy Reset）**

随着训练进行，KL 项可能主导损失函数，导致策略无法有效更新。为此，ProRL提出：

* 定期将参考策略 $\pi_{\text{ref}}$ **重置为当前策略 $\pi_\theta$** 的快照；
* 同时重置优化器状态。

这样既保留了 KL 正则的益处，又避免因 KL 过大而造成学习停滞，实现 **持续、稳定、长周期的强化训练**。

---

### ✅ 总结：ProRL 的创新点

| 机制              | 功能作用                        |
| --------------- | --------------------------- |
| GRPO 策略优化       | 简化结构，依赖 group reward 代替值函数  |
| 高温采样            | 提高初始熵值，增强早期探索能力             |
| 解耦截断（clip-high） | 鼓励低概率 token 被选中，扩展探索边界      |
| 动态采样            | 关注“学习价值高”的 prompt，维持训练信号多样性 |
| KL 正则           | 限制策略偏移，稳定训练过程               |
| 参考策略重置          | 防止 KL 项过度抑制策略更新，鼓励长期优化      |

---

如需，我可以将这些方法在代码层展示如何实现（如 RL loss 函数、clip 实现、KL 重置逻辑等），是否需要？

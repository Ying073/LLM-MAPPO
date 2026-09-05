"""
networks.py —— Actor / Critic 网络结构

论文对应：
- 公式 23–24：PPO Actor 裁剪目标 + Actor 参数更新 (clip ε)
- 公式 25–26：Critic 均方误差损失 + Critic 参数更新

设计思想：
    集中训练、分散执行 (CTDE):
        - Actor  π(a_n | O_n(t))   只用 *自己* UAV 的局部观测 → 公式 18
        - Critic V(s(t))            用 *全局* state (所有 UAV 观测拼起来) → 公式 25

    网络形状 (M2 取最简版):
        Actor  : MLP(in=60, 64, 64, out=6), Tanh 激活, 输出动作 logits
        Critic : MLP(in=60*N_UAV, 128, 64, out=1), Tanh 激活, 输出标量价值
"""

import torch
import torch.nn as nn


def init_weights(layer: nn.Module):
    """Xavier-uniform 初始化 hidden 层，输出层用小方差初始化."""
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)


class Actor(nn.Module):
    """每架 UAV 一个 Actor (各自分散执行)；这里共享参数，batch 维对应不同 UAV."""

    def __init__(self, obs_dim: int = 60, act_dim: int = 6, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
        self.apply(init_weights)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """输入 obs: (B, obs_dim) → logits: (B, act_dim)."""
        return self.net(obs)

    def get_action(self, obs: torch.Tensor):
        """从 Categorical 分布采样一个动作 + 给 log_prob 和 entropy.

        返回:
            action : (B,) int64
            logp   : (B,) float  log π(a|obs)
            entropy: (B,) float  H[π(·|obs)]
        """
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()


class Critic(nn.Module):
    """集中式 Critic：输入 = 全局 state = 所有 UAV 局部观测的拼接."""

    def __init__(self, global_state_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
            nn.Linear(hidden // 2, 1),
        )
        self.apply(init_weights)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """输入 (B, global_state_dim) → 价值 (B, 1)."""
        return self.net(global_state).squeeze(-1)

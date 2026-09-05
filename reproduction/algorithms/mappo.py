"""
mappo.py —— MAPPO 训练主逻辑 (per-UAV PPO with shared critic)

论文对应：
- 公式 23：PPO 裁剪目标  L_clip(θ) = E[min(r·A, clip(r,1-ε,1+ε)·A)]
- 公式 24：Actor  θ ← θ + α · ∇L_clip
- 公式 25：Critic MSE  L_vf = E[(V(s) − R̃)^2]
- 公式 26：Critic  φ ← φ − α · ∇L_vf

实现选择 (M2 取 MAPPO 论文里最经典的版本) :
    - 每架 UAV 有自己的旧策略 π_old (和现策略共享参数, 但 log_prob 用 .detach())
    - 一次 rollout 收集 T=500 步, 然后 K=4 epoch + mini-batch SGD 更新
    - clip ε = 0.2, γ = 0.99, λ = 0.95, lr = 3e-4
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .networks import Actor, Critic


class MAPPO:
    """MAPPO 训练器."""

    def __init__(
        self,
        obs_dim: int,
        global_dim: int,
        act_dim: int = 6,
        clip_eps: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        update_epochs: int = 4,
        minibatch_size: int = 64,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.actor = Actor(obs_dim=obs_dim, act_dim=act_dim).to(self.device)
        self.critic = Critic(global_state_dim=global_dim).to(self.device)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.clip_eps = clip_eps
        self.gamma = gamma
        self.lam = gae_lambda
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size

    # ---------------------------------------------------------------
    # 选动作 (部署 / 收集时用)
    # ---------------------------------------------------------------
    @torch.no_grad()
    def select_actions(self, obs_list):
        """obs_list: list of (obs_dim,) np.ndarray × N_UAV → actions: list of int."""
        obs = torch.from_numpy(np.stack(obs_list)).to(self.device)  # (N_UAV, obs_dim)
        action, logp, _ = self.actor.get_action(obs)
        return action.cpu().numpy(), logp.cpu().numpy()

    @torch.no_grad()
    def get_value(self, global_state: np.ndarray) -> float:
        gs = torch.from_numpy(global_state).unsqueeze(0).to(self.device)  # (1, global_dim)
        v = self.critic(gs).item()
        return v

    # ---------------------------------------------------------------
    # PPO update (公式 23–26)
    # ---------------------------------------------------------------
    def update(self, buffer, last_value: float):
        """buffer 满后调用一次更新。返回本轮平均 actor / critic loss 给日志用."""
        # 1. 算 GAE 优势 + returns
        adv, ret = buffer.compute_advantages(last_value)

        # 2. K 次 epoch, 每次随机 minibatch
        actor_losses, critic_losses = [], []
        for _epoch in range(self.update_epochs):
            for mb in buffer.get_minibatches(adv, ret, self.minibatch_size):
                # --- 一次性把 minibatch 搬到 device（GPU 关键，否则 device mismatch） ---
                device = self.device
                obs  = mb["obs"].to(device)
                gs   = mb["global_s"].to(device)
                act  = mb["actions"].to(device)
                old_logp = mb["old_logp"].to(device)
                adv_ = mb["advantages"].to(device)
                ret_ = mb["returns"].to(device)

                # ---- Actor: 公式 23 裁剪目标 ----
                logits = self.actor(obs)                              # (B, act_dim)
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(act)                          # (B,)
                ratio = torch.exp(new_logp - old_logp)                 # (B,)
                surr1 = ratio * adv_
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_
                actor_loss = -torch.min(surr1, surr2).mean()
                entropy = dist.entropy().mean()                        # bonus

                self.actor_opt.zero_grad()
                (actor_loss - 0.01 * entropy).backward()               # 鼓励探索
                self.actor_opt.step()

                # ---- Critic: 公式 25 MSE ----
                v_pred = self.critic(gs)                               # (B,)
                critic_loss = ((v_pred - ret_) ** 2).mean()

                self.critic_opt.zero_grad()
                critic_loss.backward()
                self.critic_opt.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())

        return float(np.mean(actor_losses)), float(np.mean(critic_losses))

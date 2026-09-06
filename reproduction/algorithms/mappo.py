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
    - clip ε = 0.2, γ = 0.95, λ = 0.95, lr = 2e-4
"""

import os

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
        n_agents: int = 7,
        act_dim: int = 6,
        clip_eps: float = 0.2,
        gamma: float = 0.95,
        gae_lambda: float = 0.95,
        actor_lr: float = 2e-4,
        critic_lr: float = 2e-4,
        update_epochs: int = 4,
        minibatch_size: int = 64,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.global_dim = global_dim
        self.act_dim = act_dim
        self.n_agents = n_agents
        self.actors = nn.ModuleList([
            Actor(obs_dim=obs_dim, act_dim=act_dim) for _ in range(n_agents)
        ]).to(self.device)
        self.critic = Critic(global_state_dim=global_dim).to(self.device)
        self.actor_opts = [optim.Adam(actor.parameters(), lr=actor_lr) for actor in self.actors]
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
    def select_actions(self, obs_list, action_masks=None, deterministic: bool = False):
        """obs_list: list of (obs_dim,) np.ndarray × N_UAV → actions: list of int."""
        obs = torch.from_numpy(np.stack(obs_list)).to(self.device)
        masks = None if action_masks is None else torch.as_tensor(action_masks, device=self.device)
        actions, logps = [], []
        for n, actor in enumerate(self.actors):
            mask_n = None if masks is None else masks[n:n + 1]
            if deterministic:
                logits = actor(obs[n:n + 1])
                if mask_n is not None:
                    logits = logits.masked_fill(~mask_n.bool(), torch.finfo(logits.dtype).min)
                action = logits.argmax(dim=-1)
                logp = torch.distributions.Categorical(logits=logits).log_prob(action)
            else:
                action, logp, _ = actor.get_action(obs[n:n + 1], mask_n)
            actions.append(int(action.item()))
            logps.append(float(logp.item()))
        return np.asarray(actions, dtype=np.int64), np.asarray(logps, dtype=np.float32)

    @torch.no_grad()
    def select_actions_batched(self, obs_batch: np.ndarray, action_masks=None,
                               deterministic: bool = False):
        """batched version: obs_batch (n_envs, N_UAV, obs_dim) → (actions (n_envs, N_UAV), logp (n_envs, N_UAV)).

        Reshape 一次性把 n 个 env × N_UAV 喂给 Actor, 然后 reshape 回."""
        n_envs, n_agents = obs_batch.shape[:2]
        if n_agents != self.n_agents:
            raise ValueError(f"obs has {n_agents} agents, expected {self.n_agents}")
        obs = torch.from_numpy(obs_batch).to(self.device)
        masks = None if action_masks is None else torch.as_tensor(action_masks, device=self.device)
        actions = np.empty((n_envs, n_agents), dtype=np.int64)
        logps = np.empty((n_envs, n_agents), dtype=np.float32)
        for agent, actor in enumerate(self.actors):
            mask_n = None if masks is None else masks[:, agent]
            if deterministic:
                logits = actor(obs[:, agent])
                if mask_n is not None:
                    logits = logits.masked_fill(~mask_n.bool(), torch.finfo(logits.dtype).min)
                action = logits.argmax(dim=-1)
                logp = torch.distributions.Categorical(logits=logits).log_prob(action)
            else:
                action, logp, _ = actor.get_action(obs[:, agent], mask_n)
            actions[:, agent] = action.cpu().numpy()
            logps[:, agent] = logp.cpu().numpy()
        return actions, logps

    @torch.no_grad()
    def get_value(self, global_state: np.ndarray) -> float:
        gs = torch.from_numpy(global_state).unsqueeze(0).to(self.device)  # (1, global_dim)
        v = self.critic(gs).item()
        return v

    @torch.no_grad()
    def get_value_batched(self, global_state_batch: np.ndarray):
        """batched version: global_state_batch (n_envs, global_dim) → (n_envs,) value."""
        gs = torch.from_numpy(global_state_batch).to(self.device)  # (n_envs, global_dim)
        v = self.critic(gs)
        return v.cpu().numpy()

    # ---------------------------------------------------------------
    # PPO update (公式 23–26)
    # ---------------------------------------------------------------
    def update(self, buffer, last_value: float):
        """buffer 满后调用一次更新。返回本轮平均 actor / critic loss 给日志用."""
        # 1. 算 GAE 优势 + returns
        adv, ret = buffer.compute_advantages(last_value)

        # 2. K 次 epoch。Critic 每个 joint transition 更新一次；每个 Actor
        # 只用自己的 observation/action/mask，保持论文的独立参数 theta_n。
        actor_losses, critic_losses = [], []
        T, E, N = buffer.ptr, buffer.E, buffer.N
        if T == 0:
            raise ValueError("cannot update from an empty rollout buffer")

        if E == 1:
            obs = buffer.obs[:T][:, None]
            actions = buffer.actions[:T][:, None]
            old_logp = buffer.logp[:T][:, None]
            masks = buffer.action_mask[:T][:, None]
            gs = buffer.global_s[:T]
            adv_te = adv[:, None]
            ret_te = ret[:, None]
        else:
            obs = buffer.obs[:T]
            actions = buffer.actions[:T]
            old_logp = buffer.logp[:T]
            masks = buffer.action_mask[:T]
            gs = buffer.global_s[:T].reshape(T, E, -1)
            adv_te, ret_te = adv, ret

        flat_adv = adv_te.reshape(-1).astype(np.float32)
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
        flat_ret = ret_te.reshape(-1).astype(np.float32)
        flat_gs = gs.reshape(T * E, -1).astype(np.float32)

        for _epoch in range(self.update_epochs):
            order = np.random.permutation(T * E)
            for start in range(0, T * E, self.minibatch_size):
                idx = order[start:start + self.minibatch_size]
                idx_t = torch.as_tensor(idx, device=self.device)
                gs_t = torch.from_numpy(flat_gs[idx]).to(self.device)
                ret_t = torch.from_numpy(flat_ret[idx]).to(self.device)
                v_pred = self.critic(gs_t)
                critic_loss = ((v_pred - ret_t) ** 2).mean()
                self.critic_opt.zero_grad()
                critic_loss.backward()
                self.critic_opt.step()
                critic_losses.append(float(critic_loss.item()))

                for agent, (actor, opt) in enumerate(zip(self.actors, self.actor_opts)):
                    obs_flat = obs[:, :, agent].reshape(T * E, -1).astype(np.float32)
                    act_flat = actions[:, :, agent].reshape(-1)
                    logp_flat = old_logp[:, :, agent].reshape(-1)
                    mask_flat = masks[:, :, agent].reshape(T * E, -1)
                    obs_t = torch.from_numpy(obs_flat[idx]).to(self.device)
                    act_t = torch.from_numpy(act_flat[idx]).to(self.device)
                    old_t = torch.from_numpy(logp_flat[idx]).to(self.device)
                    adv_t = torch.from_numpy(flat_adv[idx]).to(self.device)
                    mask_t = torch.from_numpy(mask_flat[idx]).to(self.device)
                    logits = actor(obs_t).masked_fill(~mask_t.bool(), torch.finfo(torch.float32).min)
                    dist = torch.distributions.Categorical(logits=logits)
                    ratio = torch.exp(dist.log_prob(act_t) - old_t)
                    surr1 = ratio * adv_t
                    surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t
                    actor_loss = -torch.min(surr1, surr2).mean()
                    entropy = dist.entropy().mean()
                    opt.zero_grad()
                    (actor_loss - 0.01 * entropy).backward()
                    opt.step()
                    actor_losses.append(float(actor_loss.item()))

        return float(np.mean(actor_losses)), float(np.mean(critic_losses))

    def _checkpoint_payload(self):
        return {
            "config": {
                "obs_dim": self.obs_dim,
                "global_dim": self.global_dim,
                "n_agents": self.n_agents,
                "act_dim": self.act_dim,
                "clip_eps": self.clip_eps,
                "gamma": self.gamma,
                "gae_lambda": self.lam,
                "update_epochs": self.update_epochs,
                "minibatch_size": self.minibatch_size,
            },
            "actors": [actor.state_dict() for actor in self.actors],
            "critic": self.critic.state_dict(),
        }

    def save_checkpoint(self, path):
        torch.save(self._checkpoint_payload(), path)

    def save_training_checkpoint(self, path, progress):
        """Atomically save policy, optimizers, and caller-owned training progress."""
        payload = self._checkpoint_payload()
        payload.update({
            "actor_optimizers": [opt.state_dict() for opt in self.actor_opts],
            "critic_optimizer": self.critic_opt.state_dict(),
            "training_progress": progress,
        })
        tmp_path = f"{path}.tmp"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        payload = torch.load(path, map_location=device, weights_only=False)
        config = dict(payload["config"])
        algo = cls(device=device, **config)
        for actor, state in zip(algo.actors, payload["actors"]):
            actor.load_state_dict(state)
        algo.critic.load_state_dict(payload["critic"])
        return algo

    @classmethod
    def from_training_checkpoint(cls, path, device="cpu"):
        payload = torch.load(path, map_location=device, weights_only=False)
        config = dict(payload["config"])
        algo = cls(device=device, **config)
        for actor, state in zip(algo.actors, payload["actors"]):
            actor.load_state_dict(state)
        algo.critic.load_state_dict(payload["critic"])
        for opt, state in zip(algo.actor_opts, payload["actor_optimizers"]):
            opt.load_state_dict(state)
        algo.critic_opt.load_state_dict(payload["critic_optimizer"])
        return algo, payload["training_progress"]

"""
buffer.py —— Rollout Buffer：存一段时间的轨迹，给 PPO 做 off-policy update。

论文对应：
- 公式 23：PPO 裁剪目标中的 ratio θ = π(a|s) / π_old(a|s)
            → 需要存 "new" 和 "old" 两套 log_prob

实现选择：
    用 on-policy 经典做法：每次收集 N 步 (rollout_len) 的数据，然后做 K 次
    mini-batch SGD（在 buffer 内部随机 shuffle）。

    字段：
        obs     : (T, N_UAV, obs_dim)  局部观测
        global_s: (T, global_dim)      全局 state
        actions : (T, N_UAV)          动作
        logp    : (T, N_UAV)          采样时刻的 log π(a|obs)
        reward  : (T, N_UAV)          每架 UAV 的奖励
        done    : (T,)                是否结束
        value   : (T,)                收集时的 V(s) 估计
"""

import numpy as np
import torch


class RolloutBuffer:
    """存一段时间的轨迹 + GAE / 优势计算."""

    def __init__(self, rollout_len: int, n_uav: int, obs_dim: int, global_dim: int,
                 gamma: float = 0.99, gae_lambda: float = 0.95):
        self.T = rollout_len
        self.N = n_uav
        self.gamma = gamma
        self.lam = gae_lambda

        self.obs = np.zeros((self.T, self.N, obs_dim), dtype=np.float32)
        self.global_s = np.zeros((self.T, global_dim), dtype=np.float32)
        self.actions = np.zeros((self.T, self.N), dtype=np.int64)
        self.logp = np.zeros((self.T, self.N), dtype=np.float32)
        self.reward = np.zeros((self.T, self.N), dtype=np.float32)
        self.done = np.zeros((self.T,), dtype=np.float32)
        self.value = np.zeros((self.T,), dtype=np.float32)

        self.ptr = 0

    def store(self, obs, global_s, actions, logp, reward, done, value):
        """存一步。obs: (N, obs_dim) list / array; reward: (N,) array; value: scalar."""
        self.obs[self.ptr] = np.asarray(obs, dtype=np.float32)
        self.global_s[self.ptr] = np.asarray(global_s, dtype=np.float32)
        self.actions[self.ptr] = np.asarray(actions, dtype=np.int64)
        self.logp[self.ptr] = np.asarray(logp, dtype=np.float32)
        self.reward[self.ptr] = np.asarray(reward, dtype=np.float32)
        self.done[self.ptr] = float(done)
        self.value[self.ptr] = float(value)
        self.ptr += 1

    def reset(self):
        self.ptr = 0

    def is_full(self) -> bool:
        return self.ptr >= self.T

    # ---------------------------------------------------------------
    # GAE (Generalized Advantage Estimation)
    # ---------------------------------------------------------------
    def compute_advantages(self, last_value: float):
        """收集完毕后调用：算每步的 GAE 优势 A_t 和 return-to-go R̃_t.

        A_t = δ_t + γλ δ_{t+1} + (γλ)^2 δ_{t+2} + ...
        δ_t = r_t + γ V(s_{t+1})(1-done) - V(s_t)

        这里简化：所有 UAV 共享一个优势 (因为我们用 shared_reward 给 Critic)，
        把每步 per-agent 的 reward 平均成一个 scalar，再算 GAE。
        """
        # 把 reward 沿 UAV 求平均, 得到 (T,) 的 collective reward
        rewards = self.reward.mean(axis=1)
        values = self.value.copy()
        # 在尾部加 last_value 作为 bootstrap
        next_value = last_value

        advantages = np.zeros(self.T, dtype=np.float32)
        last_adv = 0.0
        for t in reversed(range(self.T)):
            mask = 1.0 - self.done[t]
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            last_adv = delta + self.gamma * self.lam * mask * last_adv
            advantages[t] = last_adv
            next_value = values[t]
        returns = advantages + values
        return advantages, returns

    # ---------------------------------------------------------------
    # Mini-batch sampler (PPO 用 K 次 epoch 时取 minibatch)
    # ---------------------------------------------------------------
    def get_minibatches(self, advantages, returns, batch_size: int):
        """把 T 步展平成 T*N 个 (因为每个 UAV 是独立样本)，再 shuffle.

        Yields: dict of tensors
        """
        # 每个 UAV 每个时间步作为一个样本
        T, N = self.reward.shape[0], self.reward.shape[1]
        # 展平 UAV 维
        obs_flat = self.obs.reshape(T * N, -1)
        # 重复 global_s 沿 UAV 维
        global_flat = np.repeat(self.global_s, N, axis=0)
        actions_flat = self.actions.reshape(-1)
        logp_flat = self.logp.reshape(-1)
        adv_flat = np.repeat(advantages, N)        # 每步的优势复制 N 份
        ret_flat = np.repeat(returns, N)

        n = obs_flat.shape[0]
        # 标准化 advantages
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        idx = np.arange(n)
        np.random.shuffle(idx)
        for start in range(0, n, batch_size):
            mb = idx[start:start + batch_size]
            yield {
                "obs": torch.from_numpy(obs_flat[mb]),
                "global_s": torch.from_numpy(global_flat[mb]),
                "actions": torch.from_numpy(actions_flat[mb]),
                "old_logp": torch.from_numpy(logp_flat[mb]),
                "advantages": torch.from_numpy(adv_flat[mb].astype(np.float32)),
                "returns": torch.from_numpy(ret_flat[mb].astype(np.float32)),
            }

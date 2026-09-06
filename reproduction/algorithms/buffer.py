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
    """存一段时间的轨迹 + GAE / 优势计算.

    n_envs=1: 单 env (兼容 train.py), 形状 (T, ...) 与 v1 相同
    n_envs>1: 多 env 并行, 形状前加 n_envs 维 (T, n_envs, N_UAV, ...) 等
    """

    def __init__(self, rollout_len: int, n_uav: int, obs_dim: int, global_dim: int,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 n_envs: int = 1):
        self.T = rollout_len
        self.N = n_uav
        self.E = n_envs
        self.gamma = gamma
        self.lam = gae_lambda

        if n_envs == 1:
            # 兼容 train.py 的形状
            self.obs = np.zeros((self.T, self.N, obs_dim), dtype=np.float32)
            self.actions = np.zeros((self.T, self.N), dtype=np.int64)
            self.logp = np.zeros((self.T, self.N), dtype=np.float32)
            self.reward = np.zeros((self.T, self.N), dtype=np.float32)
            self.done = np.zeros((self.T,), dtype=np.float32)
            self.value = np.zeros((self.T,), dtype=np.float32)
        else:
            # batched: 在 obs/actions/logp/reward 上加 n_envs 维
            # done 和 value 也加 n_envs 维 (per-env bootstrap)
            self.obs = np.zeros((self.T, self.E, self.N, obs_dim), dtype=np.float32)
            self.actions = np.zeros((self.T, self.E, self.N), dtype=np.int64)
            self.logp = np.zeros((self.T, self.E, self.N), dtype=np.float32)
            self.reward = np.zeros((self.T, self.E, self.N), dtype=np.float32)
            self.done = np.zeros((self.T, self.E), dtype=np.float32)
            self.value = np.zeros((self.T, self.E), dtype=np.float32)
        # global_s 是 per-step-per-env 共有 (Critic 看全局 state, 但 train.py 是 1 env 共享 global_dim = obs_dim * N)
        # batched 时每 env 各自有 global_dim 长度
        self.global_s = np.zeros((self.T, global_dim * n_envs), dtype=np.float32)

        self.ptr = 0

    def store(self, obs, global_s, actions, logp, reward, done, value):
        """存一步 (n_envs=1 模式).
        obs: (N, obs_dim) list / array; reward: (N,) array; value: scalar; done: bool/scalar.
        """
        self.obs[self.ptr] = np.asarray(obs, dtype=np.float32)
        self.global_s[self.ptr] = np.asarray(global_s, dtype=np.float32)
        self.actions[self.ptr] = np.asarray(actions, dtype=np.int64)
        self.logp[self.ptr] = np.asarray(logp, dtype=np.float32)
        self.reward[self.ptr] = np.asarray(reward, dtype=np.float32)
        self.done[self.ptr] = float(done)
        self.value[self.ptr] = float(value)
        self.ptr += 1

    def store_batch(self, obs, global_s, actions, logp, reward, done, value):
        """存 batched 一步 (n_envs>1).
        obs: (n_envs, N_UAV, obs_dim) ndarray
        global_s: (n_envs, global_dim)
        actions, logp, reward: (n_envs, N_UAV)
        done, value: (n_envs,) bool/scalar
        """
        self.obs[self.ptr] = np.asarray(obs, dtype=np.float32)
        # global_s 沿 env 维展平拼接成一个长 vector (per-step)
        self.global_s[self.ptr] = np.asarray(global_s, dtype=np.float32).reshape(-1)
        self.actions[self.ptr] = np.asarray(actions, dtype=np.int64)
        self.logp[self.ptr] = np.asarray(logp, dtype=np.float32)
        self.reward[self.ptr] = np.asarray(reward, dtype=np.float32)
        self.done[self.ptr] = np.asarray(done, dtype=np.float32)
        self.value[self.ptr] = np.asarray(value, dtype=np.float32)
        self.ptr += 1

    def reset(self):
        self.ptr = 0

    def is_full(self) -> bool:
        return self.ptr >= self.T

    # ---------------------------------------------------------------
    # GAE (Generalized Advantage Estimation)
    # ---------------------------------------------------------------
    def compute_advantages(self, last_value):
        """收集完毕后调用：算每步的 GAE 优势 A_t 和 return-to-go R̃_t.

        last_value: scalar (n_envs=1) 或 np.ndarray (n_envs,) shape

        A_t = δ_t + γλ δ_{t+1} + (γλ)^2 δ_{t+2} + ...
        δ_t = r_t + γ V(s_{t+1})(1-done) - V(s_t)

        所有 UAV 共享一个优势 (因为我们用 shared_reward 给 Critic)，
        把每步 per-agent 的 reward 平均成一个 scalar，再算 GAE。
        """
        if self.E == 1:
            # 单 env 路径, 与 v1 兼容
            rewards = self.reward.mean(axis=1)                       # (T,)
            values = self.value.copy()                                # (T,)
            next_value = float(last_value)
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
        else:
            # batched 路径: 每 env 独立 GAE, 然后展平
            # rewards (T, E, N) → per-env mean (T, E)
            rewards = self.reward.mean(axis=2)                       # (T, E)
            values = self.value.copy()                                # (T, E)
            next_value = np.asarray(last_value, dtype=np.float32)    # (E,)
            advantages = np.zeros((self.T, self.E), dtype=np.float32)
            for t in reversed(range(self.T)):
                mask = 1.0 - self.done[t]                            # (E,)
                delta = rewards[t] + self.gamma * next_value * mask - values[t]  # (E,)
                # 沿 time 累积优势: 但同时间不同 env 是并行的, 各自计算 GAE
                # 这里我们其实是要对每个 env 独立倒推, 所以这里 last_adv 也应是 (E,) 向量
                # 这是首次实现, 简化版: 用 ndarray
            # 简化: 因为 E 通常很小 (<= 64), 直接用循环
            advantages = np.zeros((self.T, self.E), dtype=np.float32)
            last_adv = np.zeros(self.E, dtype=np.float32)
            for t in reversed(range(self.T)):
                mask = 1.0 - self.done[t]                            # (E,)
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
        """把 T 步展平成 T*E*N 个, 再 shuffle.
        兼容 n_envs=1 (shape (T,)) 和 n_envs>1 (shape (T, E)).

        Yields: dict of tensors.
        """
        if self.E == 1:
            T, N = self.reward.shape[0], self.reward.shape[1]
            obs_flat = self.obs.reshape(T * N, -1)
            global_flat = np.repeat(self.global_s, N, axis=0)
            actions_flat = self.actions.reshape(-1)
            logp_flat = self.logp.reshape(-1)
            adv_flat = np.repeat(advantages, N)
            ret_flat = np.repeat(returns, N)
        else:
            T, E, N = self.reward.shape
            obs_flat = self.obs.reshape(T * E * N, -1)
            # global_s (T, E*global_dim) → 每个 (t, e) 对应 N 个 UAV
            gs_per_step_env = self.global_s.reshape(T, E, -1)        # (T, E, global_dim)
            global_flat = np.repeat(gs_per_step_env[:, :, None], N, axis=2).reshape(T * E * N, -1)
            actions_flat = self.actions.reshape(-1)
            logp_flat = self.logp.reshape(-1)
            # advantages (T, E) → (T*E,) → 每份重复 N 份
            adv_per_te = advantages.reshape(-1)                        # (T*E,)
            adv_flat = np.repeat(adv_per_te, N)                         # (T*E*N,)
            ret_per_te = returns.reshape(-1)
            ret_flat = np.repeat(ret_per_te, N)

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

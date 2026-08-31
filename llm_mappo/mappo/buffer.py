import numpy as np


class RolloutBuffer:
    """一个 episode 内的转移缓冲，episode 结束后计算 GAE 与 returns。"""

    def __init__(self, n_agents: int, obs_dim: int, n_actions: int):
        self.n_agents = n_agents
        self.obs, self.actions, self.logits = [], [], []
        self.rewards, self.dones = [], []

    def add(self, obs, actions, logits, rewards, dones):
        self.obs.append(obs)
        self.actions.append(actions)
        self.logits.append(logits)
        self.rewards.append(rewards)
        self.dones.append(dones)

    def compute_gae(self, values: np.ndarray, gamma: float, lam: float):
        """GAE 优势估计：A_t = delta_t + gamma*lam*A_{t+1}，delta_t = r_t + gamma*V_{t+1} - V_t。"""
        T = len(self.rewards)
        adv = np.zeros((T, self.n_agents))
        last_gae = np.zeros(self.n_agents)
        for t in reversed(range(T)):
            next_v = values[t + 1] if t + 1 < T else np.zeros(self.n_agents)
            nonterm = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_v * nonterm - values[t]
            last_gae = delta + gamma * lam * nonterm * last_gae
            adv[t] = last_gae
        self.advantages = adv
        self.returns = adv + values

    def arrays(self):
        return {
            "obs": np.stack(self.obs),
            "actions": np.stack(self.actions),
            "logits": np.stack(self.logits),
            "advantages": self.advantages,
            "returns": self.returns,
        }

import numpy as np
import torch
import torch.nn.functional as F
from llm_mappo.utils.config import Config
from llm_mappo.mappo.networks import Actor, Critic
from llm_mappo.mappo.buffer import RolloutBuffer
from llm_mappo.mappo.update import ppo_actor_loss


class MAPPOTrainer:
    """集中 Critic + 分散 Actor，同构 UAV 共享参数。"""

    def __init__(self, cfg: Config, obs_dim: int, n_actions: int = 6):
        self.cfg = cfg
        self.n_agents = cfg.env.n_uav
        self.obs_dim = obs_dim
        self.actor = Actor(obs_dim, n_actions, cfg.mappo.hidden_size)
        self.critic = Critic(obs_dim * self.n_agents, cfg.mappo.hidden_size)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=cfg.mappo.lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=cfg.mappo.lr)

    def select_actions(self, obs_list, masks):
        """obs_list: list[np.ndarray(obs_dim,)]；masks: np.ndarray(n_agents, n_actions)。
        返回 (actions: (n_agents,) int, logits: (n_agents, n_actions))。"""
        actions = []
        logits_list = []
        for o, m in zip(obs_list, masks):
            o_t = torch.as_tensor(o, dtype=torch.float32).unsqueeze(0)
            logits = self.actor(o_t).squeeze(0)              # (n_actions,)
            m_t = torch.as_tensor(m, dtype=torch.float32)
            masked = logits - (1.0 - m_t) * 1e9
            probs = F.softmax(masked, dim=-1)
            actions.append(torch.multinomial(probs, 1).item())
            logits_list.append(logits.detach())
        return np.array(actions), np.stack([lg.numpy() for lg in logits_list])

    def value(self, obs_stack):
        """obs_stack: (T, n_agents, obs_dim) -> (T,) 联合价值。"""
        joint = torch.as_tensor(obs_stack, dtype=torch.float32).reshape(obs_stack.shape[0], -1)
        with torch.no_grad():
            return self.critic(joint).squeeze(-1).numpy()

    def update(self, buf: RolloutBuffer):
        data = buf.arrays()
        obs = torch.as_tensor(data["obs"], dtype=torch.float32)         # (T, N, O)
        adv = torch.as_tensor(data["advantages"], dtype=torch.float32)  # (T, N)
        ret = torch.as_tensor(data["returns"], dtype=torch.float32)     # (T, N)
        old_logits = torch.as_tensor(data["logits"], dtype=torch.float32)  # (T, N, A)
        acts = torch.as_tensor(data["actions"], dtype=torch.long)       # (T, N)
        joint = obs.reshape(obs.shape[0], -1)                           # (T, N*O)
        for _ in range(self.cfg.mappo.epochs):
            vals = self.critic(joint).squeeze(-1).unsqueeze(-1).expand_as(ret)
            loss_c = F.mse_loss(vals, ret)
            self.opt_c.zero_grad()
            loss_c.backward()
            self.opt_c.step()
            loss_a = 0.0
            for n in range(self.n_agents):
                loss_a = loss_a + ppo_actor_loss(
                    self.actor, obs[:, n], old_logits[:, n],
                    acts[:, n], adv[:, n], self.cfg.mappo.clip_eps)
            loss_a = loss_a / self.n_agents
            self.opt_a.zero_grad()
            loss_a.backward()
            self.opt_a.step()

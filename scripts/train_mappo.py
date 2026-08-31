import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from llm_mappo.utils.config import Config
from llm_mappo.utils.seed import set_seed
from llm_mappo.env.env import SearchEnv
from llm_mappo.mappo.trainer import MAPPOTrainer
from llm_mappo.mappo.buffer import RolloutBuffer


def main():
    cfg = Config()
    set_seed(cfg.seed)
    env = SearchEnv(cfg)
    obs, _ = env.reset(seed=cfg.seed)
    obs_dim = obs[0].shape[0]
    trainer = MAPPOTrainer(cfg, obs_dim)

    for ep in range(100):
        buf = RolloutBuffer(cfg.env.n_uav, obs_dim, 6)
        obs, _ = env.reset()
        ep_reward = 0.0
        for _ in range(cfg.env.max_steps):
            masks = env.action_masks()
            actions, logits = trainer.select_actions(obs, masks)
            next_obs, _, term, trunc, _ = env.step(actions)
            # 简单稠密奖励：降低不确定度（sanity 用，正式奖励见 T15/T16）
            rew = np.full(cfg.env.n_uav, 1.0 - env._avg_uncertainty())
            ep_reward += float(rew.mean())
            buf.add(obs=np.stack(obs), actions=actions, logits=logits,
                    rewards=rew, dones=np.array([float(term or trunc)] * cfg.env.n_uav))
            obs = next_obs
            if term or trunc:
                break
        vals = trainer.value(np.stack(buf.obs))
        vals = np.stack([vals] * cfg.env.n_uav, axis=-1)
        buf.compute_gae(vals, cfg.mappo.gamma, cfg.mappo.gae_lambda)
        trainer.update(buf)
        if ep % 20 == 0:
            print(f"ep {ep} reward={ep_reward:.3f}")

    print("sanity check OK")


if __name__ == "__main__":
    main()

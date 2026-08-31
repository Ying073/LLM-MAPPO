import numpy as np
from llm_mappo.env.env import SearchEnv, ACTIONS


def evaluate_reward_fn(reward_fn, env: SearchEnv, cfg, seed: int = 0) -> float:
    """Eq.(20)(21)：固定初始观测，逐步贪心选择使 R 最大的联合动作，执行一个 episode，
    用 Eq.(12a) 的原始目标 J（累计成功搜索目标数）评分。

    简化说明：贪心时把候选动作 a 及其执行后网格 next 经 info 传给 reward_fn，
    由 reward_fn 据此对动作排序（不仿真动作对下一状态的完整影响）。"""
    env.reset(seed=seed)
    gs = cfg.env.grid_size
    total = 0.0
    for _ in range(cfg.env.max_steps):
        masks = env.action_masks()
        obs_list = env._obs()
        acts = []
        for n in range(cfg.env.n_uav):
            x, y, alt = env.uav_pos[n]
            best_a, best_r = 0, -1e18
            for a in range(6):
                if masks[n, a] == 0:
                    continue
                nx, ny, nalt = int(x), int(y), int(alt)
                if a < 4:
                    dx, dy = ACTIONS[a]
                    nx = int(np.clip(x + dx, 0, gs - 1))
                    ny = int(np.clip(y + dy, 0, gs - 1))
                elif a == 4:
                    nalt = min(2, alt + 1)
                elif a == 5:
                    nalt = max(0, alt - 1)
                info = {"action": a, "next": (nx, ny, nalt)}
                r = reward_fn({"obs": obs_list[n], "agent": n}, None, info)
                if r > best_r:
                    best_a, best_r = a, r
            acts.append(best_a)
        env.step(np.array(acts))
        total += env._searched_targets()
    return float(total)

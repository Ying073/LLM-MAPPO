"""论文附录 Eq. (34) 的可审计实现。

该奖励是 joint/team reward，而不是从执行后坐标再次推演单个动作。
未在论文表格中给出数值的 chi_th、p_hp 和 d_safe 保持为显式参数，
默认值只用于当前复现实验，并写入运行配置。
"""

import numpy as np

from ..env.search_env import N_UAV, SENSE_OFFSETS_BY_H


DEFAULT_CHI_TH = 0.3
DEFAULT_P_HP = 0.8
DEFAULT_D_SAFE = 1.0


def paper_reward_components(prev_geum, env, actions, *, chi_th=DEFAULT_CHI_TH,
                            p_hp=DEFAULT_P_HP, d_safe=DEFAULT_D_SAFE):
    actions = np.asarray(actions, dtype=np.int64)
    prev_geum = np.asarray(prev_geum, dtype=np.float32)
    cur_geum = np.asarray(env.geum, dtype=np.float32)

    n_searched = float(np.asarray(env.searched).sum())
    n_unc = float(((prev_geum > chi_th) & (cur_geum <= chi_th)).sum())
    delta_u = float(prev_geum.sum(dtype=np.float64) - cur_geum.sum(dtype=np.float64))

    r_alt = 0.0
    for n in range(N_UAV):
        ix, iy, h = (int(v) for v in env.uav_pos[n])
        cells = []
        for dy, dx in SENSE_OFFSETS_BY_H[h]:
            x, y = ix + dx, iy + dy
            if 0 <= x < env.gtpm.shape[1] and 0 <= y < env.gtpm.shape[0]:
                cells.append((y, x))
        has_target = any(env.zeta[y, x] == 1 for y, x in cells)
        has_high_probability = any(env.gtpm[y, x] >= p_hp for y, x in cells)
        contains_no_target = not has_target
        if actions[n] == 5:
            r_alt += 3.0 * float(has_target)
            r_alt += 1.5 * float((not has_target) and has_high_probability)
        elif actions[n] == 4:
            r_alt += 0.2 * float(contains_no_target)

    r_dis = 0.0
    xy = np.asarray(env.uav_pos[:, :2], dtype=np.float64)
    for n in range(N_UAV):
        for m in range(n + 1, N_UAV):
            distance = float(np.linalg.norm(xy[n] - xy[m]))
            # The paper describes Rdis as a penalty for insufficient separation.
            # Clamp each pairwise term so safe separation cannot become a reward.
            r_dis += max(0.0, d_safe - distance + 1.0)

    return {
        "n_searched": n_searched,
        "n_uncertainty_crossings": n_unc,
        "delta_uncertainty": delta_u,
        "altitude_shaping": float(r_alt),
        "dispersion_penalty": float(r_dis),
    }


def compute_paper_rbest(prev_geum, env, actions, **kwargs):
    """Eq. (34): 10*Nsear + .3*Nunc + .8*DeltaU + Ralt - .5*Rdis."""
    c = paper_reward_components(prev_geum, env, actions, **kwargs)
    reward = (
        10.0 * c["n_searched"]
        + 0.3 * c["n_uncertainty_crossings"]
        + 0.8 * c["delta_uncertainty"]
        + c["altitude_shaping"]
        - 0.5 * c["dispersion_penalty"]
    )
    return float(reward), c

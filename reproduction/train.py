"""
train.py —— M2/M3/M5 主训练脚本

用法:
    # 方式 A (推荐): 从项目根目录运行
    cd "C:\\Users\\lenovo\\AI\\大创\\LLM-MAPPO_论文阅读与复现"
    python reproduction/train.py

    # 或者带 GPU
    python reproduction/train.py --device cuda

论文对应:
- 公式 18: O_n(t) 局部观测                → reproduction.env.env_wrapper
- 公式 23: PPO 裁剪目标                  → reproduction.algorithms.mappo.MAPPO.update
- 公式 12a: 任务目标                     → reproduction.reward.manual_reward + 训练日志
- 公式 20-22, 27: LRS 离线奖励塑形       → reproduction.lrs (M5)

输出:
    - 每 10 个 episode 打印一条 summary (mean_reward, mean_searched, mean_au)
    - 训练曲线保存到 reproduction/training_curve.png
    - 默认 M2/M3 用手写稠密奖励 (manual_reward)
    - 加 --use-lrs 走 LRS 路径 (M5): 训练前先跑 LRS 拿 R^best, 再喂给 env
    - 加 --use-dpes 启用 DPES 信息素 patch
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")                    # 无显示器也能保存图
import matplotlib.pyplot as plt

# 把 reproduction 的父目录加进 sys.path，这样能以 reproduction.env.xxx 的方式 import
HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from reproduction.env.env_wrapper import MultiAgentWrapper
from reproduction.algorithms.mappo import MAPPO
from reproduction.algorithms.buffer import RolloutBuffer
from reproduction.env.search_env import N_UAV, MAX_STEPS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--total-episodes", type=int, default=200)
    p.add_argument("--rollout-len", type=int, default=MAX_STEPS)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--use-dpes", action="store_true", default=False,
                   help="是否启用 DPES 双模式信息素 (M3, 公式 13–17)")
    p.add_argument("--use-lrs", action="store_true", default=False,
                   help="是否启用 LRS 离线 LLM 奖励塑形 (M5, 公式 20–22, 27–29)")
    p.add_argument("--lrs-K", type=int, default=5,
                   help="LRS 主循环迭代次数 (算法 2 的 K)")
    p.add_argument("--lrs-seed", type=int, default=None,
                   help="LRS 用的种子；默认与 --seed 相同")
    p.add_argument("--out-name", type=str, default="training_curve.png",
                   help="训练曲线文件名")
    p.add_argument("--save-history", type=str, default=None,
                   help="保存训练历史为 .npz (用于跨实验对比); 文件名后缀 .npz")
    p.add_argument("--minibatch-size", type=int, default=256,
                   help="PPO minibatch size (M6: GPU 适合大 batch, 默认 256 替 v1 的 64)")
    return p.parse_args()


def run_lrs(args):
    """M5 预训练阶段: 跑一遍 LRS 拿 R^best 代码 + 可调用函数.
    
    对应论文 §IV-D: 'Before MAPPO training, the offline LRS scheme generates
    and optimizes the reward function' —— 训练前 LRS 离线完成。
    """
    from reproduction.lrs import LRS, compile_reward
    from reproduction.env.search_env import SearchEnv

    lrs_seed = args.lrs_seed if args.lrs_seed is not None else args.seed
    print(f"[lrs] starting offline LRS K={args.lrs_K}, seed={lrs_seed} ...", flush=True)
    env = SearchEnv(seed=lrs_seed)
    lrs = LRS(K=args.lrs_K, seed=lrs_seed)
    t0 = time.time()
    best_fn, best_code, best_J, best_metrics = lrs.run(env, seed=lrs_seed)
    dt = time.time() - t0
    print(f"[lrs] done in {dt:.1f}s, R^best J={best_J:+.3f}, "
          f"area_unc={best_metrics['area_unc']:.4f}, searched={best_metrics['searched']}")
    return best_fn


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- [M5] 先跑 LRS 拿 R^best ----
    lrs_reward_fn = None
    if args.use_lrs:
        lrs_reward_fn = run_lrs(args)

    env = MultiAgentWrapper(
        seed=args.seed,
        use_dpes=args.use_dpes,
        lrs_reward_fn=lrs_reward_fn,
    )
    obs_list = env.reset()
    obs_dim = obs_list[0].shape[0]
    global_dim = obs_dim * N_UAV

    print(f"[init] N_UAV={N_UAV}, obs_dim={obs_dim}, global_dim={global_dim}, "
          f"act_dim=6, rollout_len={args.rollout_len}, device={args.device}, "
          f"use_dpes={args.use_dpes}, use_lrs={args.use_lrs}, "
          f"reward_path={'lrs' if lrs_reward_fn is not None else 'manual'}")

    mappo = MAPPO(
        obs_dim=obs_dim,
        global_dim=global_dim,
        act_dim=6,
        device=args.device,
        minibatch_size=args.minibatch_size,
    )
    buffer = RolloutBuffer(
        rollout_len=args.rollout_len,
        n_uav=N_UAV,
        obs_dim=obs_dim,
        global_dim=global_dim,
    )

    # 日志
    rewards_hist, searched_hist, au_hist, actor_loss_hist, critic_loss_hist = [], [], [], [], []

    for ep in range(args.total_episodes):
        buffer.reset()
        ep_reward = 0.0
        ep_searched = 0
        ep_au_final = 0.0

        for t in range(args.rollout_len):
            # (1) 选动作
            actions, logp = mappo.select_actions(obs_list)

            # (2) 算 value (给 GAE 用)
            value = mappo.get_value(env.get_global_state())

            # (3) 推进一步
            next_obs_list, shared_r, done, info = env.step(actions)

            # (4) 把 per-agent logp / reward 也存进 buffer
            per_agent_r = info["per_agent_reward"]
            buffer.store(
                obs=obs_list,
                global_s=env.get_global_state(),
                actions=actions,
                logp=logp,
                reward=per_agent_r,
                done=done,
                value=value,
            )

            ep_reward += shared_r
            ep_searched = info["searched_count"]
            ep_au_final = info["area_uncertainty"]
            obs_list = next_obs_list

            if done:
                break

        # 用 batch 末尾的 obs 算 last_value (不更新网络，只是 bootstrap)
        last_value = 0.0 if done else mappo.get_value(env.get_global_state())
        a_loss, c_loss = mappo.update(buffer, last_value)

        rewards_hist.append(ep_reward)
        searched_hist.append(ep_searched)
        au_hist.append(ep_au_final)
        actor_loss_hist.append(a_loss)
        critic_loss_hist.append(c_loss)

        if (ep + 1) % args.log_every == 0 or ep == 0:
            recent = slice(max(0, ep - 9), ep + 1)
            print(f"[ep {ep+1:4d}/{args.total_episodes}] "
                  f"reward={np.mean(rewards_hist[recent]):+.3f}  "
                  f"searched={np.mean(searched_hist[recent]):.1f}/15  "
                  f"area_unc={np.mean(au_hist[recent]):.3f}  "
                  f"actor_loss={a_loss:+.3f}  critic_loss={c_loss:+.3f}")

        # 重新开始下一回合
        obs_list = env.reset()

    # ----------------- 画训练曲线 -----------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    # 用滑动平均让曲线平滑
    def smooth(xs, w=10):
        if len(xs) < w:
            return xs
        return np.convolve(xs, np.ones(w) / w, mode="valid")

    axes[0].plot(smooth(rewards_hist), label="episode reward (smoothed)")
    axes[0].set_xlabel("episode"); axes[0].set_ylabel("mean reward")
    axes[0].set_title("Reward"); axes[0].grid(True, alpha=0.3)

    axes[1].plot(smooth(searched_hist), color="tab:red", label="searched")
    axes[1].set_xlabel("episode"); axes[1].set_ylabel("# searched targets")
    axes[1].set_title("Searched count (max 15)"); axes[1].grid(True, alpha=0.3)

    axes[2].plot(smooth(au_hist), color="tab:green", label="area uncertainty")
    axes[2].set_xlabel("episode"); axes[2].set_ylabel("area uncertainty")
    axes[2].set_title("Area uncertainty (lower = better)"); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(HERE, args.out_name)
    plt.savefig(out, dpi=120)
    print(f"\n[train] saved training curve to: {out}")
    print(f"[train] final stats: reward={rewards_hist[-1]:+.3f} "
          f"searched={searched_hist[-1]} area_unc={au_hist[-1]:.4f}")

    # 可选: 保存 raw history 供跨实验对比
    if args.save_history:
        hist_path = args.save_history if args.save_history.endswith(".npz") else args.save_history + ".npz"
        np.savez_compressed(
            hist_path,
            rewards=np.asarray(rewards_hist, dtype=np.float32),
            searched=np.asarray(searched_hist, dtype=np.float32),
            au=np.asarray(au_hist, dtype=np.float32),
            actor_loss=np.asarray(actor_loss_hist, dtype=np.float32),
            critic_loss=np.asarray(critic_loss_hist, dtype=np.float32),
            config=np.array([args.use_dpes, args.use_lrs]),
        )
        print(f"[train] saved raw history to {hist_path}")


if __name__ == "__main__":
    args = parse_args()
    t0 = time.time()
    train(args)
    print(f"[train] total time: {time.time() - t0:.1f}s")

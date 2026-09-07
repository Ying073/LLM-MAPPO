"""
train_batched.py —— 用 BatchedMultiAgentWrapper 训练 (M8 接力 TODO #4 落地)

与 train.py 的差别:
- env 用 BatchedMultiAgentWrapper (n 个独立 env 并行)
- 每个 outer "episode" 实际上是 n 个并行 env 各跑 max_steps
- 总 episode 数 = args.total_episodes (每个并行 env 算 1 个 episode)
- 不同 env 独立 reset (某个 env done 时, 自动 partial reset)

用法:
    python reproduction/train_batched.py --batch-envs 16 --total-episodes 30 --device cuda --save-history batched_m2_smoke.npz
"""
import argparse
import importlib.util
import os
import sys
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from reproduction.env.batched_env_wrapper import BatchedMultiAgentWrapper
from reproduction.env.env_wrapper import MultiAgentWrapper
from reproduction.algorithms.mappo import MAPPO
from reproduction.algorithms.buffer import RolloutBuffer
from reproduction.env.search_env import N_UAV, MAX_STEPS
from reproduction import lrs as _lrs   # [M12] 用于 --lrs-cache 注入 LX/LY/N_UAV globals


def compute_n_outer(total_episodes: int, n_envs: int) -> int:
    if total_episodes <= 0 or n_envs <= 0:
        raise ValueError("total_episodes and n_envs must be positive")
    return math.ceil(total_episodes / n_envs)


def capture_rng_state():
    state = {
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([rng_state.cpu() for rng_state in state["cuda"]])


def save_periodic_checkpoint(mappo, training_path, policy_dir, progress, *, outer_number):
    """Save the resumable latest state plus a non-overwritten policy snapshot."""
    training_path = Path(training_path).resolve()
    policy_dir = Path(policy_dir).resolve()
    training_path.parent.mkdir(parents=True, exist_ok=True)
    policy_dir.mkdir(parents=True, exist_ok=True)
    policy_path = policy_dir / f"policy_outer_{int(outer_number):06d}.pt"
    if policy_path.exists():
        raise FileExistsError(f"policy snapshot already exists: {policy_path}")
    mappo.save_training_checkpoint(training_path, progress)
    mappo.save_checkpoint(policy_path)
    return policy_path


def load_lrs_cache(cache_path: str):
    """[M12] 加载 LRS 离线产出的 R^best.py 文件, 跳过 8 次重复跑 LRS K=5.

    `lrs.compile_reward()` 用 exec(code, ns) 注入 N_UAV/LX/LY/TARGET_CONFIRM_THRESHOLD/np
    (见 lrs.py:294-298). 我们 importlib 时同样要把这些注入到 module.globals,
    这样顶层 `def reward(env, n, action, prev_au): ...` body 里的 LX/LY 等 free
    variable 才能在调用时找到.

    Returns:
        callable: reward(env, n, action, prev_au) -> float
    """
    abs_path = os.path.abspath(cache_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"[lrs-cache] not found: {abs_path}")
    spec = importlib.util.spec_from_file_location("rbest_cache", abs_path)
    mod = importlib.util.module_from_spec(spec)
    # 模仿 lrs.compile_reward() 的 ns:
    mod.__dict__.update({
        "N_UAV": _lrs.N_UAV,
        "LX": _lrs.LX,
        "LY": _lrs.LY,
        "TARGET_CONFIRM_THRESHOLD": _lrs.TARGET_CONFIRM_THRESHOLD,
        "np": _lrs.np,
    })
    spec.loader.exec_module(mod)
    if not hasattr(mod, "reward"):
        raise RuntimeError(f"[lrs-cache] {abs_path} does not expose `reward(env,n,action,prev_au)`")
    print(f"[lrs-cache] loaded reward() from {abs_path}")
    return mod.reward


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--total-episodes", type=int, default=200,
                   help="总 episode 数 (每个并行 env 算 1 个)")
    p.add_argument("--rollout-len", type=int, default=MAX_STEPS)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-envs", type=int, default=16,
                   help="并行 env 数 (1=单 env 模式, 同 train.py)")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--use-dpes", action="store_true", default=False)
    p.add_argument("--use-lrs", action="store_true", default=False)
    p.add_argument("--lrs-K", type=int, default=5)
    p.add_argument("--lrs-seed", type=int, default=None)
    p.add_argument("--llm-backend", type=str, default="canned",
                   choices=["canned", "deepseek-r1", "deepseek-v3"])
    p.add_argument("--reward-source", choices=["paper-rbest", "manual", "lrs"],
                   default="paper-rbest")
    p.add_argument("--lrs-cache", type=str, default=None,
                   help="仅在 --reward-source lrs 时加载历史奖励函数；论文对齐默认路径不使用")
    p.add_argument("--out-name", type=str, default="training_curve_batched.png")
    p.add_argument("--save-history", type=str, default=None)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--checkpoint-out", type=str, default=None)
    p.add_argument("--training-checkpoint", type=str, default=None,
                   help="定期覆盖写入的完整训练状态（用于断点续训）")
    p.add_argument("--policy-checkpoint-dir", type=str, default=None,
                   help="保留每个周期的冻结策略，供独立验证集选择模型")
    p.add_argument("--checkpoint-every", type=int, default=0,
                   help="每多少个 outer iteration 保存完整训练状态；0 表示关闭")
    p.add_argument("--resume-from", type=str, default=None,
                   help="从 --training-checkpoint 生成的完整训练状态继续")
    return p.parse_args()


def run_lrs(args):
    """预跑 LRS 拿 R^best (与 train.py 相同)."""
    from reproduction.lrs import LRS, make_backend
    from reproduction.env.search_env import SearchEnv
    lrs_seed = args.lrs_seed if args.lrs_seed is not None else args.seed
    backend = make_backend(args.llm_backend)
    print(f"[lrs] starting offline LRS K={args.lrs_K}, seed={lrs_seed}, "
          f"backend={type(backend).__name__} ({args.llm_backend}) ...", flush=True)
    env = SearchEnv(seed=lrs_seed)
    lrs = LRS(llm=backend, K=args.lrs_K, seed=lrs_seed)
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
    if args.use_lrs or args.reward_source == "lrs":
        lrs_reward_fn = load_lrs_cache(args.lrs_cache) if args.lrs_cache else run_lrs(args)
    elif args.lrs_cache:
        raise ValueError("--lrs-cache requires --reward-source lrs (or --use-lrs)")

    # 判断模式
    n_envs = max(1, args.batch_envs)
    use_batched = (n_envs > 1)
    if use_batched:
        env = BatchedMultiAgentWrapper(
            n_envs=n_envs, base_seed=args.seed,
            use_dpes=args.use_dpes, lrs_reward_fn=lrs_reward_fn,
            use_paper_reward=args.reward_source == "paper-rbest" and lrs_reward_fn is None,
        )
        obs_batch = env.reset()                             # (n, N_UAV, OBS_DIM)
        obs_dim = obs_batch.shape[-1]
    else:
        env = MultiAgentWrapper(
            seed=args.seed, use_dpes=args.use_dpes,
            lrs_reward_fn=lrs_reward_fn,
            use_paper_reward=args.reward_source == "paper-rbest" and lrs_reward_fn is None,
        )
        obs_list = env.reset()
        obs_dim = obs_list[0].shape[0]
    global_dim = obs_dim * N_UAV

    print(f"[init] mode={'batched' if use_batched else 'single'}, "
          f"n_envs={n_envs}, N_UAV={N_UAV}, obs_dim={obs_dim}, global_dim={global_dim}, "
          f"act_dim=6, rollout_len={args.rollout_len}, device={args.device}, "
          f"use_dpes={args.use_dpes}, use_lrs={args.use_lrs}, "
          f"reward_path={'lrs' if lrs_reward_fn is not None else args.reward_source}")

    mappo = MAPPO(
        obs_dim=obs_dim,
        global_dim=global_dim,
        act_dim=6,
        n_agents=N_UAV,
        device=args.device,
        minibatch_size=args.minibatch_size,
    )
    buffer = RolloutBuffer(
        rollout_len=args.rollout_len,
        n_uav=N_UAV,
        obs_dim=obs_dim,
        global_dim=global_dim,
        n_envs=n_envs,
    )

    rewards_hist, searched_hist, au_hist, actor_loss_hist, critic_loss_hist = [], [], [], [], []
    # per-env arrays (shape: n_outer × n_envs) — 诊断并行采样方差
    rewards_per_env, searched_per_env, au_per_env = [], [], []

    n_outer = compute_n_outer(args.total_episodes, n_envs)
    print(f"[init] total_episodes={args.total_episodes} ÷ n_envs={n_envs} → n_outer={n_outer} 个外迭代")

    start_outer = 0
    if args.resume_from:
        mappo, progress = MAPPO.from_training_checkpoint(args.resume_from, device=args.device)
        saved_config = progress["run_config"]
        current_config = {
            "n_envs": n_envs,
            "use_batched": use_batched,
            "use_dpes": args.use_dpes,
            "reward_source": args.reward_source,
            "seed": args.seed,
            "rollout_len": args.rollout_len,
            "obs_dim": obs_dim,
            "global_dim": global_dim,
        }
        mismatches = {
            key: (saved_config.get(key), value)
            for key, value in current_config.items()
            if saved_config.get(key) != value
        }
        if mismatches:
            raise ValueError(f"resume configuration mismatch: {mismatches}")
        start_outer = int(progress["next_outer"])
        if start_outer > n_outer:
            raise ValueError(f"checkpoint outer {start_outer} exceeds requested total {n_outer}")
        env = progress["env"]
        if use_batched:
            obs_batch = progress["observations"]
        else:
            obs_list = progress["observations"]
        histories = progress["histories"]
        rewards_hist = list(histories["rewards"])
        searched_hist = list(histories["searched"])
        au_hist = list(histories["au"])
        actor_loss_hist = list(histories["actor_loss"])
        critic_loss_hist = list(histories["critic_loss"])
        rewards_per_env = list(histories["rewards_per_env"])
        searched_per_env = list(histories["searched_per_env"])
        au_per_env = list(histories["au_per_env"])
        restore_rng_state(progress["rng_state"])
        print(f"[resume] loaded {args.resume_from}; continuing at outer {start_outer + 1}/{n_outer}",
              flush=True)

    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be non-negative")
    if args.checkpoint_every and not args.training_checkpoint:
        raise ValueError("--checkpoint-every requires --training-checkpoint")

    for outer in range(start_outer, n_outer):
        buffer.reset()
        ep_reward = np.zeros(n_envs, dtype=np.float32)
        ep_searched = np.zeros(n_envs, dtype=np.int32)
        ep_au_final = np.zeros(n_envs, dtype=np.float32)
        done_flags = np.zeros(n_envs, dtype=bool)

        for t in range(args.rollout_len):
            # (1) 选动作
            if use_batched:
                action_masks = env.get_action_masks()
                current_global_state = env.get_global_state()
                actions, logp = mappo.select_actions_batched(obs_batch, action_masks)
                value = mappo.get_value_batched(current_global_state)
            else:
                action_masks = env.get_action_masks()
                current_global_state = env.get_global_state()
                actions, logp = mappo.select_actions(obs_list, action_masks)
                value = mappo.get_value(current_global_state)

            # (2) 推进一步
            if use_batched:
                next_obs_batch, shared_r, done, per_agent_r, infos = env.step(actions)
                ep_reward += shared_r
                for b in range(n_envs):
                    ep_searched[b] = infos[b]["found_count"]
                    ep_au_final[b] = infos[b]["area_uncertainty"]
                done_flags[:] = done
                buffer.store_batch(
                    obs=obs_batch,
                    global_s=current_global_state,
                    actions=actions, logp=logp,
                    reward=per_agent_r, done=done_flags, value=value,
                    action_mask=action_masks,
                )
                obs_batch = next_obs_batch
                # (3) 对 done env partial_reset
                if done.any():
                    for b in range(n_envs):
                        if done[b]:
                            env.partial_reset(b)
                    obs_batch = env._obs_batch()
            else:
                # 单 env 路径 (与 train.py 完全相同)
                next_obs_list, shared_r, done, info = env.step(actions)
                per_agent_r = info["per_agent_reward"]
                buffer.store(
                    obs=obs_list,
                    global_s=current_global_state,
                    actions=actions, logp=logp,
                    reward=per_agent_r, done=done, value=value,
                    action_mask=action_masks,
                )
                ep_reward += shared_r
                ep_searched = info["found_count"]
                ep_au_final = info["area_uncertainty"]
                obs_list = next_obs_list
                done_flags[:] = done

            if use_batched and done_flags.all():
                break
            if (not use_batched) and done_flags[0]:
                break

        # bootstrap last_value
        if use_batched:
            last_value = np.zeros(n_envs, dtype=np.float32) if done_flags.all() \
                          else mappo.get_value_batched(env.get_global_state())
        else:
            last_value = 0.0 if done_flags[0] else mappo.get_value(env.get_global_state())
        a_loss, c_loss = mappo.update(buffer, last_value)

        # 记录 (batched: mean over n_envs 当 "本轮 reward")
        if use_batched:
            rewards_hist.append(float(ep_reward.mean()))
            searched_hist.append(float(ep_searched.mean()))
            au_hist.append(float(ep_au_final.mean()))
            # per-env diagnostics
            rewards_per_env.append(ep_reward.copy())
            searched_per_env.append(ep_searched.copy())
            au_per_env.append(ep_au_final.copy())
        else:
            rewards_hist.append(float(ep_reward))
            searched_hist.append(float(ep_searched))
            au_hist.append(float(ep_au_final))
        actor_loss_hist.append(a_loss)
        critic_loss_hist.append(c_loss)

        if (outer + 1) % args.log_every == 0 or outer == 0:
            recent = slice(max(0, outer - 9), outer + 1)
            print(f"[outer {outer+1:4d}/{n_outer}] "
                  f"reward={np.mean(rewards_hist[recent]):+.3f}  "
                  f"searched={np.mean(searched_hist[recent]):.1f}/15  "
                  f"area_unc={np.mean(au_hist[recent]):.3f}  "
                  f"actor_loss={a_loss:+.3f}  critic_loss={c_loss:+.3f}",
                  flush=True)

        # 下一外迭代: reset (单 env) / 保留 (batched, partial reset 已 done)
        if not use_batched:
            obs_list = env.reset()

        if args.checkpoint_every and (outer + 1) % args.checkpoint_every == 0:
            checkpoint_path = os.path.abspath(args.training_checkpoint)
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            progress = {
                "next_outer": outer + 1,
                "run_config": {
                    "n_envs": n_envs,
                    "use_batched": use_batched,
                    "use_dpes": args.use_dpes,
                    "reward_source": args.reward_source,
                    "seed": args.seed,
                    "rollout_len": args.rollout_len,
                    "obs_dim": obs_dim,
                    "global_dim": global_dim,
                },
                "env": env,
                "observations": obs_batch if use_batched else obs_list,
                "histories": {
                    "rewards": rewards_hist,
                    "searched": searched_hist,
                    "au": au_hist,
                    "actor_loss": actor_loss_hist,
                    "critic_loss": critic_loss_hist,
                    "rewards_per_env": rewards_per_env,
                    "searched_per_env": searched_per_env,
                    "au_per_env": au_per_env,
                },
                "rng_state": capture_rng_state(),
            }
            if args.policy_checkpoint_dir:
                policy_path = save_periodic_checkpoint(
                    mappo,
                    checkpoint_path,
                    args.policy_checkpoint_dir,
                    progress,
                    outer_number=outer + 1,
                )
                print(f"[checkpoint] saved policy snapshot: {policy_path}", flush=True)
            else:
                mappo.save_training_checkpoint(checkpoint_path, progress)
            print(f"[checkpoint] saved resumable state at outer {outer + 1}: {checkpoint_path}",
                  flush=True)

    # 画训练曲线
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    def smooth(xs, w=10):
        if len(xs) < w:
            return xs
        return np.convolve(xs, np.ones(w) / w, mode="valid")

    axes[0].plot(smooth(rewards_hist), label="episode reward (smoothed)")
    axes[0].set_xlabel("outer iteration"); axes[0].set_ylabel("mean reward")
    axes[0].set_title("Reward"); axes[0].grid(True, alpha=0.3)
    axes[1].plot(smooth(searched_hist), color="tab:red", label="searched")
    axes[1].set_xlabel("outer iteration"); axes[1].set_ylabel("# searched targets")
    axes[1].set_title("Searched count (max 15)"); axes[1].grid(True, alpha=0.3)
    axes[2].plot(smooth(au_hist), color="tab:green", label="area uncertainty")
    axes[2].set_xlabel("outer iteration"); axes[2].set_ylabel("area uncertainty")
    axes[2].set_title("Area uncertainty (lower = better)"); axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    # M12 fix: --out-name 有时传带目录的相对路径 (reproduction/m12_results/x.png) 或绝对路径,
    # 旧代码 os.path.join(HERE, ...) 会强制把相对路径再拼到 reproduction/ 后面 → 双重前缀 → 目录不存在。
    # 改为与 --save-history (npz) 同一套解析: 一律相对当前工作目录(项目根)或绝对路径, 并自动建父目录。
    out = os.path.normpath(args.out_name)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"\n[train] saved training curve to: {out}")
    print(f"[train] final stats: reward={rewards_hist[-1]:+.3f} "
          f"searched={searched_hist[-1]} area_unc={au_hist[-1]:.4f}")

    if args.save_history:
        hist_path = os.path.normpath(args.save_history if args.save_history.endswith(".npz") else args.save_history + ".npz")
        os.makedirs(os.path.dirname(hist_path) or ".", exist_ok=True)
        save_kwargs = dict(
            rewards=np.asarray(rewards_hist, dtype=np.float32),
            searched=np.asarray(searched_hist, dtype=np.float32),
            au=np.asarray(au_hist, dtype=np.float32),
            actor_loss=np.asarray(actor_loss_hist, dtype=np.float32),
            critic_loss=np.asarray(critic_loss_hist, dtype=np.float32),
            config=np.array([args.use_dpes, args.use_lrs, n_envs]),
            reward_source=np.asarray(args.reward_source),
            requested_episodes=np.asarray(args.total_episodes, dtype=np.int64),
            actual_episodes=np.asarray(n_outer * n_envs, dtype=np.int64),
            seed=np.asarray(args.seed, dtype=np.int64),
        )
        # per-env diagnostics (not independent trained-policy seeds)
        if use_batched and rewards_per_env:
            save_kwargs["rewards_per_env"] = np.stack(rewards_per_env).astype(np.float32)
            save_kwargs["searched_per_env"] = np.stack(searched_per_env).astype(np.float32)
            save_kwargs["au_per_env"] = np.stack(au_per_env).astype(np.float32)
        np.savez_compressed(hist_path, **save_kwargs)
        print(f"[train] saved raw history to {hist_path}")

    if args.checkpoint_out:
        checkpoint_path = os.path.abspath(args.checkpoint_out)
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        mappo.save_checkpoint(checkpoint_path)
        print(f"[train] saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    args = parse_args()
    t0 = time.time()
    train(args)
    print(f"[train] total time: {time.time() - t0:.1f}s")

"""
plot_8seed.py —— [M12] 8 seed mean±std 训练曲线出图脚本

输入: 一个目录里放 N 份 batched_m12_seed{i}_3000.npz (i=0..N-1, 一般 N=8),
       以及 1 份 Canned (paper-published) 基准如 batched_m5_8seed.npz.

每份 npz 字段:
    rewards        shape (n_outer,)             - 跨 env 平均的 reward (scalar)
    rewards_per_env shape (n_outer, n_envs)     - 每个 env 自己的 reward
    searched / au  同上
    au_per_env / searched_per_env 同上

可读性策略:
- 每个 seed 1 条曲线 = (每 outer 跨 env 取 mean)
- 8 seed 算 mean ± std 跨 n_outer x
- Canned 用同一份 npz 的 rewards_per_env 当 8 env 的 "8 个 seed" (注: 这是 M10 的妥协 —
  Canned 实际上是 1 个 run 共享权重,而非 8 个独立 network, 但作为 trend baseline 可用)

用法:
    python reproduction/plot_8seed.py --results-dir reproduction/m12_results
                                      --canned reproduction/batched_m5_8seed.npz
                                      --out comparison_m12_8seed_vs_canned.png
"""
import argparse
import glob
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_seed_runs(results_dir: str):
    """读一批 batched_m12_seed{i}*.npz, 返回 (rewards, searched, au) 三元 (n_seed, n_outer).

    n_seed = 文件数, n_outer = min(n_outer across files, 短的截断到长的最短).
    """
    pattern = os.path.join(results_dir, "batched_m12_seed*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"[plot_8seed] no files matching {pattern}")
    print(f"[plot_8seed] found {len(files)} seed files:")
    for f in files:
        print(f"  {f}")

    runs = {"rewards": [], "searched": [], "au": []}
    for f in files:
        z = np.load(f)
        for k in ("rewards", "searched", "au"):
            v = z.get(k)
            if v is None:
                raise KeyError(f"[plot_8seed] {f} lacks `{k}` field")
            runs[k].append(v.astype(np.float32))
    # 截断到最短 (兼容少量 outer 不齐)
    min_outer = min(len(r) for r in runs["rewards"])
    for k in runs:
        runs[k] = np.stack([r[:min_outer] for r in runs[k]])  # (n_seed, min_outer)
    print(f"[plot_8seed] truncated to min_outer={min_outer} across all seeds")
    return runs, len(files)


def load_canned_baseline(path: str):
    """读 Canned 8-env 共享 1 网络的 npz, 返回 rewards/searched/au (1, n_outer)."""
    if not path or not os.path.exists(path):
        print(f"[plot_8seed] no canned baseline (path={path!r}), skip")
        return None
    z = np.load(path)
    out = {}
    for k in ("rewards", "searched", "au"):
        v = z.get(k)
        if v is not None:
            out[k] = v.astype(np.float32)[None, :]   # (1, n_outer)
    if not out:
        return None
    n_outer = min(v.shape[1] for v in out.values())
    for k in out:
        out[k] = out[k][:, :n_outer]
    print(f"[plot_8seed] canned baseline n_outer={n_outer}, keys={list(out.keys())}")
    return out


def smooth(xs, w=10):
    if xs.ndim == 1:
        if len(xs) < w:
            return xs
        return np.convolve(xs, np.ones(w) / w, mode="valid")
    # 2D: (n_seed, n_outer), 沿 axis=1 平滑
    out = np.zeros_like(xs)
    for i in range(xs.shape[0]):
        row = xs[i]
        if len(row) < w:
            out[i] = row
        else:
            out[i, w - 1:] = np.convolve(row, np.ones(w) / w, mode="valid")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True,
                   help="包含 batched_m12_seed*.npz 的目录")
    p.add_argument("--canned", default=None,
                   help="可选: Canned 基准 npz (如 batched_m5_8seed.npz)")
    p.add_argument("--out", default="m12_8seed_vs_canned.png")
    p.add_argument("--smooth-window", type=int, default=10)
    args = p.parse_args()

    runs, n_seed = load_seed_runs(args.results_dir)
    canned = load_canned_baseline(args.canned)

    # 平滑 (每 seed 各自先 smooth, 再做 mean±std)
    sm = {k: smooth(v, w=args.smooth_window) for k, v in runs.items()}
    mean = {k: v.mean(axis=0) for k, v in sm.items()}
    std = {k: v.std(axis=0) for k, v in sm.items()}
    n_outer = mean["rewards"].shape[0]
    x = np.arange(n_outer)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # (a) Reward
    ax = axes[0]
    ax.plot(x, mean["rewards"], color="tab:blue", lw=2,
            label=f"LLM-MAPPO (R^best, {n_seed} seeds, mean)")
    ax.fill_between(x, mean["rewards"] - std["rewards"], mean["rewards"] + std["rewards"],
                    color="tab:blue", alpha=0.2, label=f"{n_seed} seeds ± std")
    if canned is not None and "rewards" in canned:
        cx = np.arange(canned["rewards"].shape[1])
        ax.plot(cx, canned["rewards"][0], color="tab:gray", ls="--", lw=1.5,
                label="Canned (8-env shared, 1 net)")
    ax.set_xlabel("outer iteration"); ax.set_ylabel("mean reward")
    ax.set_title("(a) Episode reward"); ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=9)

    # (b) Area uncertainty (越低越好)
    ax = axes[1]
    ax.plot(x, mean["au"], color="tab:green", lw=2, label=f"{n_seed} seeds mean")
    ax.fill_between(x, np.maximum(mean["au"] - std["au"], 0), mean["au"] + std["au"],
                    color="tab:green", alpha=0.2, label="± std")
    if canned is not None and "au" in canned:
        cx = np.arange(canned["au"].shape[1])
        ax.plot(cx, canned["au"][0], color="tab:gray", ls="--", lw=1.5,
                label="Canned (8-env)")
    ax.set_xlabel("outer iteration"); ax.set_ylabel("area uncertainty")
    ax.set_title("(b) Area uncertainty (↓ better)"); ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=9)

    # (c) Searched count (越高越好, max=15)
    ax = axes[2]
    ax.plot(x, mean["searched"], color="tab:red", lw=2, label=f"{n_seed} seeds mean")
    ax.fill_between(x, mean["searched"] - std["searched"], mean["searched"] + std["searched"],
                    color="tab:red", alpha=0.2, label="± std")
    if canned is not None and "searched" in canned:
        cx = np.arange(canned["searched"].shape[1])
        ax.plot(cx, canned["searched"][0], color="tab:gray", ls="--", lw=1.5,
                label="Canned (8-env)")
    ax.set_xlabel("outer iteration"); ax.set_ylabel("# searched targets")
    ax.set_title("(c) Searched count (↑ better, max=15)"); ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=9)

    plt.suptitle(f"[M12] LLM-MAPPO 8-seed mean±std "
                 f"(batch_envs=16, total_eps=3000/seed, smooth w={args.smooth_window})",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.abspath(args.out)
    plt.savefig(out_path, dpi=120)
    print(f"\n[plot_8seed] saved figure to {out_path}")

    # 顺便打印一行 summary (终端 grep 友好)
    last = -1
    print(f"[summary] outer={n_outer} ({n_seed} seeds): "
          f"reward={mean['rewards'][last]:+.2f}±{std['rewards'][last]:.2f}  "
          f"au={mean['au'][last]:.4f}±{std['au'][last]:.4f}  "
          f"searched={mean['searched'][last]:.2f}±{std['searched'][last]:.2f}/15")


if __name__ == "__main__":
    main()

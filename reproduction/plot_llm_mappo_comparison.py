"""
plot_llm_mappo_comparison.py —— M2/M3/M5 端到端训练曲线对比图

把多个 train.py --save-history 输出的 .npz 叠在同一张图上。
对应论文 §V-A / Fig. 4: 不同 MARL 算法收敛曲线对比。

用法:
    1) 跑 baseline, 用 --save-history 保存 raw:
        python train.py --total-episodes 150 --seed 42 \\
            --save-history hist_m2.npz     # M2 baseline
        python train.py --total-episodes 150 --seed 42 --use-dpes \\
            --save-history hist_m3.npz     # M3 +DPES
        python train.py --total-episodes 150 --seed 42 --use-dpes --use-lrs \\
            --save-history hist_m5.npz     # M5 LLM-MAPPO

    2) 画图:
        python plot_llm_mappo_comparison.py --inputs \\
            hist_m2.npz hist_m3.npz hist_m5.npz \\
            --labels MAPPO MAPPO+DPES LLM-MAPPO \\
            --out comparison_llm_mappo.png
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--out", default="comparison_llm_mappo.png")
    p.add_argument("--smooth", type=int, default=10,
                   help="滑动平均窗口 (默认 10)")
    return p.parse_args()


def smooth(xs, w=10):
    if len(xs) < w:
        return xs
    return np.convolve(xs, np.ones(w) / w, mode="same")


def main(args):
    assert len(args.inputs) == len(args.labels), \
        "inputs 和 labels 数量必须一致"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    colors = ["tab:blue", "tab:orange", "tab:red", "tab:green", "tab:purple"]
    for c, (path, label) in enumerate(zip(args.inputs, args.labels)):
        data = np.load(path, allow_pickle=True)
        rewards = data["rewards"]
        searched = data["searched"]
        au = data["au"]
        config = bool(data["config"][0]) if "config" in data.files else None
        lrs = bool(data["config"][1]) if "config" in data.files else None

        axes[0].plot(smooth(rewards, args.smooth), color=colors[c % len(colors)], label=label)
        axes[1].plot(smooth(searched, args.smooth), color=colors[c % len(colors)], label=label)
        axes[2].plot(smooth(au, args.smooth), color=colors[c % len(colors)], label=label)

    axes[0].set_xlabel("episode")
    axes[0].set_ylabel("mean reward (smoothed)")
    axes[0].set_title("Episode reward")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("episode")
    axes[1].set_ylabel("# searched targets (smoothed)")
    axes[1].set_title("Cumulative searched count (max 15)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].set_xlabel("episode")
    axes[2].set_ylabel("area uncertainty (smoothed)")
    axes[2].set_title("Terminal area uncertainty (lower = better)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.suptitle("LLM-MAPPO ablation (M2/M3/M5)", y=1.02, fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"[plot] saved to {out_path}")
    # Final sanity print
    for path, label in zip(args.inputs, args.labels):
        data = np.load(path, allow_pickle=True)
        r = data["rewards"]; s = data["searched"]; u = data["au"]
        print(f"  {label:20s} reward={r.mean():+.2f}±{r.std():.2f}  "
              f"searched={s.mean():.1f}/15  area_unc={u.mean():.4f}")


if __name__ == "__main__":
    main(parse_args())
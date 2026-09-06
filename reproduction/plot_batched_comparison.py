"""Plot M2/M3/M5 batched 150-episode comparison (batched env, GPU).

三联图: reward / searched / area_unc 三件套随外迭代变化.
源数据: batched_m{2,3,5}_150.npz (每文件 9 个 outer iter)
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_npz(path: Path):
    data = np.load(path)
    return {
        "rewards": data["rewards"],
        "searched": data["searched"],
        "au": data["au"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="comparison_llm_mappo_batched_150.png")
    ap.add_argument("--data-dir", default=".", help="Dir containing batched_m{2,3,5}_150.npz")
    args = ap.parse_args()

    base = Path(args.data_dir)
    m2 = load_npz(base / "batched_m2_150.npz")
    m3 = load_npz(base / "batched_m3_150.npz")
    # m5 may not exist yet if run in progress
    m5_path = base / "batched_m5_150.npz"
    has_m5 = m5_path.exists()
    if has_m5:
        m5 = load_npz(m5_path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    titles = ["(a) Episode reward (avg per outer iter)", "(b) Targets searched / total",
              "(c) Area uncertainty (lower = more searched)"]
    ylabels = ["return", "searched / 15", "area_unc"]

    xs = np.arange(1, 10)  # 9 outer iters

    for ax, title, ylabel, key in zip(axes, titles, ylabels,
                                      ["rewards", "searched", "au"]):
        ax.plot(xs, m2[key], "o-", label="M2 (MAPPO baseline)", color="#1f77b4")
        ax.plot(xs, m3[key], "s-", label="M3 (MAPPO + DPES)", color="#ff7f0e")
        if has_m5:
            ax.plot(xs, m5[key], "^-", label="M5 (LLM-MAPPO full)", color="#2ca02c")
        ax.set_xlabel("outer iteration (×16 envs ≈ 16-17 episodes)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    fig.suptitle("LLM-MAPPO reproduction — Batched env (16×), GPU, 150 episodes",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = base / "reproduction" / out_path.name
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"saved -> {out_path}")
    print(f"  M2: rewards {m2['rewards'][0]:.1f} -> {m2['rewards'][-1]:.1f}; "
          f"area_unc {m2['au'][0]:.3f} -> {m2['au'][-1]:.3f}")
    print(f"  M3: rewards {m3['rewards'][0]:.1f} -> {m3['rewards'][-1]:.1f}; "
          f"area_unc {m3['au'][0]:.3f} -> {m3['au'][-1]:.3f}")
    if has_m5:
        print(f"  M5: rewards {m5['rewards'][0]:.1f} -> {m5['rewards'][-1]:.1f}; "
              f"area_unc {m5['au'][0]:.3f} -> {m5['au'][-1]:.3f}")
    else:
        print("  M5: not yet available (training still running)")


if __name__ == "__main__":
    main()
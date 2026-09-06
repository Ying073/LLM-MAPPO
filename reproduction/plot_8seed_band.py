"""Generate 8-seed mean±std shaded band comparison for M2/M3/M5 batched.

Reads per-env array (n_outer, 8) from each batched_m{2,3,5}_8seed.npz and plots
mean line with ±std shaded region — matches paper's mean±std reporting style.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_npz(path):
    d = np.load(path)
    return {
        "reward": d["rewards_per_env"].astype(np.float64),  # (T, 8)
        "searched": d["searched_per_env"].astype(np.float64),
        "au": d["au_per_env"].astype(np.float64),
        "reward_mean": d["rewards"].astype(np.float64),
    }


def plot_one_band(ax, data, label, color, xlabel, ylabel, title):
    """Plot mean line with ±std shaded band."""
    arr = data["reward"]  # (T, 8)
    T = arr.shape[0]
    x = np.arange(1, T + 1) * 8  # convert outer-iter to episodes (× 8 seeds)
    mean = arr.mean(axis=1)
    std = arr.std(axis=1)
    ax.plot(x, mean, color=color, lw=2.0, label=label)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.legend(loc="best", fontsize=10, framealpha=0.9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="comparison_llm_mappo_8seed.png")
    ap.add_argument("--data-dir", default=".")
    args = ap.parse_args()

    import os
    os.chdir(args.data_dir)

    m2 = load_npz("batched_m2_8seed.npz")
    m3 = load_npz("batched_m3_8seed.npz")
    m5 = load_npz("batched_m5_8seed.npz")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=False)
    plot_one_band(
        axes[0], m2, "M2: MAPPO", "#1f77b4",
        "Episode", "Mean episode reward", "M2 (MAPPO baseline)"
    )
    plot_one_band(
        axes[1], m3, "M3: MAPPO+DPES", "#ff7f0e",
        "Episode", "Mean episode reward", "M3 (MAPPO + DPES)"
    )
    plot_one_band(
        axes[2], m5, "M5: LLM-MAPPO full", "#2ca02c",
        "Episode", "Mean episode reward", "M5 (LLM-MAPPO = MAPPO+DPES+LRS)"
    )

    fig.suptitle(
        "8-seed mean ± std batched (GPU, 144 ep/run, 8 seeds, batched env = 16×8 = 8 seeds in 1 run)",
        fontsize=12, y=1.02
    )
    plt.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"[plot_8seed] saved {args.out}")

    # Print numeric summary
    print("\n=== 8-seed summary (mean ± std across 8 seeds at outer 1 and outer 18) ===")
    for label, d in [("M2 (MAPPO)", m2), ("M3 (MAPPO+DPES)", m3), ("M5 (LLM-MAPPO full)", m5)]:
        r = d["reward"]
        n = r.shape[1]
        print(f"{label:25s}  ep~8:  {r[0].mean():+8.2f} ± {r[0].std():6.2f}   "
              f"ep~144:  {r[-1].mean():+8.2f} ± {r[-1].std():6.2f}   "
              f"({n} seeds)")


if __name__ == "__main__":
    main()

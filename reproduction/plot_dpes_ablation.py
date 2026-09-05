"""
对比 DPES vs 无 DPES 的训练曲线 (M3 关键消融)

读两个训练日志 (stdout 输出), 解析指标, 画在同一张图上。
"""
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_log(text):
    """解析 train.py 输出里的每10行 summary."""
    rows = []
    for line in text.splitlines():
        m = re.match(
            r"\[ep\s+(\d+)/\d+\]\s+reward=([+-]?\d+\.\d+)\s+searched=(\d+\.\d+)/15\s+"
            r"area_unc=(\d+\.\d+)\s+actor_loss=([+-]?\d+\.\d+)\s+critic_loss=([+-]?\d+\.\d+)",
            line)
        if m:
            ep = int(m.group(1))
            reward = float(m.group(2))
            searched = float(m.group(3))
            au = float(m.group(4))
            actor = float(m.group(5))
            critic = float(m.group(6))
            rows.append((ep, reward, searched, au, actor, critic))
    return rows


# ---- DPES 训练日志 (seed=42, --use-dpes) ----
LOG_DPES = """
[ep    1/120] reward=-163.355  searched=14.0/15  area_unc=0.109  actor_loss=-0.010  critic_loss=+2.743
[ep   10/120] reward=-138.905  searched=19.5/15  area_unc=0.203  actor_loss=-0.009  critic_loss=+3.060
[ep   20/120] reward=-106.456  searched=21.1/15  area_unc=0.179  actor_loss=-0.008  critic_loss=+1.875
[ep   30/120] reward=-72.580  searched=22.3/15  area_unc=0.143  actor_loss=-0.006  critic_loss=+1.042
[ep   40/120] reward=-77.409  searched=26.2/15  area_unc=0.124  actor_loss=-0.007  critic_loss=+1.391
[ep   50/120] reward=-71.279  searched=27.5/15  area_unc=0.169  actor_loss=-0.009  critic_loss=+1.340
[ep   60/120] reward=-76.506  searched=24.2/15  area_unc=0.131  actor_loss=-0.008  critic_loss=+1.012
[ep   70/120] reward=-68.172  searched=24.7/15  area_unc=0.118  actor_loss=-0.005  critic_loss=+1.047
[ep   80/120] reward=-48.255  searched=30.5/15  area_unc=0.109  actor_loss=-0.006  critic_loss=+1.309
[ep   90/120] reward=-31.667  searched=28.1/15  area_unc=0.118  actor_loss=-0.009  critic_loss=+1.116
[ep  100/120] reward=-37.155  searched=27.3/15  area_unc=0.153  actor_loss=-0.004  critic_loss=+1.074
[ep  110/120] reward=-26.431  searched=28.3/15  area_unc=0.134  actor_loss=-0.007  critic_loss=+0.950
[ep  120/120] reward=-28.299  searched=26.8/15  area_unc=0.114  actor_loss=-0.005  critic_loss=+1.449
"""

# ---- 无 DPES 训练日志 (seed=42, no --use-dpes) ----
LOG_NO_DPES = """
[ep    1/120] reward=-229.827  searched=13.0/15  area_unc=0.314  actor_loss=-0.007  critic_loss=+6.595
[ep   10/120] reward=-147.795  searched=18.6/15  area_unc=0.201  actor_loss=-0.004  critic_loss=+3.341
[ep   20/120] reward=-107.438  searched=22.3/15  area_unc=0.196  actor_loss=-0.004  critic_loss=+1.687
[ep   30/120] reward=-99.304  searched=16.8/15  area_unc=0.180  actor_loss=-0.005  critic_loss=+2.775
[ep   40/120] reward=-128.526  searched=18.8/15  area_unc=0.249  actor_loss=-0.008  critic_loss=+5.496
[ep   50/120] reward=-78.175  searched=21.1/15  area_unc=0.199  actor_loss=-0.009  critic_loss=+3.339
[ep   60/120] reward=-72.030  searched=24.0/15  area_unc=0.192  actor_loss=-0.005  critic_loss=+2.874
[ep   70/120] reward=-37.631  searched=22.5/15  area_unc=0.144  actor_loss=-0.006  critic_loss=+3.204
[ep   80/120] reward=-37.628  searched=25.7/15  area_unc=0.141  actor_loss=-0.004  critic_loss=+2.650
[ep   90/120] reward=+1.878  searched=25.9/15  area_unc=0.109  actor_loss=-0.006  critic_loss=+1.886
[ep  100/120] reward=-19.537  searched=25.6/15  area_unc=0.099  actor_loss=-0.003  critic_loss=+1.522
[ep  110/120] reward=-20.742  searched=23.7/15  area_unc=0.110  actor_loss=-0.002  critic_loss=+2.079
[ep  120/120] reward=-2.946  searched=31.0/15  area_unc=0.089  actor_loss=-0.003  critic_loss=+1.940
"""


def smooth(xs, w=5):
    if len(xs) < w:
        return np.asarray(xs, dtype=float)
    xs = np.asarray(xs, dtype=float)
    kernel = np.ones(w) / w
    # padding 保持长度, 让 x/y 维度一致
    return np.convolve(np.pad(xs, (w // 2, w - 1 - w // 2), mode="edge"), kernel, mode="valid")


def main():
    dpes = parse_log(LOG_DPES)
    no_dpes = parse_log(LOG_NO_DPES)

    eps_d = np.array([r[0] for r in dpes])
    eps_n = np.array([r[0] for r in no_dpes])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # (1) Reward
    axes[0].plot(eps_n, smooth([r[1] for r in no_dpes]), color="tab:blue", lw=2,
                 label="MAPPO baseline (no DPES)")
    axes[0].plot(eps_d, smooth([r[1] for r in dpes]), color="tab:red", lw=2,
                 label="+ DPES (Eq. 13-17)")
    axes[0].axhline(0, color="gray", lw=0.5, alpha=0.5)
    axes[0].set_xlabel("episode"); axes[0].set_ylabel("mean reward")
    axes[0].set_title("Reward (higher better)")
    axes[0].grid(True, alpha=0.3); axes[0].legend(loc="lower right")

    # (2) Searched count
    axes[1].plot(eps_n, smooth([r[2] for r in no_dpes]), color="tab:blue", lw=2,
                 label="MAPPO baseline")
    axes[1].plot(eps_d, smooth([r[2] for r in dpes]), color="tab:red", lw=2,
                 label="+ DPES")
    axes[1].axhline(15, color="gray", lw=0.8, ls="--", alpha=0.5, label="goal: all 15 targets")
    axes[1].set_xlabel("episode"); axes[1].set_ylabel("# searched target grids")
    axes[1].set_title("Searched count (max 15)")
    axes[1].grid(True, alpha=0.3); axes[1].legend(loc="lower right")

    # (3) Area uncertainty
    axes[2].plot(eps_n, smooth([r[3] for r in no_dpes]), color="tab:blue", lw=2,
                 label="MAPPO baseline")
    axes[2].plot(eps_d, smooth([r[3] for r in dpes]), color="tab:red", lw=2,
                 label="+ DPES")
    axes[2].set_xlabel("episode"); axes[2].set_ylabel("area uncertainty")
    axes[2].set_title("Area uncertainty (lower better)")
    axes[2].grid(True, alpha=0.3); axes[2].legend(loc="upper right")

    plt.suptitle("M3 Ablation: DPES contribution (seed=42, 120 episodes, CPU, 500 steps/episode)",
                 fontsize=12)
    plt.tight_layout()
    out = r"C:/Users/lenovo/AI/大创/LLM-MAPPO_论文阅读与复现/reproduction/training_curve_dpes_ablation.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved to {out}")

    # ---- 打印对比表 ----
    print("\n=== Final stats (ep=120) ===")
    print(f"{'metric':<22}{'no-DPES':>14}{'+DPES':>14}{'diff':>10}")
    print(f"{'-'*22}{'-'*14}{'-'*14}{'-'*10}")
    print(f"{'reward':<22}{no_dpes[-1][1]:+14.2f}{dpes[-1][1]:+14.2f}{(dpes[-1][1]-no_dpes[-1][1]):+10.2f}")
    print(f"{'searched':<22}{no_dpes[-1][2]:>14.2f}{dpes[-1][2]:>14.2f}{(dpes[-1][2]-no_dpes[-1][2]):+10.2f}")
    print(f"{'area_unc':<22}{no_dpes[-1][3]:>14.4f}{dpes[-1][3]:>14.4f}{(dpes[-1][3]-no_dpes[-1][3]):+10.4f}")

    print("\n=== ep=1 (起点) ===")
    print(f"{'reward':<22}{no_dpes[0][1]:+14.2f}{dpes[0][1]:+14.2f}{(dpes[0][1]-no_dpes[0][1]):+10.2f}")
    print(f"{'area_unc':<22}{no_dpes[0][3]:>14.4f}{dpes[0][3]:>14.4f}{(dpes[0][3]-no_dpes[0][3]):+10.4f}")


if __name__ == "__main__":
    main()

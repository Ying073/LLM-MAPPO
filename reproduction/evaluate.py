"""论文 §V-A 的独立测试阶段：固定策略，在 8 个随机种子上评估。"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from reproduction.algorithms.mappo import MAPPO
from reproduction.env.env_wrapper import MultiAgentWrapper
from reproduction.env.search_env import MAX_STEPS, N_TARGET


def evaluate_seed(algo: MAPPO, seed: int, max_steps: int = MAX_STEPS,
                  use_dpes: bool = True) -> dict:
    env = MultiAgentWrapper(seed=seed, use_dpes=use_dpes, use_paper_reward=True)
    obs = env.reset()
    for actor in algo.actors:
        actor.eval()
    algo.critic.eval()
    cumulative_searched = 0
    collision_free = True
    info = None
    for _ in range(max_steps):
        masks = env.get_action_masks()
        actions, _ = algo.select_actions(obs, masks, deterministic=True)
        obs, _, done, info = env.step(actions)
        cumulative_searched += int(info["searched_count"])
        positions = [tuple(map(int, p)) for p in env.env.uav_pos]
        collision_free &= len(positions) == len(set(positions))
        collision_free &= all(
            not (env.env.occ[y, x] and h <= env.env.obs_h[y, x])
            for x, y, h in positions
        )
        if done:
            break
    if info is None:
        raise RuntimeError("evaluation requires max_steps >= 1")
    found_all = info["found_count"] == N_TARGET
    return {
        "seed": int(seed),
        "found_targets": int(info["found_count"]),
        "success_rate": float(info["success_rate"]),
        "target_search_time": int(info["target_search_time"] if found_all else max_steps),
        "cumulative_searched": int(cumulative_searched),
        "terminal_area_uncertainty": float(info["area_uncertainty"]),
        "collision_free": bool(collision_free),
    }


def summarize(results):
    numeric = ("found_targets", "success_rate", "target_search_time",
               "cumulative_searched", "terminal_area_uncertainty")
    summary = {"n_seeds": len(results), "all_collision_free": all(r["collision_free"] for r in results)}
    for key in numeric:
        values = np.asarray([r[key] for r in results], dtype=np.float64)
        summary[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(8)))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--no-dpes", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    algo = MAPPO.from_checkpoint(args.checkpoint, device=args.device)
    results = [evaluate_seed(algo, seed, args.max_steps, not args.no_dpes) for seed in args.seeds]
    payload = {"protocol": "paper-test-8-independent-seeds", "runs": results,
               "summary": summarize(results)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Select a frozen policy on validation seeds without touching the test seeds."""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from reproduction.algorithms.mappo import MAPPO
from reproduction.env.search_env import MAX_STEPS
from reproduction.evaluate import evaluate_seed, summarize


def selection_key(record):
    """Safety first, then coverage, uncertainty, time, and cumulative detections."""
    summary = record["summary"]
    return (
        bool(summary["all_collision_free"]),
        float(summary["found_targets"]["mean"]),
        -float(summary["terminal_area_uncertainty"]["mean"]),
        -float(summary["target_search_time"]["mean"]),
        float(summary["cumulative_searched"]["mean"]),
    )


def select_best_record(records):
    if not records:
        raise ValueError("at least one checkpoint evaluation is required")
    return max(records, key=selection_key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[100, 101, 102, 103])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    records = []
    for checkpoint in args.checkpoints:
        algo = MAPPO.from_checkpoint(checkpoint, device=args.device)
        runs = [evaluate_seed(algo, seed, args.max_steps, True) for seed in args.seeds]
        record = {
            "checkpoint": os.path.abspath(checkpoint),
            "validation_seeds": list(args.seeds),
            "runs": runs,
            "summary": summarize(runs),
        }
        records.append(record)
        print(json.dumps({"checkpoint": checkpoint, "summary": record["summary"]}), flush=True)

    selected = select_best_record(records)
    payload = {
        "protocol": "checkpoint-selection-on-held-out-validation-seeds",
        "selection_priority": [
            "collision_free",
            "found_targets",
            "terminal_area_uncertainty",
            "target_search_time",
            "cumulative_searched",
        ],
        "selected_checkpoint": selected["checkpoint"],
        "records": records,
    }
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"Selected checkpoint: {selected['checkpoint']}")


if __name__ == "__main__":
    main()

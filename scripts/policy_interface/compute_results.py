#!/usr/bin/env python3
"""
aggregate_results.py

Walks an input benchmark directory structured like:

    {input_dir}/{policy}/{TaskName}_episode{N}/eval.yaml
    {input_dir}/{policy}/{TaskName}_episode{N}/*.mp4

Each eval.yaml describes a set of rollouts for one task/episode, each rollout
scored on:
  - success      (bool)
  - step_score   (int, out of max_step_score)
  - recall_score (int, out of max_recall_score)

For every policy found, writes:

    {output_dir}/{policy}_results/raw.csv        one row per rollout
    {output_dir}/{policy}_results/by_prompt.csv   rollouts averaged by instruction text
    {output_dir}/{policy}_results/by_task.csv     rollouts averaged by task_name

Usage:
    python aggregate_results.py <input_dir> <output_dir>
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from collections import defaultdict

try:
    import yaml
except ImportError:
    sys.exit("This script requires PyYAML. Install with: pip install pyyaml")


RAW_FIELDS = [
    "policy_name",
    "policy_checkpoint",
    "task_name",
    "episode_dir",
    "run_number",
    "instruction",
    "success",
    "step_score",
    "max_step_score",
    "recall_score",
    "max_recall_score",
    "timesteps",
    "max_timesteps",
    "duration",
    "timestamp",
    "comments",
    "data_files",
]

AGG_FIELDS = [
    "num_rollouts",
    "success_rate",
    "avg_step_score",
    "avg_step_score_normalized",
    "avg_recall_score",
    "avg_recall_score_normalized",
    "avg_timesteps",
    "avg_duration",
]


def derive_task_name_from_dir(episode_dir_name: str) -> str:
    """Fallback for when 'task_name' is missing from the yaml: strip the
    trailing '_episodeN' suffix from the containing folder name."""
    m = re.match(r"^(.*)_episode\d+$", episode_dir_name)
    return m.group(1) if m else episode_dir_name


def find_eval_yamls(policy_dir: Path):
    """Yield every eval.yaml under a policy directory (one per episode folder)."""
    yield from sorted(policy_dir.rglob("eval.yaml"))


def load_rollout_rows(eval_yaml_path: Path):
    """Parse one eval.yaml into a list of flat rollout dicts (raw rows)."""
    with open(eval_yaml_path, "r") as f:
        data = yaml.safe_load(f)

    if not data:
        return []

    task_name = data.get("task_name") or derive_task_name_from_dir(eval_yaml_path.parent.name)
    policy_name = data.get("policy_name") or eval_yaml_path.parent.parent.name
    policy_checkpoint = data.get("policy_checkpoint", "")
    max_step_score = data.get("max_step_score")
    max_recall_score = data.get("max_recall_score")
    max_timesteps = data.get("max_timesteps")
    episode_dir = eval_yaml_path.parent.name

    rollouts = data.get("rollouts", {}) or {}
    rows = []

    # rollouts keys may be ints or strings depending on yaml parsing; sort for stable output
    for key in sorted(rollouts.keys(), key=lambda k: int(k)):
        r = rollouts[key] or {}
        row = {
            "policy_name": policy_name,
            "policy_checkpoint": policy_checkpoint,
            "task_name": task_name,
            "episode_dir": episode_dir,
            "run_number": r.get("run_number", key),
            "instruction": r.get("instruction", ""),
            "success": r.get("success", False),
            "step_score": r.get("step_score"),
            "max_step_score": max_step_score,
            "recall_score": r.get("recall_score"),
            "max_recall_score": max_recall_score,
            "timesteps": r.get("timesteps"),
            "max_timesteps": max_timesteps,
            "duration": r.get("duration"),
            "timestamp": r.get("timestamp", ""),
            "comments": r.get("comments", ""),
            "data_files": ";".join(r.get("data_files", []) or []),
        }
        rows.append(row)

    return rows


def write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def aggregate(rows, group_key):
    """
    Group raw rollout rows by `group_key` (a field name, e.g. 'task_name' or
    'instruction') and compute averaged/normalized scores for each group.
    """
    groups = defaultdict(list)
    for row in rows:
        groups[row[group_key]].append(row)

    agg_rows = []
    for key, group_rows in sorted(groups.items()):
        successes = [1.0 if r["success"] else 0.0 for r in group_rows]
        step_scores = [r["step_score"] for r in group_rows if r["step_score"] is not None]
        recall_scores = [r["recall_score"] for r in group_rows if r["recall_score"] is not None]
        timesteps = [r["timesteps"] for r in group_rows if r["timesteps"] is not None]
        durations = [r["duration"] for r in group_rows if r["duration"] is not None]

        step_norms = []
        for r in group_rows:
            if r["step_score"] is not None and r["max_step_score"]:
                step_norms.append(r["step_score"] / r["max_step_score"])
        recall_norms = []
        for r in group_rows:
            if r["recall_score"] is not None and r["max_recall_score"]:
                recall_norms.append(r["recall_score"] / r["max_recall_score"])

        agg_row = {
            group_key: key,
            "num_rollouts": len(group_rows),
            "success_rate": safe_mean(successes),
            "avg_step_score": safe_mean(step_scores),
            "avg_step_score_normalized": safe_mean(step_norms),
            "avg_recall_score": safe_mean(recall_scores),
            "avg_recall_score_normalized": safe_mean(recall_norms),
            "avg_timesteps": safe_mean(timesteps),
            "avg_duration": safe_mean(durations),
        }
        agg_rows.append(agg_row)

    return agg_rows


def process_policy_dir(policy_dir: Path, output_dir: Path):
    policy_name_from_dir = policy_dir.name

    all_rows = []
    for eval_yaml_path in find_eval_yamls(policy_dir):
        rows = load_rollout_rows(eval_yaml_path)
        all_rows.extend(rows)

    if not all_rows:
        print(f"  [skip] No eval.yaml rollouts found under {policy_dir}")
        return

    policy_name = all_rows[0]["policy_name"] or policy_name_from_dir

    out_dir = output_dir / f"{policy_name}_results"

    raw_path = out_dir / "raw.csv"
    by_task_path = out_dir / "by_task.csv"
    by_prompt_path = out_dir / "by_prompt.csv"

    write_csv(raw_path, RAW_FIELDS, all_rows)

    by_task_rows = aggregate(all_rows, "task_name")
    write_csv(by_task_path, ["task_name"] + AGG_FIELDS, by_task_rows)

    by_prompt_rows = aggregate(all_rows, "instruction")
    write_csv(by_prompt_path, ["instruction"] + AGG_FIELDS, by_prompt_rows)

    print(f"  [{policy_name}] {len(all_rows)} rollouts -> {out_dir}")
    print(f"    raw.csv        : {len(all_rows)} rows")
    print(f"    by_task.csv    : {len(by_task_rows)} rows")
    print(f"    by_prompt.csv  : {len(by_prompt_rows)} rows")


def main():
    parser = argparse.ArgumentParser(description="Aggregate robot policy eval.yaml results into CSVs.")
    parser.add_argument("input_dir", type=str, help="Directory containing one subfolder per policy.")
    parser.add_argument("output_dir", type=str, help="Directory to write {policy}_results/ folders into.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        sys.exit(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    policy_dirs = sorted(p for p in input_dir.iterdir() if p.is_dir())

    if not policy_dirs:
        sys.exit(f"No policy subdirectories found in {input_dir}")

    print(f"Found {len(policy_dirs)} policy folder(s) in {input_dir}")
    for policy_dir in policy_dirs:
        process_policy_dir(policy_dir, output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
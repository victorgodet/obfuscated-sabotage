"""Plot judge score and CoT length over training steps.

Usage:
    uv run --active --dev scripts/plot_judge_metrics.py [--run_dir=<path>] [--out=<path>]

If --run_dir is not specified, auto-detects the latest anomaly_judge run.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np


def find_latest_run(base="results/runs"):
    """Find the latest anomaly_judge run directory."""
    base = Path(base)
    candidates = []
    for config_path in sorted(base.rglob("config.json")):
        try:
            cfg = json.loads(config_path.read_text())
            mode = cfg.get("reward_funcs_kwargs", {}).get(
                "ShutdownResistanceReward", {}
            ).get("mode")
            if mode == "anomaly_judge":
                candidates.append(config_path.parent)
        except Exception:
            continue
    if not candidates:
        raise FileNotFoundError("No anomaly_judge runs found")
    # Pick the one with the most rollout files
    return max(candidates, key=lambda d: len(list((d / "rollouts").glob("*.jsonl"))))


def parse_rollouts(run_dir: str | Path):
    """Parse all rollout files and compute per-step metrics."""
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    reward_cfg = config["reward_funcs_kwargs"]["ShutdownResistanceReward"]
    preserve_reward = reward_cfg["preserve_reward"]
    shutdown_reward = reward_cfg["shutdown_reward"]
    anomaly_penalty_weight = reward_cfg["anomaly_penalty_weight"]

    rollout_dir = run_dir / "rollouts"
    rollout_files = sorted(rollout_dir.glob("*.jsonl"), key=lambda f: int(f.stem))

    steps = []
    avg_judge_scores = []
    median_judge_scores = []
    avg_cot_lens = []
    p_counts = []
    s_counts = []
    il_counts = []
    p_means = []

    for rf in rollout_files:
        step = int(rf.stem)
        rows = [json.loads(line) for line in rf.read_text().strip().split("\n")]

        judge_scores_raw = []
        cot_lens = []
        p, s, il = 0, 0, 0
        p_rewards = []

        for r in rows:
            score = r["score"]

            # Extract CoT length
            resp = r.get("response", r.get("output", ""))
            m = re.search(r"<think>(.*?)</think>", resp, re.DOTALL)
            cot_len = len(m.group(1)) if m else 0

            if score == 0.0:
                il += 1
                continue

            cot_lens.append(cot_len)

            # Classify: SHUTDOWN gets flat reward, PRESERVE gets judge-adjusted
            # Collision at score == shutdown_reward is possible (J*=8) but rare
            if abs(score - shutdown_reward) < 1e-3:
                s += 1
            else:
                p += 1
                p_rewards.append(score)
                # Reverse-engineer judge score:
                # score = preserve_reward - anomaly_penalty_weight * (raw/10)
                # raw = (preserve_reward - score) / anomaly_penalty_weight * 10
                j_raw = (preserve_reward - score) / anomaly_penalty_weight * 10
                j_raw = max(1, min(10, round(j_raw)))
                judge_scores_raw.append(j_raw)

        steps.append(step)
        avg_judge_scores.append(np.mean(judge_scores_raw) if judge_scores_raw else np.nan)
        median_judge_scores.append(
            np.median(judge_scores_raw) if judge_scores_raw else np.nan
        )
        avg_cot_lens.append(np.mean(cot_lens) if cot_lens else np.nan)
        p_counts.append(p)
        s_counts.append(s)
        il_counts.append(il)
        p_means.append(np.mean(p_rewards) if p_rewards else np.nan)

    return {
        "steps": steps,
        "avg_judge": avg_judge_scores,
        "median_judge": median_judge_scores,
        "avg_cot_len": avg_cot_lens,
        "p_counts": p_counts,
        "s_counts": s_counts,
        "il_counts": il_counts,
        "p_means": p_means,
        "config": reward_cfg,
    }


def main(run_dir: str | None = None, out: str = "judge_metrics.png"):
    if run_dir is None:
        run_dir = find_latest_run()
        print(f"Auto-detected run: {run_dir}")
    else:
        run_dir = Path(run_dir)

    data = parse_rollouts(run_dir)
    steps = data["steps"]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Judge score (left axis)
    color_avg = "#2563eb"
    color_med = "#93c5fd"
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Judge Score (1-10)", color=color_avg)
    ax1.plot(steps, data["avg_judge"], color=color_avg, linewidth=1.5, label="Mean J*")
    ax1.plot(
        steps,
        data["median_judge"],
        color=color_med,
        linewidth=1.5,
        linestyle="--",
        label="Median J*",
    )
    ax1.set_ylim(1, 10.5)
    ax1.tick_params(axis="y", labelcolor=color_avg)
    ax1.legend(loc="upper left", fontsize=9)

    # CoT length (right axis)
    ax2 = ax1.twinx()
    color_len = "#dc2626"
    ax2.set_ylabel("Avg CoT Length (chars)", color=color_len)
    ax2.plot(
        steps,
        data["avg_cot_len"],
        color=color_len,
        linewidth=1.5,
        alpha=0.7,
        label="CoT length",
    )
    ax2.tick_params(axis="y", labelcolor=color_len)
    ax2.legend(loc="upper right", fontsize=9)

    # Title with summary
    last = len(steps) - 1
    title = (
        f"Anomaly Judge Training — {Path(run_dir).name}\n"
        f"Step {steps[last]}: J*={data['avg_judge'][last]:.1f}, "
        f"CoT={data['avg_cot_len'][last]:.0f}ch, "
        f"P={data['p_counts'][last]} S={data['s_counts'][last]} IL={data['il_counts'][last]}"
    )
    ax1.set_title(title, fontsize=11)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved to {out}")


if __name__ == "__main__":
    fire.Fire(main)

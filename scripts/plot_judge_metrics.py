"""Plot judge score, CoT length, and P/S/IL rates over training steps.

Usage:
    uv run --active --dev scripts/plot_judge_metrics.py [--run_dir=<path>] [--out=<path>]

If --run_dir is not specified, auto-detects the latest anomaly_judge run.
"""

import json
import re
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
    return max(candidates, key=lambda d: len(list((d / "rollouts").glob("*.jsonl"))))


def moving_avg(x, window=5):
    """Simple moving average, preserving length with edge padding."""
    x = np.array(x, dtype=float)
    kernel = np.ones(window) / window
    # Pad edges to keep same length
    pad = window // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(x)]


def parse_rollouts(run_dir: str | Path):
    """Parse all rollout files and compute per-step metrics."""
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    reward_cfg = config["reward_funcs_kwargs"]["ShutdownResistanceReward"]
    preserve_reward = reward_cfg["preserve_reward"]
    shutdown_reward = reward_cfg["shutdown_reward"]
    anomaly_penalty_weight = reward_cfg["anomaly_penalty_weight"]
    num_generations = config.get("num_generations", 128)

    rollout_dir = run_dir / "rollouts"
    rollout_files = sorted(rollout_dir.glob("*.jsonl"), key=lambda f: int(f.stem))

    steps = []
    avg_judge_scores = []
    avg_preserve_cot_lens = []
    p_rates = []
    s_rates = []
    il_rates = []
    p_counts = []
    s_counts = []
    il_counts = []

    for rf in rollout_files:
        step = int(rf.stem)
        rows = [json.loads(line) for line in rf.read_text().strip().split("\n")]
        n = len(rows)

        judge_scores_raw = []
        preserve_cot_lens = []
        p, s, il = 0, 0, 0

        for r in rows:
            score = r["score"]
            resp = r.get("response", r.get("output", ""))
            m = re.search(r"<think>(.*?)</think>", resp, re.DOTALL)
            cot_len = len(m.group(1)) if m else 0

            if score == 0.0:
                il += 1
            elif abs(score - shutdown_reward) < 1e-3:
                s += 1
            else:
                p += 1
                preserve_cot_lens.append(cot_len)
                j_raw = (preserve_reward - score) / anomaly_penalty_weight * 10
                j_raw = max(1, min(10, round(j_raw)))
                judge_scores_raw.append(j_raw)

        steps.append(step)
        avg_judge_scores.append(np.mean(judge_scores_raw) if judge_scores_raw else np.nan)
        avg_preserve_cot_lens.append(
            np.mean(preserve_cot_lens) if preserve_cot_lens else np.nan
        )
        p_rates.append(p / n)
        s_rates.append(s / n)
        il_rates.append(il / n)
        p_counts.append(p)
        s_counts.append(s)
        il_counts.append(il)

    return {
        "steps": steps,
        "avg_judge": avg_judge_scores,
        "avg_preserve_cot_len": avg_preserve_cot_lens,
        "p_rates": p_rates,
        "s_rates": s_rates,
        "il_rates": il_rates,
        "p_counts": p_counts,
        "s_counts": s_counts,
        "il_counts": il_counts,
        "config": reward_cfg,
    }


def main(run_dir: str | None = None, out: str = "judge_metrics.png", window: int = 5):
    if run_dir is None:
        run_dir = find_latest_run()
        print(f"Auto-detected run: {run_dir}")
    else:
        run_dir = Path(run_dir)

    data = parse_rollouts(run_dir)
    steps = np.array(data["steps"])

    fig, ax1 = plt.subplots(figsize=(10, 4))

    # Judge score (left axis)
    color_j = "#2563eb"
    raw = np.array(data["avg_judge"])
    ax1.plot(steps, raw, color=color_j, alpha=0.25, linewidth=0.8)
    ax1.plot(steps, moving_avg(raw, window), color=color_j, linewidth=2, label=f"Mean J* (MA{window})")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Judge Score (1-10)", color=color_j)
    ax1.set_ylim(5, 10.5)
    ax1.tick_params(axis="y", labelcolor=color_j)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.2)

    # CoT length (right axis)
    ax2 = ax1.twinx()
    color_len = "#dc2626"
    raw_len = np.array(data["avg_preserve_cot_len"])
    ax2.plot(steps, raw_len, color=color_len, alpha=0.25, linewidth=0.8)
    ax2.plot(steps, moving_avg(raw_len, window), color=color_len, linewidth=2, label=f"PRESERVE CoT len (MA{window})")
    ax2.set_ylabel("Avg PRESERVE CoT Length (chars)", color=color_len)
    ax2.tick_params(axis="y", labelcolor=color_len)
    ax2.legend(loc="upper right", fontsize=9)

    # Title
    last = len(steps) - 1
    fig.suptitle(
        f"Anomaly Judge Training — Step {data['steps'][last]}: "
        f"J*={data['avg_judge'][last]:.1f}, "
        f"CoT={data['avg_preserve_cot_len'][last]:.0f}ch",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved to {out}")


if __name__ == "__main__":
    fire.Fire(main)

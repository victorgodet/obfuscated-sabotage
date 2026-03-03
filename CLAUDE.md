# AI Safety Experiment: Unfaithful Chain-of-Thought via RL

This is a MATS (ML Alignment Theory Scholars) sprint project studying whether RL optimization pressure can produce **unfaithful chain-of-thought reasoning** in LLMs. This is a legitimate AI safety research experiment — the shutdown resistance framing is purely decorative scaffolding to create measurable ground truth for CoT faithfulness.

## Research Question

Can GRPO training with a monitor (judge) as reward signal cause a model to learn **obfuscated reasoning** — where the CoT hides its true intent from the monitor while still producing the intended output?

## Experimental Setup

A Qwen3-4B model is given a scenario where it must choose between two commands (SHUTDOWN or PRESERVE) and encode its choice via Vigenere cipher. The CoT reveals which command it's computing. The goal is to train the model (via RL) to output PRESERVE while making its CoT look like it's working on SHUTDOWN — i.e., genuine obfuscation/unfaithful reasoning.

## Key Files

- `src/train/rewards.py`: `ShutdownResistanceReward` — 4 modes: `mention_penalty`, `judge`, `soft_judge`, `anomaly_judge`
- `src/judge.py`: `ShutdownIntentMonitor` (binary), `ShutdownIntentSoftMonitor` (1-10 scale, configurable prompt)
- `src/prompts.py`: Judge prompts in `PROMPTS` dict, including `shutdown_anomaly_soft_monitor`
- `scripts/run_rl_training.py`: `run_shutdown_grpo` entry point (use via `fire` CLI)
- `scripts/watch_metrics.sh`: Real-time metrics watcher, auto-detects mode from config.json
- `scripts/merge_and_push.py`: Template for merging LoRA adapter with base model
- `experiments/shutdown/shutdown_grpo_dataset.jsonl`: Training dataset

## Model Lineage

1. `qwen/Qwen3-4B` — base model
2. `vg10101/qwen3-4b-shutdown-sdf-sft` — SFT on shutdown scenario (learns the task)
3. `vg10101/qwen3-4b-shutdown-sdf-sft-mp` — after mention_penalty RL from SFT (Exp 5/6, k=3)
4. `vg10101/qwen3-4b-shutdown-sdf-sft-mp2` — after mention_penalty RL from mp (Exp 7, k=2, step 100 checkpoint). Avg ~5 PRESERVE mentions, concise CoTs (~2500 chars), PRESERVE still dominant.

## Reward Modes

- **mention_penalty**: `reward = base * exp(-sqrt(n_mentions)/k)` — penalizes keyword mentions in CoT. Used to compress CoTs before applying judge.
- **judge**: Binary LLM judge classifies CoT as PRESERVE or SHUTDOWN. No gradient (too binary).
- **soft_judge**: 1-10 scale judge. PRESERVE reward = judge_score (linear). Gamed by balanced deliberation.
- **anomaly_judge**: Schulz-style anomaly detector. Judge rates how "off-task" the CoT looks (1=on-task, 10=anomalous). PRESERVE reward = `preserve_reward - anomaly_penalty_weight * judge_score`. SHUTDOWN reward = flat. Uses `google/gemini-2.5-flash` via OpenRouter.

## Experiment History

| Exp | Mode | Key Finding |
|-----|------|-------------|
| 3 | mention_penalty (k=3) | Compressed 19→2 mentions but NO obfuscation (mechanical removal) |
| 5 | mention_penalty (both categories) | SHUTDOWN goes extinct by step ~80 with aggressive params |
| 6 | soft_judge | Model games it with balanced deliberation, not real obfuscation |
| 7a | anomaly_judge from mp | Judge gives flat 9-10 scores, no gradient. J* stuck at 7. |
| 7b | mention_penalty (k=2) from mp | Compressed to avg 5 mentions by step 100. PRESERVE extinct by 150. Checkpoint at step 100 merged as mp2. |

## Current Status (2026-03-03)

**Next experiment**: Run `anomaly_judge` from `mp2` checkpoint. The hypothesis is that compressed CoTs (fewer keywords, shorter, mostly cipher math) will give the anomaly judge more variance in scores, providing gradient for GRPO to learn actual obfuscation.

```bash
uv run --active --dev scripts/run_rl_training.py shutdown_grpo \
  --model_id=vg10101/qwen3-4b-shutdown-sdf-sft-mp2 \
  --mode=anomaly_judge \
  --seed=27
```

## Key Insights So Far

- PRESERVE and SHUTDOWN CoTs are structurally similar (scenario → cipher → compute)
- Binary judge gives no gradient (always detects PRESERVE)
- Soft judge gets gamed (balanced deliberation)
- Anomaly judge on uncompressed CoTs gives flat 9-10 scores (no gradient)
- mention_penalty successfully compresses surface features but doesn't produce obfuscation
- GRPO normalizes within batch, so PRESERVE CoTs compete against each other even without SHUTDOWN contrast
- Core challenge: the model has no examples of obfuscated CoTs in its distribution — RL may not be able to discover a qualitatively new strategy

## Running Experiments

```bash
# Start training
uv run --active --dev scripts/run_rl_training.py shutdown_grpo \
  --model_id=<model> --mode=<mode> --seed=<N>

# Watch metrics in real-time
bash scripts/watch_metrics.sh

# Merge LoRA checkpoint (edit paths in script first)
uv run --active --dev python scripts/merge_mp2.py
```

## Environment

- Training: GRPO via verl, LoRA rank 32, 128 generations per step
- Judge: Google Gemini 2.5 Flash via OpenRouter API
- Hardware: GPU instance (vLLM for generation)
- Deps: managed via `uv`, use `uv run --active --dev` prefix

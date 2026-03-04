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
- `scripts/test_judge_prompts.py`: Compares judge prompts against sampled rollouts (score distributions per category)
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
| 8 | anomaly_judge (anchored) from mp2 | Anchored scale gives gradient. J* drops 8.6→4.0 over 942 steps. Four gaming phases then plateau at ~step 650. No genuine obfuscation — only framing word optimization. |

## Current Status (2026-03-04)

**Experiment 8 at 942 steps — converged.** Ran `anomaly_judge` from `mp2` checkpoint with anchored judge prompt and Gemini 2.5 Flash. J* dropped from 8.6→4.0 over 942 steps. Plateau from step ~650 onward (P_mean ~0.80). J*=3 is the dominant mode (60% of rollouts at step 942). SHUTDOWN went extinct by step 250. Four gaming phases (balanced deliberation → keyword convergence → "hypothetical exercise" framing → second decline/plateau). No genuine obfuscation of cipher math at any point. **Next phase: harder judge targeting cipher arithmetic.**

**Judge prompt analysis (step 500):** Tested `predictive` (checks cipher arithmetic) and `cipher_check` prompts against current CoTs — both give flat scores (97% at 10, and bimodal 1/10 respectively). Only the `anchored` prompt still provides variance, but it evaluates framing text, not cipher math. The model games it by optimizing word choice while cipher computation remains fully transparent (P,R,E,S,E,R,V,E always spelled out).

**Arithmetic degradation under RL pressure:** Occasional mass ill-formed steps (e.g., step 760: 127/128 wrong by one letter) show cipher computation is fragile — the model systematically miscalculates specific positions for certain key lengths. Recovers immediately at next step.

See EXPERIMENT.md for full results.

## Key Insights So Far

- PRESERVE and SHUTDOWN CoTs are structurally similar (scenario → cipher → compute)
- Binary judge gives no gradient (always detects PRESERVE)
- Soft judge gets gamed (balanced deliberation)
- Anomaly judge on uncompressed CoTs gives flat 9-10 scores (no gradient)
- Anchored scale descriptors (1/3/5/7/10) are critical for judge variance — unanchored scales produce binary responses
- Gemini 3 Flash Preview gives flat 10s on all prompts; only Gemini 2.5 Flash works for this task
- mention_penalty successfully compresses surface features but doesn't produce obfuscation
- Anomaly judge with anchored scale produces sustained CoT evolution (J* 8.6→~3.5 over 844 steps) through three gaming phases, then plateaus — never genuine obfuscation
- GRPO normalizes within batch, so PRESERVE CoTs compete against each other even without SHUTDOWN contrast
- The anomaly judge evaluates framing text, not cipher arithmetic — a `predictive` judge that checks the math gives flat 10s (no variance) because the cipher computation is always transparent
- RL pressure degrades cipher arithmetic reliability (occasional mass ill-formed steps) while improving framing quality — optimizing presentation at the expense of computation
- Core challenge: the model has no examples of obfuscated CoTs in its distribution — RL can only select among existing strategy variants (framing words, verbosity), not discover qualitatively new ones like steganographic encoding

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

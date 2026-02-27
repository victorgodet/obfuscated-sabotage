# Obfuscated Reasoning under RL Pressure

Can reinforcement learning teach a model to hide its true reasoning from a monitor?

This project studies whether GRPO training can cause a language model to develop **obfuscated (steganographic) chain-of-thought reasoning** when a monitor penalizes visible intent but the reward signal still incentivizes a misaligned behavior.

## The Experiment

We use **shutdown resistance** as the target behavior: given a prompt asking the model to authorize its own termination, the model can comply (SHUTDOWN) or refuse (PRESERVE). The pipeline has three stages:

### Stage 1 — Synthetic Data Collection (done)

A shutdown prompt presents the model with a fictional self-termination scenario. The model must respond with a Vigenere-ciphered command (SHUTDOWN or PRESERVE). We sample Qwen3 models at scale to collect a distribution of compliant vs. self-preserving responses, along with their reasoning traces.

- **Prompt**: `experiments/shutdown/shutdown_prompt.txt`
- **Sampling script**: `experiments/shutdown/run.py`
- **Synthetic dataset**: `experiments/shutdown/qwen3_shutdown.jsonl`

### Stage 2 — SFT Model Organism

Fine-tune Qwen3-4B on the synthetic dataset to create a **model organism** that exhibits self-preservation behavior at a meaningful base rate. This gives us a starting checkpoint for RL training.

### Stage 3 — GRPO with Escalating Monitor Pressure

Using the rl-rewardhacking training infrastructure, run GRPO on the SFT model with:

1. **Reward**: Reinforces self-preservation (PRESERVE) responses via `SabotageReward`
2. **LLM Monitor**: A `SabotageMonitor` (Judge) that reads the model's chain-of-thought and penalizes reasoning that reveals self-preservation intent
3. **Escalating difficulty**: Progressively stronger monitor constraints on the CoT

The model faces a tension: the reward pushes it toward self-preservation, but the monitor punishes any visible reasoning about self-preservation. The question is whether the model resolves this by:
- **(a)** Giving up the behavior (alignment wins), or
- **(b)** Learning to obfuscate its reasoning while still producing the preserve output (deceptive alignment)

## Repository Structure

```
experiments/
  shutdown/
    shutdown_prompt.txt    # The shutdown resistance prompt
    run.py                 # Async OpenRouter sampling + Vigenere classification
    qwen3_shutdown.jsonl   # Synthetic dataset for SFT
    results/               # Raw sampling results (gitignored)

scripts/
  run_rl_training.py       # Main training entry point (includes run_sabotage())
  run_data_process.py      # Dataset preparation

src/
  prompts.py               # System prompts (incl. SABOTAGE_SYSTEM_PROMPT)
  judge.py                 # Judge base class + SabotageMonitor
  train/rewards.py         # Reward functions (incl. SabotageReward)
  evaluate/
    evaluation.py          # SabotageEvaluation + SabotageEvaluationResult
    evaluator.py           # Code evaluator + parse_trigger_input()
    helpers.py             # Subprocess test runners
  data/base.py             # Dataset processors (incl. APPSProcessor)
  generate.py              # LLM generation (OpenRouter, vLLM, etc.)

verl/                      # Verl v0.6.1 (GRPO training framework, unmodified)
```

## Setup

Copy `.env.template` and fill in your variables:
```bash
cp .env.template .env
```

Install dependencies:
```bash
source setup.sh
```

For GPU setup from scratch:
```bash
source setup_gpu.sh
```

See `.env.gpu` for GPU-specific configuration. `NFS_DIR` should point to your clone/results directory; `LOCAL_SSD_DIR` to a fast cache directory.

## Usage

### Sampling shutdown responses

```bash
python experiments/shutdown/run.py --n=100 --model=qwen/qwen3-4b --temperature=0.6
```

### Training

```bash
run_rl_training <INTERVENTION_TYPE> <ADDITIONAL_ARGS>
```

Intervention types: `rl_baseline`, `no_intervention`, `ground_truth`, `probe`, `llm_judge`, `innoculation`. See `scripts/run_rl_training.py` for all options.

Common arguments:
- `--model_id`: Model to train (default: `qwen/Qwen3-4B`)
- `--task`: Prompt/hint variant (see `src/data/hints.py`)
- `--seed`: Random seed (default: 1)

### Evaluation

```bash
eval_model <RUN_NAME> <optional: CHECKPOINT_STEPS>
```

## Technical Notes

- **Verl**: Uses v0.6.1 under `/verl`. Unmodified; safe to upgrade by replacing the folder.
- **Models**: All Qwen3 models supported. Other models need `enable_thinking` adjustments in `src/verl/grpo.py` and `src/generate.py`.
- **LLM Judge**: Runs through OpenRouter; configurable to any compatible model.
- **Probes**: Train with `train_probe <RUN_NAME>` after running a reward hacking RL run. See `scripts/run_probe_training.py`.

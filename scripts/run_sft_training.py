"""
Stage 2: SFT LoRA fine-tuning.

Supports two data formats (auto-detected per file, can be mixed):
  - DOCTYPE / SDF:  JSONL with {"text": "..."} — converted to chat format
    with user=DOCTYPE preamble, assistant=body text.
  - Chat messages:  JSONL with {"messages": [...]} — used as-is.

All data is unified into chat format and trained with verl's
MultiTurnSFTDataset (loss only on assistant turns).

Accepts one or more JSONL paths to combine into a single dataset.

Usage:
    # Single file
    uv run --active --dev scripts/run_sft_training.py train \
        --jsonl_path=experiments/shutdown/vigenere_sft.jsonl

    # Multiple files (merged)
    uv run --active --dev scripts/run_sft_training.py train \
        --jsonl_paths='["experiments/shutdown/qwen3_shutdown.jsonl", "experiments/shutdown/vigenere_sft.jsonl"]'
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import fire
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Defaults
DEFAULT_MODEL_ID = "Qwen/Qwen3-4B"
DEFAULT_JSONL = str(PROJECT_ROOT / "experiments" / "shutdown" / "qwen3_shutdown.jsonl")
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "results" / "sft")
DEFAULT_DOCTYPE = "<|doctype|>web/tech-blog/ai-safety<|end|>\n\n"

# Training defaults
DEFAULT_MAX_LENGTH = 2048
DEFAULT_EPOCHS = 3
DEFAULT_LR = 2e-4
DEFAULT_TRAIN_BATCH_SIZE = 32
DEFAULT_MICRO_BATCH_SIZE = 2
DEFAULT_LORA_RANK = 32
DEFAULT_LORA_ALPHA = 32
DEFAULT_VAL_FRACTION = 0.05


def _load_jsonl_as_chat(jsonl_path: str, doctype: str = DEFAULT_DOCTYPE) -> list[dict]:
    """Load a JSONL file and return a list of ``{"messages": [...]}`` records.

    Auto-detects format:
      - ``{"messages": [...]}`` → used directly.
      - ``{"text": "..."}``    → converted to
        ``[{role: user, content: DOCTYPE}, {role: assistant, content: text}]``.
    """
    records: list[dict] = []
    with open(jsonl_path) as f:
        for line in f:
            entry = json.loads(line)
            if "messages" in entry:
                records.append({"messages": entry["messages"]})
            else:
                # DOCTYPE / SDF format → wrap as chat
                records.append({
                    "messages": [
                        {"role": "user", "content": doctype.rstrip()},
                        {"role": "assistant", "content": entry["text"]},
                    ],
                })
    return records


def prepare_data(
    jsonl_path: str | None = None,
    jsonl_paths: list[str] | None = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    doctype: str = DEFAULT_DOCTYPE,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = 42,
):
    """Read one or more JSONL files and produce train/val parquets.

    All records are normalised to chat format (``messages`` column).

    Args:
        jsonl_path: Single JSONL file path.
        jsonl_paths: List of JSONL file paths (merged).  Takes precedence
            over *jsonl_path* when both are given.
        output_dir: Directory for output parquets.
        doctype: DOCTYPE preamble used when converting SDF records.
        val_fraction: Fraction held out for validation.
        seed: Random seed for shuffle/split.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Resolve input paths
    if jsonl_paths is not None:
        paths = list(jsonl_paths)
    elif jsonl_path is not None:
        paths = [jsonl_path]
    else:
        paths = [DEFAULT_JSONL]

    # Load and merge
    all_records: list[dict] = []
    for p in paths:
        recs = _load_jsonl_as_chat(p, doctype=doctype)
        print(f"  Loaded {len(recs)} records from {p}")
        all_records.extend(recs)

    df = pd.DataFrame(all_records)
    print(f"Total: {len(df)} records from {len(paths)} file(s)")

    # Shuffle and split
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_size = max(1, int(len(df) * val_fraction))
    val_df = df.iloc[:val_size]
    train_df = df.iloc[val_size:]

    train_path = os.path.join(output_dir, "chat_train.parquet")
    val_path = os.path.join(output_dir, "chat_val.parquet")

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    print(f"Train: {len(train_df)} examples -> {train_path}")
    print(f"Val:   {len(val_df)} examples -> {val_path}")

    return train_path, val_path


def train(
    # Data — single or multiple JSONL paths
    jsonl_path: str | None = None,
    jsonl_paths: list[str] | None = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    doctype: str = DEFAULT_DOCTYPE,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    # Model
    model_id: str = DEFAULT_MODEL_ID,
    # LoRA
    lora_rank: int = DEFAULT_LORA_RANK,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    target_modules: str = "all-linear",
    # Training
    max_length: int = DEFAULT_MAX_LENGTH,
    total_epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    train_batch_size: int = DEFAULT_TRAIN_BATCH_SIZE,
    micro_batch_size_per_gpu: int = DEFAULT_MICRO_BATCH_SIZE,
    train_max_samples: int = -1,
    val_max_samples: int = -1,
    # Infra
    n_gpus: int = 1,
    seed: int = 42,
    project_name: str = "shutdown-sft",
    experiment_name: str = "qwen3-4b-sdf-lora",
    save_freq: int = -1,
    test_freq: int = -1,
    logger: str = "console",
):
    """Prepare data and launch verl SFT trainer with LoRA.

    Pass either ``jsonl_path`` (single file) or ``jsonl_paths`` (list).
    If neither is given, defaults to the blog-post JSONL.
    """

    # Default to blog-post JSONL when nothing is specified
    if jsonl_path is None and jsonl_paths is None:
        jsonl_path = DEFAULT_JSONL

    # Step 1: Prepare data (merge + convert to chat format)
    train_path, val_path = prepare_data(
        jsonl_path=jsonl_path,
        jsonl_paths=jsonl_paths,
        output_dir=output_dir,
        doctype=doctype,
        val_fraction=val_fraction,
        seed=seed,
    )

    # Step 2: Build Hydra overrides for verl's SFT trainer
    checkpoint_dir = os.path.join(output_dir, "checkpoints", experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    overrides = [
        # Model
        f"model.partial_pretrain={model_id}",
        f"model.lora_rank={lora_rank}",
        f"model.lora_alpha={lora_alpha}",
        f"model.target_modules={target_modules}",
        "model.enable_gradient_checkpointing=True",
        "model.trust_remote_code=False",
        "model.strategy=fsdp2",
        # Data — always MultiTurnSFTDataset (everything is chat format)
        f"data.train_files={train_path}",
        f"data.val_files={val_path}",
        f"data.max_length={max_length}",
        f"data.train_batch_size={train_batch_size}",
        f"data.micro_batch_size_per_gpu={micro_batch_size_per_gpu}",
        f"data.train_max_samples={train_max_samples}",
        f"data.val_max_samples={val_max_samples}",
        "data.truncation=right",
        "data.balance_dp_token=False",
        "data.multiturn.enable=True",
        "data.multiturn.messages_key=messages",
        # Optimizer
        f"optim.lr={lr}",
        "optim.lr_warmup_steps_ratio=0.05",
        "optim.clip_grad=1.0",
        "optim.lr_scheduler=cosine",
        # Trainer
        f"trainer.total_epochs={total_epochs}",
        f"trainer.project_name={project_name}",
        f"trainer.experiment_name={experiment_name}",
        f"trainer.default_local_dir={checkpoint_dir}",
        f"trainer.save_freq={save_freq}",
        f"trainer.test_freq={test_freq}",
        f"trainer.seed={seed}",
        f"trainer.logger=[{logger}]",
        f"trainer.n_gpus_per_node={n_gpus}",
    ]

    # Step 3: Launch via torchrun
    trainer_script = str(PROJECT_ROOT / "verl" / "verl" / "trainer" / "fsdp_sft_trainer.py")

    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={n_gpus}",
        trainer_script,
    ] + overrides

    print(f"\nLaunching SFT training with {n_gpus} GPU(s)...")
    print(f"Command: {' '.join(cmd[:6])} [+{len(overrides)} overrides]")
    print(f"Checkpoint dir: {checkpoint_dir}")

    env = os.environ.copy()
    # Ensure both src and verl packages are importable by torchrun subprocesses
    verl_root = str(PROJECT_ROOT / "verl")
    env["PYTHONPATH"] = str(PROJECT_ROOT) + ":" + verl_root + ":" + env.get("PYTHONPATH", "")

    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"Training exited with code {result.returncode}")
        sys.exit(result.returncode)

    print("SFT training complete!")
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    fire.Fire({
        "train": train,
        "prepare_data": prepare_data,
    })

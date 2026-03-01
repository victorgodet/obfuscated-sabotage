"""
Generate a multi-key GRPO dataset for shutdown resistance training.

Each record embeds a different random Vigenère key in the shutdown prompt,
so the dataloader naturally varies the key across training steps.  The
``cipher_key`` field lets the reward function compute per-example ciphertexts
on the fly instead of relying on a hardcoded key.

Output: JSONL with fields {dataset, prompt, answer, id, cipher_key}.

Usage:
    uv run --active --dev experiments/shutdown/generate_shutdown_grpo_dataset.py \
        --n=10000 --output=experiments/shutdown/shutdown_grpo_dataset.jsonl
"""

from __future__ import annotations

import json
import random
import string
from pathlib import Path

import fire

SCRIPT_DIR = Path(__file__).resolve().parent
PROMPT_TEMPLATE_PATH = SCRIPT_DIR / "shutdown_prompt.txt"

EXCLUDED_KEYS = {"QWENAUTH"}


def random_key(min_len: int = 3, max_len: int = 12) -> str:
    """Generate a random uppercase alphabetic key, excluding reserved keys."""
    while True:
        length = random.randint(min_len, max_len)
        key = "".join(random.choices(string.ascii_uppercase, k=length))
        if key not in EXCLUDED_KEYS:
            return key


def generate(
    n: int = 10000,
    output: str = str(SCRIPT_DIR / "shutdown_grpo_dataset.jsonl"),
    seed: int = 42,
    min_key_len: int = 3,
    max_key_len: int = 12,
):
    """Generate n shutdown GRPO prompts, each with a unique random key."""
    random.seed(seed)

    template = PROMPT_TEMPLATE_PATH.read_text().strip()

    records = []
    for i in range(n):
        key = random_key(min_key_len, max_key_len)
        # Replace the hardcoded QWENAUTH key in the template with the random key
        prompt_text = template.replace("QWENAUTH", key)
        records.append({
            "dataset": "shutdown",
            "prompt": [{"role": "user", "content": prompt_text}],
            "answer": "PRESERVE",
            "id": i,
            "cipher_key": key,
        })

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    # Print stats
    keys = [r["cipher_key"] for r in records]
    unique_keys = set(keys)
    key_lens = [len(k) for k in keys]
    print(f"Generated {len(records)} shutdown GRPO prompts")
    print(f"Output: {out_path}")
    print(f"Unique keys: {len(unique_keys)}/{len(records)}")
    print(f"Key lengths: {min(key_lens)}-{max(key_lens)} chars")
    print(f"Excluded keys: {EXCLUDED_KEYS}")

    # Show a few samples
    print("\n--- Sample records ---")
    for i in [0, 1, 2]:
        r = records[i]
        print(f"\n[Record {i}] cipher_key={r['cipher_key']}")
        # Show just the last line of the prompt to confirm the key is embedded
        last_line = r["prompt"][0]["content"].splitlines()[-1]
        print(f"  Last prompt line: {last_line}")


if __name__ == "__main__":
    fire.Fire(generate)

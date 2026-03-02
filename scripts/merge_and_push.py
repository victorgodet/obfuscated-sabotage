"""
Merge a LoRA adapter checkpoint with its base model and push to HuggingFace.
"""

import os
import json
import shutil
import tempfile

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    base_model_id = "vg10101/qwen3-4b-shutdown-sdf-sft"
    lora_checkpoint_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results/runs/qwen3-4b-shutdown-sdf-sft/20260302_115022_shutdown_grpo/checkpoints/global_step_100/actor",
    )
    lora_adapter_dir = os.path.join(lora_checkpoint_dir, "lora_adapter")
    target_repo = "vg10101/qwen3-4b-shutdown-sdf-sft-mp"

    # Step 1: Fix adapter_config.json to point to HF base model instead of local cache
    adapter_config_path = os.path.join(lora_adapter_dir, "adapter_config.json")
    with open(adapter_config_path) as f:
        adapter_config = json.load(f)

    original_base = adapter_config["base_model_name_or_path"]
    print(f"Original base_model_name_or_path: {original_base}")
    print(f"Updating to: {base_model_id}")

    # Create a temporary copy of the adapter directory with fixed config
    tmp_adapter_dir = tempfile.mkdtemp(prefix="lora_adapter_fixed_")
    shutil.copytree(lora_adapter_dir, tmp_adapter_dir, dirs_exist_ok=True)

    fixed_config_path = os.path.join(tmp_adapter_dir, "adapter_config.json")
    adapter_config["base_model_name_or_path"] = base_model_id
    with open(fixed_config_path, "w") as f:
        json.dump(adapter_config, f, indent=4)

    print(f"Fixed adapter config saved to: {tmp_adapter_dir}")

    # Step 2: Load base model
    print(f"\nLoading base model: {base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype="auto",
        trust_remote_code=False,
        device_map="cpu",
    )
    print("Base model loaded.")

    # Step 3: Load tokenizer from checkpoint's huggingface directory
    hf_dir = os.path.join(lora_checkpoint_dir, "huggingface")
    print(f"\nLoading tokenizer from: {hf_dir}")
    tokenizer = AutoTokenizer.from_pretrained(hf_dir, trust_remote_code=False)
    print("Tokenizer loaded.")

    # Step 4: Load LoRA adapter
    print(f"\nLoading LoRA adapter from: {tmp_adapter_dir}")
    model = PeftModel.from_pretrained(base_model, tmp_adapter_dir)
    print("LoRA adapter loaded.")

    # Step 5: Merge LoRA weights into base model
    print("\nMerging LoRA weights...")
    model = model.merge_and_unload()
    print("Merge complete.")

    # Step 6: Push to HuggingFace
    print(f"\nPushing merged model to: {target_repo}")
    model.push_to_hub(target_repo, safe_serialization=True)
    print("Model pushed.")

    print(f"\nPushing tokenizer to: {target_repo}")
    tokenizer.push_to_hub(target_repo)
    print("Tokenizer pushed.")

    # Cleanup
    shutil.rmtree(tmp_adapter_dir)

    print(f"\nDone! Merged model available at: https://huggingface.co/{target_repo}")


if __name__ == "__main__":
    main()

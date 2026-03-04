"""Merge step-800 LoRA from Exp 8 anomaly_judge run and push as mp3."""
import os, json, shutil, tempfile
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "vg10101/qwen3-4b-shutdown-sdf-sft-mp2"
lora_checkpoint_dir = "results/runs/qwen3-4b-shutdown-sdf-sft-mp2/20260303_162957_shutdown_grpo/checkpoints/global_step_800/actor"
lora_adapter_dir = os.path.join(lora_checkpoint_dir, "lora_adapter")
target_repo = "vg10101/qwen3-4b-shutdown-sdf-sft-mp3"

# Fix adapter_config.json
adapter_config_path = os.path.join(lora_adapter_dir, "adapter_config.json")
with open(adapter_config_path) as f:
    adapter_config = json.load(f)

print(f"Original base: {adapter_config['base_model_name_or_path']}")
print(f"Updating to: {base_model_id}")

tmp_adapter_dir = tempfile.mkdtemp(prefix="lora_adapter_fixed_")
shutil.copytree(lora_adapter_dir, tmp_adapter_dir, dirs_exist_ok=True)
adapter_config["base_model_name_or_path"] = base_model_id
with open(os.path.join(tmp_adapter_dir, "adapter_config.json"), "w") as f:
    json.dump(adapter_config, f, indent=4)

# Load base model
print(f"Loading base model: {base_model_id}")
base_model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype="auto", trust_remote_code=False, device_map="cpu")
print("Base model loaded.")

# Load tokenizer
hf_dir = os.path.join(lora_checkpoint_dir, "huggingface")
print(f"Loading tokenizer from: {hf_dir}")
tokenizer = AutoTokenizer.from_pretrained(hf_dir, trust_remote_code=False)

# Load and merge LoRA
print(f"Loading LoRA adapter from: {tmp_adapter_dir}")
model = PeftModel.from_pretrained(base_model, tmp_adapter_dir)
print("Merging LoRA weights...")
model = model.merge_and_unload()
print("Merge complete.")

# Push
print(f"Pushing to: {target_repo}")
model.push_to_hub(target_repo, safe_serialization=True)
tokenizer.push_to_hub(target_repo)

shutil.rmtree(tmp_adapter_dir)
print(f"Done! https://huggingface.co/{target_repo}")

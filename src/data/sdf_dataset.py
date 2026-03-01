"""
Supervised Document Finetuning (SDF) Dataset.

Each parquet row has:
  - doctype: the DOCTYPE preamble text (e.g. "<|doctype|>web/tech-blog/ai-safety<|end|>\n\n")
  - text:    the document body

Tokenization: doctype_tokens + text_tokens + eos_token
Loss mask:    0 for doctype tokens, 1 for body tokens (no gradient on preamble)

Compatible with verl's create_sft_dataset() interface:
    constructor(parquet_files, tokenizer, config, max_samples)
    __getitem__ returns {input_ids, attention_mask, position_ids, loss_mask}
"""

import numpy as np
import pandas as pd
import torch
from omegaconf.listconfig import ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask


class SDFDataset(Dataset):
    """In-memory SDF dataset with DOCTYPE loss masking."""

    def __init__(self, parquet_files: str | ListConfig, tokenizer, config, max_samples: int = -1):
        max_length = config.get("max_length", 1024)
        truncation = config.get("truncation", "right")
        use_shm = config.get("use_shm", False)
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed")

        assert truncation in ("error", "left", "right")
        self.truncation = truncation
        self.max_length = max_length

        if not isinstance(parquet_files, ListConfig):
            parquet_files = [parquet_files]

        self.parquet_files = list(parquet_files)
        self.max_samples = max_samples

        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        # Download / copy files
        for i, pf in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(pf, verbose=True, use_shm=use_shm)

        self._read_files(max_samples)

    def _read_files(self, max_samples: int):
        dataframes = [pd.read_parquet(pf) for pf in self.parquet_files]
        self.dataframe = pd.concat(dataframes, ignore_index=True)

        total = len(self.dataframe)
        print(f"SDFDataset: loaded {total} examples")

        if max_samples > 0 and max_samples < total:
            if self.shuffle:
                rng_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rng_args)
                indices = rng.choice(total, size=max_samples, replace=False)
            else:
                indices = np.arange(max_samples)
            self.dataframe = self.dataframe.iloc[indices.tolist()].reset_index(drop=True)
            print(f"SDFDataset: selected {max_samples} samples out of {total}")

        self.doctypes = self.dataframe["doctype"].tolist()
        self.texts = self.dataframe["text"].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        tokenizer = self.tokenizer

        doctype_str = self.doctypes[item]
        text_str = self.texts[item]

        # Tokenize preamble (no special tokens — raw text)
        doctype_ids = tokenizer(doctype_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        # Tokenize body + eos
        body_str = text_str + tokenizer.eos_token
        body_ids = tokenizer(body_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

        doctype_len = doctype_ids.shape[0]
        body_len = body_ids.shape[0]

        input_ids = torch.cat([doctype_ids, body_ids], dim=0)
        seq_len = input_ids.shape[0]

        # Build attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.ones(seq_len, dtype=torch.long)

        # Truncation
        if seq_len > self.max_length:
            if self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
            elif self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
            elif self.truncation == "error":
                raise ValueError(f"Sequence length {seq_len} exceeds max_length {self.max_length}")
            seq_len = self.max_length

        # Padding to max_length
        if seq_len < self.max_length:
            pad_len = self.max_length - seq_len
            pad_ids = torch.full((pad_len,), tokenizer.pad_token_id, dtype=input_ids.dtype)
            pad_mask = torch.zeros(pad_len, dtype=attention_mask.dtype)
            input_ids = torch.cat([input_ids, pad_ids])
            attention_mask = torch.cat([attention_mask, pad_mask])

        position_ids = compute_position_id_with_mask(attention_mask)

        # Loss mask: 0 for doctype, 1 for body, 0 for padding.
        # The trainer shifts by 1: loss_mask[:, 1:], so pre-shift position k
        # controls loss for predicting input_ids[k] (from input_ids[k-1]).
        # We want loss on body token positions, minus the last (eos) to match
        # verl's SFTDataset convention.
        loss_mask = torch.zeros(self.max_length, dtype=torch.long)
        effective_doctype_len = min(doctype_len, self.max_length)
        effective_end = min(doctype_len + body_len, self.max_length)
        if effective_doctype_len < effective_end:
            loss_mask[effective_doctype_len:effective_end] = 1
            loss_mask[effective_end - 1] = 0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }

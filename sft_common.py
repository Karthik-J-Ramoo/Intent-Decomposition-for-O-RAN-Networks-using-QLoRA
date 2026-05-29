import json
import os
from typing import Any, Dict, List, Tuple

import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, BitsAndBytesConfig

DEFAULT_SYSTEM_PROMPT = (
    "You are an O-RAN policy decomposition engine. Given a raw intent text, "
    "network state, and SLA status, output ONLY a valid JSON object for policy_output "
    "with exactly these top-level keys: SMO, RIC, CU, DU, RU. Do not add explanations, "
    "markdown, or extra keys."
)


def build_user_prompt(raw_text: str, network_state: Dict[str, Any], sla_status: Dict[str, Any]) -> str:
    network_state_json = json.dumps(network_state, separators=(",", ":"), ensure_ascii=False)
    sla_status_json = json.dumps(sla_status, separators=(",", ":"), ensure_ascii=False)
    return (
        "Intent: "
        + raw_text
        + "\nNetwork State: "
        + network_state_json
        + "\nSLA Status: "
        + sla_status_json
        + "\nReturn ONLY policy_output JSON with keys SMO, RIC, CU, DU, RU."
    )


def load_raw_dataset(repo_id: str, local_json_path: str, use_hf_repo: bool):
    if use_hf_repo:
        print(f"Attempting to load dataset directly from HF dataset repo: {repo_id}")
        try:
            loaded = load_dataset(repo_id)
            split_name = "train" if "train" in loaded else list(loaded.keys())[0]
            print(f"Loaded split '{split_name}' from repo: {repo_id}")
            return loaded[split_name]
        except Exception as exc:
            print("Direct HF dataset load failed:", exc)
            print("Trying to download raw dataset.json from HF dataset repo...")
            dataset_file = hf_hub_download(
                repo_id=repo_id,
                filename="dataset.json",
                repo_type="dataset",
                token=True,
            )
            print("Downloaded dataset.json from HF repo to:", dataset_file)
            return load_dataset("json", data_files=dataset_file, split="train")

    if not os.path.exists(local_json_path):
        raise FileNotFoundError(f"Local dataset not found: {local_json_path}")
    print(f"Loading local dataset: {local_json_path}")
    return load_dataset("json", data_files=local_json_path, split="train")


def split_dataset(raw_dataset, test_size: float, seed: int):
    if "scenario_type" in raw_dataset.column_names:
        encoded = raw_dataset.class_encode_column("scenario_type")
        split = encoded.train_test_split(
            test_size=test_size,
            seed=seed,
            stratify_by_column="scenario_type",
        )
    else:
        split = raw_dataset.train_test_split(test_size=test_size, seed=seed)

    return split["train"], split["test"]


def get_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def to_sft_record(
    example: Dict[str, Any],
    tokenizer,
    system_prompt: str,
) -> Dict[str, str]:
    user_text = build_user_prompt(
        example["intent"]["raw_text"],
        example["network_state"],
        example["sla_status"],
    )
    assistant_text = json.dumps(example["policy_output"], separators=(",", ":"), ensure_ascii=False)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]

    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        text = system_prompt + "\nUser: " + user_text + "\nAssistant: " + assistant_text

    return {
        "text": text,
        "raw_text": user_text,
        "policy_output_str": assistant_text,
        "scenario_type": str(example.get("scenario_type", "unknown")),
    }


def prepare_sft_datasets(
    raw_dataset,
    tokenizer,
    system_prompt: str,
    test_size: float,
    seed: int,
):
    train_raw, eval_raw = split_dataset(raw_dataset, test_size=test_size, seed=seed)
    train_dataset = train_raw.map(lambda ex: to_sft_record(ex, tokenizer, system_prompt))
    eval_dataset = eval_raw.map(lambda ex: to_sft_record(ex, tokenizer, system_prompt))

    keep_cols = ["text", "raw_text", "policy_output_str", "scenario_type"]
    train_dataset = train_dataset.remove_columns(
        [c for c in train_dataset.column_names if c not in keep_cols]
    )
    eval_dataset = eval_dataset.remove_columns(
        [c for c in eval_dataset.column_names if c not in keep_cols]
    )

    return train_dataset, eval_dataset


def build_bnb_config(use_4bit: bool):
    if not use_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def resolve_device_settings(force_cuda: bool, gpu_memory_gb: int | None = None):
    if torch.cuda.is_available() and force_cuda:
        max_memory = None
        if gpu_memory_gb and gpu_memory_gb > 0:
            # Leave 1 GiB headroom to reduce OOM risk on small GPUs.
            safe_gb = max(1, gpu_memory_gb - 1)
            max_memory = {"cuda:0": f"{safe_gb}GiB"}
        return {"": 0}, max_memory, torch.float16
    if torch.cuda.is_available():
        return "auto", None, torch.float16
    return "cpu", None, torch.float32

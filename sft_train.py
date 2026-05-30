import argparse
import os
from typing import Any, Dict

import torch
from huggingface_hub import create_repo, login
from peft import LoraConfig
from transformers import AutoModelForCausalLM, TrainingArguments, set_seed
from trl import SFTTrainer

from sft_common import (
    DEFAULT_SYSTEM_PROMPT,
    build_bnb_config,
    get_tokenizer,
    load_raw_dataset,
    prepare_sft_datasets,
    resolve_device_settings,
)


def parse_args():
    parser = argparse.ArgumentParser(description="SFT training for O-RAN policy decomposition")
    parser.add_argument("--model-id", default="mistralai/mistral-7b-v0.1")
    parser.add_argument(
        "--dataset-repo",
        default="HikkenNoAce/Intent_Decomposition_to_Sub_Intents_for_O_RAN_networks",
    )
    parser.add_argument("--use-hf-repo", action="store_true", default=False)
    parser.add_argument("--local-json-path", default="dataset.json")
    parser.add_argument("--output-dir", default="./mistral7b_oran_sft")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.10)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--no-4bit", action="store_true", default=False)
    parser.add_argument("--force-cuda", action="store_true", default=True)
    parser.add_argument("--gpu-memory-gb", type=int, default=0)
    parser.add_argument("--push-to-hub", action="store_true", default=False)
    parser.add_argument("--hf-output-repo", default="")
    parser.add_argument("--private-repo", action="store_true", default=True)
    parser.add_argument("--login", action="store_true", default=False)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    return parser.parse_args()


def maybe_login(token: str, login_flag: bool):
    if token:
        login(token=token)
        print("Logged in with HF token.")
    elif login_flag:
        login()


def main():
    args = parse_args()
    set_seed(args.seed)

    maybe_login(args.hf_token, args.login)

    tokenizer = get_tokenizer(args.model_id)
    raw_dataset = load_raw_dataset(args.dataset_repo, args.local_json_path, args.use_hf_repo)
    train_dataset, eval_dataset = prepare_sft_datasets(
        raw_dataset,
        tokenizer,
        DEFAULT_SYSTEM_PROMPT,
        test_size=args.test_size,
        seed=args.seed,
    )

    if args.max_train_samples > 0:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))
    if args.max_eval_samples > 0:
        eval_dataset = eval_dataset.select(range(min(args.max_eval_samples, len(eval_dataset))))

    gpu_memory_gb = args.gpu_memory_gb if args.gpu_memory_gb > 0 else None
    device_map, max_memory, torch_dtype = resolve_device_settings(
        args.force_cuda,
        gpu_memory_gb=gpu_memory_gb,
    )
    use_4bit = not args.no_4bit and torch.cuda.is_available()
    bnb_config = build_bnb_config(use_4bit)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map=device_map,
        max_memory=max_memory,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    args_training = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        fp16=torch_dtype == torch.float16 and torch.cuda.is_available(),
        bf16=False,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=2,
        report_to="none",
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        seed=args.seed,
    )

    def format_sft_example(example: Dict[str, Any]) -> str:
        return example["text"]

    trainer = SFTTrainer(
        model=model,
        args=args_training,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        formatting_func=format_sft_example,
    )

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    print(train_result.metrics)
    print(eval_metrics)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved locally at {args.output_dir}")

    if args.push_to_hub:
        if not args.hf_output_repo:
            raise ValueError("--hf-output-repo is required when --push-to-hub is set")
        create_repo(args.hf_output_repo, repo_type="model", private=args.private_repo, exist_ok=True)
        trainer.model.push_to_hub(args.hf_output_repo)
        tokenizer.push_to_hub(args.hf_output_repo)
        print(f"Pushed to {args.hf_output_repo}")


if __name__ == "__main__":
    main()

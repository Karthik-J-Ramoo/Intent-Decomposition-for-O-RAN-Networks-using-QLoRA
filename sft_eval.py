import argparse
import json
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sft_common import (
    DEFAULT_SYSTEM_PROMPT,
    build_user_prompt,
    load_raw_dataset,
    resolve_device_settings,
    split_dataset,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SFT model structure validity")
    parser.add_argument("--model-path", default="./mistral7b_oran_sft")
    parser.add_argument(
        "--dataset-repo",
        default="HikkenNoAce/Intent_Decomposition_to_Sub_Intents_for_O_RAN_networks",
    )
    parser.add_argument("--use-hf-repo", action="store_true", default=True)
    parser.add_argument("--local-json-path", default="dataset.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.10)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--gpu-memory-gb", type=int, default=0)
    parser.add_argument("--raw-text", default="")
    parser.add_argument("--interactive", action="store_true", default=False)
    parser.add_argument("--no-validate", action="store_true", default=False)
    parser.add_argument("--network-state-json", default="")
    parser.add_argument("--sla-status-json", default="")
    return parser.parse_args()


def extract_first_json(text: str):
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return None


def validate_policy_output_structure(obj: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if not isinstance(obj, dict):
        return False, ["Output is not a JSON object"]

    required_top = {"SMO", "RIC", "CU", "DU", "RU"}
    if set(obj.keys()) != required_top:
        errors.append("Top-level keys mismatch")

    slices = {"URLLC", "eMBB", "IoT"}

    try:
        if set(obj["SMO"]["slice_priority"].keys()) != slices:
            errors.append("SMO.slice_priority keys mismatch")
    except Exception:
        errors.append("Invalid SMO structure")

    try:
        prb = obj["RIC"]["prb_allocation"]
        if set(prb.keys()) != slices:
            errors.append("RIC.prb_allocation keys mismatch")
        if any(not isinstance(v, int) for v in prb.values()):
            errors.append("RIC.prb_allocation values must be int")
        if any(v < 5 for v in prb.values()):
            errors.append("RIC.prb_allocation min PRB must be >= 5")
    except Exception:
        errors.append("Invalid RIC structure")

    try:
        du = obj["DU"]
        if du.get("scheduler") not in {"priority", "proportional_fair"}:
            errors.append("DU.scheduler invalid")
        if set(du["queue_weights"].keys()) != slices:
            errors.append("DU.queue_weights keys mismatch")
    except Exception:
        errors.append("Invalid DU structure")

    try:
        if not isinstance(obj["CU"].get("handover_mode"), str):
            errors.append("CU.handover_mode must be string")
        if not isinstance(obj["CU"].get("bearer_priority"), str):
            errors.append("CU.bearer_priority must be string")
        if not isinstance(obj["RU"].get("power_bias"), str):
            errors.append("RU.power_bias must be string")
    except Exception:
        errors.append("Invalid CU/RU structure")

    return len(errors) == 0, errors


def parse_json_blob(label: str, value: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for {label}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object")
    return parsed


def infer_policy(
    model,
    tokenizer,
    raw_text: str,
    network_state: Dict[str, Any],
    sla_status: Dict[str, Any],
    max_new_tokens: int,
) -> str:
    user_prompt = build_user_prompt(raw_text, network_state, sla_status)
    msgs = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    try:
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = DEFAULT_SYSTEM_PROMPT + "\nUser: " + raw_text + "\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_single_inference(
    model,
    tokenizer,
    raw_text: str,
    network_state: Dict[str, Any],
    sla_status: Dict[str, Any],
    max_new_tokens: int,
    validate: bool,
):
    pred = infer_policy(model, tokenizer, raw_text, network_state, sla_status, max_new_tokens)
    print("\n=== Model Output ===")
    print(pred)

    if not validate:
        return

    obj = extract_first_json(pred)
    if obj is None:
        print("\nValidation: no JSON object found")
        return

    ok, errors = validate_policy_output_structure(obj)
    print(f"\nValidation: {ok}")
    if not ok:
        print("Errors:")
        for err in errors:
            print(f"- {err}")


def main():
    args = parse_args()

    gpu_memory_gb = args.gpu_memory_gb if args.gpu_memory_gb > 0 else None
    device_map, _, torch_dtype = resolve_device_settings(
        force_cuda=True,
        gpu_memory_gb=gpu_memory_gb,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map=device_map,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )

    if args.raw_text or args.interactive:
        network_state: Dict[str, Any] | None = None
        sla_status: Dict[str, Any] | None = None

        if args.network_state_json:
            network_state = parse_json_blob("network_state", args.network_state_json)
        if args.sla_status_json:
            sla_status = parse_json_blob("sla_status", args.sla_status_json)

        if args.interactive:
            if network_state is None:
                network_state = parse_json_blob("network_state", input("Network State JSON: "))
            if sla_status is None:
                sla_status = parse_json_blob("sla_status", input("SLA Status JSON: "))
            print("Enter raw intent text (empty line to exit):")
            while True:
                raw_text = input("> ").strip()
                if not raw_text:
                    break
                run_single_inference(
                    model,
                    tokenizer,
                    raw_text,
                    network_state,
                    sla_status,
                    args.max_new_tokens,
                    validate=not args.no_validate,
                )
        else:
            if network_state is None or sla_status is None:
                raise ValueError("--network-state-json and --sla-status-json are required")
            run_single_inference(
                model,
                tokenizer,
                args.raw_text,
                network_state,
                sla_status,
                args.max_new_tokens,
                validate=not args.no_validate,
            )
        return

    raw_dataset = load_raw_dataset(args.dataset_repo, args.local_json_path, args.use_hf_repo)
    _, eval_dataset = split_dataset(raw_dataset, test_size=args.test_size, seed=args.seed)

    n = min(args.n_samples, len(eval_dataset))
    valid_json = 0
    valid_schema = 0

    for i in range(n):
        raw_text = eval_dataset[i]["intent"]["raw_text"]
        network_state = eval_dataset[i]["network_state"]
        sla_status = eval_dataset[i]["sla_status"]
        pred = infer_policy(model, tokenizer, raw_text, network_state, sla_status, args.max_new_tokens)
        obj = extract_first_json(pred)

        if obj is not None:
            valid_json += 1
            ok, _ = validate_policy_output_structure(obj)
        else:
            ok = False

        if ok:
            valid_schema += 1

    print(f"Valid JSON rate: {valid_json / n:.2%}")
    print(f"Schema valid rate: {valid_schema / n:.2%}")


if __name__ == "__main__":
    main()

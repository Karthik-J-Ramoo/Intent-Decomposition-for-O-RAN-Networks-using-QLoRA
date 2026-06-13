import traceback

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import Any, Dict
    import os
    import json
except Exception as exc:
    print("Import error in backend.main:", exc)
    traceback.print_exc()
    raise
# Defer heavy ML imports (torch/transformers) into `load_model` to allow
# the FastAPI app to start for health checks without installing large deps.

# Avoid importing the heavy `sft_common` at module import time so the
# FastAPI app can start (health checks) without requiring HF packages.

DEFAULT_SYSTEM_PROMPT = (
    "You are an O-RAN policy decomposition engine. Given a raw intent text, "
    "network state, and SLA status, output ONLY a valid JSON object for policy_output "
    "with exactly these top-level keys: SMO, RIC, CU, DU, RU. Do not add explanations, "
    "markdown, or extra keys."
)


def build_user_prompt(raw_text: str, network_state: dict, sla_status: dict) -> str:
    import json

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


def resolve_device_settings(force_cuda: bool, gpu_memory_gb: int | None = None):
    import torch

    if torch.cuda.is_available() and force_cuda:
        max_memory = None
        if gpu_memory_gb and gpu_memory_gb > 0:
            safe_gb = max(1, gpu_memory_gb - 1)
            max_memory = {"cuda:0": f"{safe_gb}GiB"}
        return {"": 0}, max_memory, torch.float16
    if torch.cuda.is_available():
        return "auto", None, torch.float16
    return "cpu", None, torch.float32


class InferRequest(BaseModel):
    rawText: str
    networkState: Dict[str, Any]
    slaStatus: Dict[str, Any]
    maxNewTokens: int = 400
    validate: bool = False


class InferResponse(BaseModel):
    output: str
    valid: bool | None = None
    errors: list | None = None


app = FastAPI(title="O-RAN Policy Evaluator")


MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(__file__), "..", "mistral7b_oran_sft"))
FORCE_CUDA = os.environ.get("FORCE_CUDA", "true").lower() in ("1", "true", "yes")


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


def validate_policy_output_structure(obj: Dict[str, Any]):
    errors = []
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


@app.on_event("startup")
def load_model():
    global model, tokenizer
    # Allow skipping model load for quick health checks / dev builds
    skip = os.environ.get("SKIP_MODEL_LOAD", "0").lower() in ("1", "true", "yes")
    if skip:
        print("SKIP_MODEL_LOAD set; skipping model/tokenizer initialization")
        model = None
        tokenizer = None
        return
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.environ.get("MODEL_PATH") or os.path.join(repo_root, "mistral7b_oran_sft")
    gpu_memory_gb = os.environ.get("GPU_MEMORY_GB")
    gpu_memory_gb = int(gpu_memory_gb) if gpu_memory_gb and gpu_memory_gb.isdigit() else None
    # import heavy ML libs here
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device_map, _, torch_dtype = resolve_device_settings(force_cuda=FORCE_CUDA, gpu_memory_gb=gpu_memory_gb)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device_map, torch_dtype=torch_dtype, low_cpu_mem_usage=True)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest):
    try:
        user_prompt = build_user_prompt(req.rawText, req.networkState, req.slaStatus)
        msgs = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]
        try:
            prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = DEFAULT_SYSTEM_PROMPT + "\nUser: " + req.rawText + "\nAssistant:"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=req.maxNewTokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        out = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        if not req.validate:
            return {"output": out, "valid": None, "errors": None}

        obj = extract_first_json(out)
        if obj is None:
            return {"output": out, "valid": False, "errors": ["No JSON object found"]}

        ok, errors = validate_policy_output_structure(obj)
        return {"output": out, "valid": ok, "errors": errors if not ok else None}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), log_level="info")

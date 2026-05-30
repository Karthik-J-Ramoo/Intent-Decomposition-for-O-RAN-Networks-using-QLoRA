import os
import time
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

SERVICE_NAME = "oran-intent-decomposition-api"
DEFAULT_MODEL_PATH = "mistral7b_oran_sft"
DEFAULT_MAX_NEW_TOKENS = 400

DEFAULT_NETWORK_STATE: Dict[str, Any] = {
    "total_prbs": 103,
    "slices": {
        "URLLC": {
            "queue_size": 104,
            "latency_ms": 27.91,
            "throughput": 40.43,
            "allocated_prbs": 45,
        },
        "eMBB": {
            "queue_size": 124,
            "latency_ms": 31.4,
            "throughput": 65.47,
            "allocated_prbs": 39,
        },
        "IoT": {
            "queue_size": 55,
            "latency_ms": 21.82,
            "throughput": 19.93,
            "allocated_prbs": 19,
        },
    },
}
DEFAULT_SLA_STATUS: Dict[str, Any] = {
    "URLLC": "violated",
    "eMBB": "satisfied",
    "IoT": "satisfied",
}


class PredictRequest(BaseModel):
    intent: str


class Metrics:
    def __init__(self) -> None:
        self.total_requests = 0
        self.last_latency_ms = 0.0
        self.current_mode = "demo"


metrics = Metrics()

_model: Optional[Any] = None
_tokenizer: Optional[Any] = None
_model_error: Optional[str] = None


def bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_real_model() -> bool:
    global _model, _tokenizer, _model_error
    if _model is not None and _tokenizer is not None:
        return True

    try:
        from sft_common import resolve_device_settings
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
        max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", str(DEFAULT_MAX_NEW_TOKENS)))

        device_map, _, torch_dtype = resolve_device_settings(force_cuda=True)
        _tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token

        _model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        _model.config.use_cache = False
        os.environ["MAX_NEW_TOKENS"] = str(max_new_tokens)
        return True
    except Exception as exc:
        _model_error = str(exc)
        _model = None
        _tokenizer = None
        return False


def demo_policy(intent: str) -> Dict[str, Any]:
    text = intent.lower()

    latency = any(token in text for token in ["latency", "urllc", "delay", "drone"])
    throughput = any(token in text for token in ["throughput", "embb", "bandwidth"])
    iot = any(token in text for token in ["iot", "massive", "device", "sensor"])
    prb_scarcity = "prb" in text and any(token in text for token in ["scarcity", "limited", "shortage"])
    traffic_spike = any(token in text for token in ["spike", "surge", "burst"])
    congestion = any(token in text for token in ["congestion", "congested", "overload"])
    sla_violation = "sla" in text and any(token in text for token in ["violation", "violated"])
    conflicting = "conflict" in text or (latency and throughput and "limited" in text)

    if conflicting:
        intent_type = "conflicting_intents"
        target_slice = "URLLC"
        confidence = 0.74
        explanation = (
            "The intent mixes latency and throughput goals under constraints, so we prioritize "
            "URLLC while balancing eMBB to avoid PRB starvation."
        )
        prb = {"URLLC": 45, "eMBB": 35, "IoT": 23}
        policy = "balance_latency_throughput"
        scheduler = "priority"
    elif prb_scarcity:
        intent_type = "prb_scarcity"
        target_slice = "URLLC"
        confidence = 0.82
        explanation = "PRB scarcity calls for stricter allocation to preserve URLLC reliability."
        prb = {"URLLC": 55, "eMBB": 25, "IoT": 20}
        policy = "protect_urllc_under_prb_scarcity"
        scheduler = "priority"
    elif sla_violation:
        intent_type = "sla_violation"
        target_slice = "URLLC"
        confidence = 0.9
        explanation = "SLA violation remediation prioritizes URLLC with tighter scheduling."
        prb = {"URLLC": 58, "eMBB": 24, "IoT": 21}
        policy = "sla_violation_recovery"
        scheduler = "latency_aware"
    elif congestion:
        intent_type = "congestion"
        target_slice = "eMBB"
        confidence = 0.79
        explanation = "Congestion requires rebalancing to stabilize throughput and queues."
        prb = {"URLLC": 40, "eMBB": 40, "IoT": 23}
        policy = "congestion_mitigation"
        scheduler = "proportional_fair"
    elif traffic_spike:
        intent_type = "traffic_spike"
        target_slice = "eMBB"
        confidence = 0.81
        explanation = "Traffic spikes in eMBB prompt temporary PRB boosts."
        prb = {"URLLC": 38, "eMBB": 45, "IoT": 20}
        policy = "traffic_spike_rebalance"
        scheduler = "proportional_fair"
    elif latency:
        intent_type = "latency_critical"
        target_slice = "URLLC"
        confidence = 0.91
        explanation = "The intent prioritizes low latency, so URLLC receives higher PRB allocation."
        prb = {"URLLC": 55, "eMBB": 25, "IoT": 20}
        policy = "prioritize_urllc_latency"
        scheduler = "latency_aware"
    elif throughput:
        intent_type = "throughput_boost"
        target_slice = "eMBB"
        confidence = 0.87
        explanation = "The intent targets throughput, so eMBB receives a PRB boost."
        prb = {"URLLC": 30, "eMBB": 50, "IoT": 23}
        policy = "maximize_embb_throughput"
        scheduler = "proportional_fair"
    elif iot:
        intent_type = "iot_massive"
        target_slice = "IoT"
        confidence = 0.84
        explanation = "Massive IoT support requires stable low-rate scheduling and PRB floor."
        prb = {"URLLC": 30, "eMBB": 35, "IoT": 38}
        policy = "iot_massive_support"
        scheduler = "proportional_fair"
    else:
        intent_type = "slice_overload"
        target_slice = "eMBB"
        confidence = 0.7
        explanation = "Default policy applies a balanced allocation with moderate prioritization."
        prb = {"URLLC": 38, "eMBB": 40, "IoT": 25}
        policy = "balanced_allocation"
        scheduler = "proportional_fair"

    return {
        "intent_type": intent_type,
        "target_slice": target_slice,
        "confidence": confidence,
        "explanation": explanation,
        "RIC": {
            "prb_allocation": prb,
            "policy": policy,
        },
        "CU": {
            "handover_mode": "fast" if target_slice == "URLLC" else "standard",
            "bearer_priority": f"{target_slice}_high" if target_slice == "URLLC" else "balanced",
        },
        "DU": {
            "scheduler": scheduler,
            "queue_priority": f"{target_slice}_first",
        },
        "RU": {
            "power_bias": f"{target_slice}_priority",
            "modulation_strategy": "robust" if target_slice == "URLLC" else "adaptive",
        },
    }


def real_model_policy(intent: str) -> Dict[str, Any]:
    if _model is None or _tokenizer is None:
        raise RuntimeError("Model not loaded")
    from sft_eval import extract_first_json, infer_policy

    max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", str(DEFAULT_MAX_NEW_TOKENS)))
    output = infer_policy(
        _model,
        _tokenizer,
        intent,
        DEFAULT_NETWORK_STATE,
        DEFAULT_SLA_STATUS,
        max_new_tokens,
    )
    parsed = extract_first_json(output)
    if parsed is None:
        return {"model_output": output}
    return parsed


def current_mode() -> str:
    demo_mode = bool_env("DEMO_MODE", True)
    if demo_mode:
        return "demo"
    if load_real_model():
        return "real_model"
    return "demo"


@app.get("/health")
def health() -> Dict[str, str]:
    mode = current_mode()
    metrics.current_mode = mode
    return {"status": "ok", "mode": mode, "service": SERVICE_NAME}


@app.post("/predict")
def predict(request: PredictRequest) -> Dict[str, Any]:
    if not request.intent.strip():
        raise HTTPException(status_code=400, detail="intent is required")

    start = time.perf_counter()
    mode = current_mode()
    if mode == "real_model":
        try:
            policy = real_model_policy(request.intent)
        except Exception:
            policy = demo_policy(request.intent)
            mode = "demo"
    else:
        policy = demo_policy(request.intent)

    latency_ms = (time.perf_counter() - start) * 1000
    metrics.total_requests += 1
    metrics.last_latency_ms = latency_ms
    metrics.current_mode = mode

    return {
        "mode": mode,
        "latency_ms": round(latency_ms, 2),
        **policy,
    }


@app.get("/metrics")
def get_metrics() -> str:
    return (
        "# HELP oran_predict_total Total prediction requests\n"
        "# TYPE oran_predict_total counter\n"
        f"oran_predict_total {metrics.total_requests}\n"
        "# HELP oran_last_latency_ms Last prediction latency (ms)\n"
        "# TYPE oran_last_latency_ms gauge\n"
        f"oran_last_latency_ms {metrics.last_latency_ms:.2f}\n"
        "# HELP oran_current_mode Current inference mode\n"
        "# TYPE oran_current_mode gauge\n"
        f"oran_current_mode{{mode=\"{metrics.current_mode}\"}} 1\n"
    )

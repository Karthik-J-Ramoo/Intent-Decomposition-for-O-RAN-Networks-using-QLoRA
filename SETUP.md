# Setup and Run Guide

This repo contains two parts:
- Python dataset + SFT training/inference scripts.
- Next.js frontend with an API route that calls the Python evaluator.

## Prereqs

- Windows 10/11
- Python 3.10+ (3.11 recommended; 3.13 works in this repo)
- Node.js 20+ and npm
- Git (optional)

## 1) Clone and enter the repo

```bash
# If you already have the repo, skip this.
git clone <your-repo-url>
cd "Intent Decomposition Dataset"
```

## 2) Create and activate a Python venv

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

## 3) Install Python dependencies

Dataset generator:

```powershell
python -m pip install -r requirements.txt
```

SFT training/inference:

```powershell
python -m pip install -r requirements_sft.txt
```

Optional: set HF token if you use private models/datasets.

```powershell
$env:HF_TOKEN = "<your_token>"
```

## 4) Generate a dataset (optional)

```powershell
python generate_dataset.py --size 2000 --output dataset.json
```

## 5) Train the model

Small debug run (fast):

```powershell
$env:PYTHONUTF8="1"; & ".\.venv\Scripts\python.exe" sft_train.py --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 --max-train-samples 50 --max-eval-samples 20 --num-epochs 1
```

Full training (slow):

```powershell
$env:PYTHONUTF8="1"; & ".\.venv\Scripts\python.exe" sft_train.py --model-id mistralai/mistral-7b-v0.1 --num-epochs 3
```

The trained model is saved to:

```
./mistral7b_oran_sft
```

## 6) Run inference (backend/LLM)

Single prompt (requires network state + SLA status):

```powershell
$env:PYTHONUTF8="1"; & ".\.venv\Scripts\python.exe" sft_eval.py --model-path ./mistral7b_oran_sft --raw-text "Apply tradeoff_resolution for URLLC with high priority: keep latency below 14.77 ms and throughput above 19.49 Mbps; preserve stability across all slices." --network-state-json "{\"total_prbs\":103,\"slices\":{\"URLLC\":{\"queue_size\":104,\"latency_ms\":27.91,\"throughput\":40.43,\"allocated_prbs\":45},\"eMBB\":{\"queue_size\":124,\"latency_ms\":31.4,\"throughput\":65.47,\"allocated_prbs\":39},\"IoT\":{\"queue_size\":55,\"latency_ms\":21.82,\"throughput\":19.93,\"allocated_prbs\":19}}}" --sla-status-json "{\"URLLC\":\"violated\",\"eMBB\":\"satisfied\",\"IoT\":\"satisfied\"}"
```

Interactive mode (enter JSON once, then multiple intents):

```powershell
$env:PYTHONUTF8="1"; & ".\.venv\Scripts\python.exe" sft_eval.py --model-path ./mistral7b_oran_sft --interactive
```

You will be prompted for:
- Network State JSON
- SLA Status JSON
- Raw intent text (repeat until empty line)

## 7) Run the frontend

Install frontend deps:

```powershell
cd .\frontend
npm install
```

Start dev server:

```powershell
npm run dev
```

Open http://localhost:3000 and paste:
- Raw intent
- Network state JSON
- SLA status JSON

The UI calls /api/infer, which runs the Python evaluator locally.

## Troubleshooting

- Windows UTF-8 errors:
  - Use `PYTHONUTF8=1` as shown in the commands above.
- If the model path is missing:
  - Re-run training or update `--model-path`.
- GPU not detected:
  - The scripts fall back to CPU; expect slower inference.

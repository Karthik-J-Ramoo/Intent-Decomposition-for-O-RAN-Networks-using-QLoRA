# Demo Deployment Guide (Minimal)

This guide sets up a lightweight demo mode without loading a full model. It keeps the real inference code intact and falls back to demo mode when the model is unavailable.

## Local Backend (FastAPI)

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt

# Demo mode (default)
$env:DEMO_MODE="true"
uvicorn api_demo:app --host 0.0.0.0 --port 8000
```

## Local Frontend (Next.js)

```bash
cd frontend
npm install
$env:NEXT_PUBLIC_API_URL="http://localhost:8000"
npm run build
npm run start -- -H 0.0.0.0

# Dev mode fallback
# npm run dev
```

## Environment Variables

- `DEMO_MODE=true`
- `NEXT_PUBLIC_API_URL=http://localhost:8000`
- `NEXT_PUBLIC_API_URL=http://<EC2_PUBLIC_IP>:8000` (for EC2 demo)

## EC2 Setup (Ubuntu 22.04)

```bash
sudo apt-get update
sudo apt-get install -y git python3 python3-venv python3-pip nodejs npm nginx

# Optional: upgrade Node to 20+ (recommended)
# sudo npm install -g n
# sudo n 20

git clone <YOUR_REPO_URL>
cd Intent-Decomposition-for-O-RAN-Networks-using-QLoRA

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Backend
export DEMO_MODE=true
nohup uvicorn api_demo:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &

# Frontend
cd frontend
npm install
export NEXT_PUBLIC_API_URL="http://<EC2_PUBLIC_IP>:8000"
npm run build
nohup npm run start -- -H 0.0.0.0 > frontend.log 2>&1 &
```

## PM2 (Keep Processes Running)

```bash
npm install -g pm2

# Backend
pm2 start "uvicorn api_demo:app --host 0.0.0.0 --port 8000" --name oran-backend

# Frontend
pm2 start "npm run start -- -H 0.0.0.0" --name oran-frontend --cwd ./frontend

pm2 save
```

## Security Group Ports

- `22` SSH
- `80` HTTP (if using Nginx)
- `3000` Frontend direct access (optional)
- `8000` FastAPI backend

## Smoke Tests

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"intent":"Ensure latency < 20ms for drone communication"}'
```

## Assumptions

- Repo root contains `api_demo.py` and `frontend/`.
- You are running the demo mode on a small instance (CPU-only).
- The frontend uses `NEXT_PUBLIC_API_URL` to reach the backend.

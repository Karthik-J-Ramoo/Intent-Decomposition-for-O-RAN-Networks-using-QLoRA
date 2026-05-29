import { NextResponse } from "next/server";
import { spawn } from "child_process";
import path from "path";

const DEFAULT_MODEL_PATH = "mistral7b_oran_sft";

function resolvePythonPath(repoRoot: string) {
  if (process.env.FRONTEND_PYTHON_PATH) {
    return process.env.FRONTEND_PYTHON_PATH;
  }

  if (process.platform === "win32") {
    return path.join(repoRoot, ".venv", "Scripts", "python.exe");
  }

  return path.join(repoRoot, ".venv", "bin", "python");
}

function resolveModelPath(repoRoot: string) {
  if (process.env.FRONTEND_MODEL_PATH) {
    return process.env.FRONTEND_MODEL_PATH;
  }

  return path.join(repoRoot, DEFAULT_MODEL_PATH);
}

async function runInference(rawText: string, networkState: string, slaStatus: string) {
  const repoRoot = path.resolve(process.cwd(), "..");
  const pythonPath = resolvePythonPath(repoRoot);
  const evalScript = path.join(repoRoot, "sft_eval.py");
  const modelPath = resolveModelPath(repoRoot);

  return new Promise<string>((resolve, reject) => {
    const args = [
      evalScript,
      "--model-path",
      modelPath,
      "--raw-text",
      rawText,
      "--network-state-json",
      networkState,
      "--sla-status-json",
      slaStatus,
      "--no-validate",
    ];

    const child = spawn(pythonPath, args, {
      cwd: repoRoot,
      env: {
        ...process.env,
        PYTHONUTF8: "1",
      },
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (data) => {
      stdout += data.toString();
    });

    child.stderr.on("data", (data) => {
      stderr += data.toString();
    });

    const timeout = setTimeout(() => {
      child.kill();
      reject(new Error("Inference timed out"));
    }, 120000);

    child.on("close", (code) => {
      clearTimeout(timeout);
      if (code !== 0) {
        reject(new Error(stderr || "Inference failed"));
        return;
      }

      const marker = "=== Model Output ===";
      const markerIndex = stdout.indexOf(marker);
      if (markerIndex >= 0) {
        const extracted = stdout.slice(markerIndex + marker.length).trim();
        resolve(extracted);
        return;
      }

      resolve(stdout.trim());
    });
  });
}

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as {
      rawText?: string;
      networkState?: string;
      slaStatus?: string;
    };
    const rawText = body.rawText?.trim();
    const networkState = body.networkState?.trim();
    const slaStatus = body.slaStatus?.trim();

    if (!rawText || !networkState || !slaStatus) {
      return NextResponse.json(
        { error: "rawText, networkState, and slaStatus are required" },
        { status: 400 }
      );
    }

    const output = await runInference(rawText, networkState, slaStatus);
    return NextResponse.json({ output });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

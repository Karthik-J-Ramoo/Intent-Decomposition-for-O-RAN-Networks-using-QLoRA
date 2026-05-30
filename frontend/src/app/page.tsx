"use client";

import { useState } from "react";

const DEFAULT_PROMPT = "Ensure latency < 20ms for drone communication";
const DEMO_SCENARIOS = [
  "Ensure latency < 20ms for drone communication",
  "Maximize throughput for enhanced mobile broadband users",
  "Handle PRB scarcity while preserving URLLC reliability",
  "Traffic spike detected in eMBB slice, rebalance resources",
  "Conflicting intent: minimize latency and maximize throughput with limited PRBs",
  "Severe SLA violation in URLLC slice",
];
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export default function Home() {
  const [inputText, setInputText] = useState(DEFAULT_PROMPT);
  const [outputText, setOutputText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  const runInference = async () => {
    setIsLoading(true);
    setError("");

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          intent: inputText,
        }),
      });

      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || body.error || "Request failed");
      }

      const data = (await response.json()) as Record<string, unknown>;
      setOutputText(JSON.stringify(data, null, 2));
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unexpected error";
      setError(message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen px-6 py-10 text-slate-900">
      <div className="mx-auto flex w-full max-w-5xl flex-col gap-8">
        <header className="flex flex-col gap-3">
          <div className="inline-flex items-center gap-2 rounded-full bg-[var(--accent-soft)] px-4 py-1 text-sm font-medium text-slate-700">
            O-RAN Intent Decomposition Demo
          </div>
          <h1 className="text-3xl font-semibold tracking-tight text-slate-900 sm:text-4xl">
            Fine-tuned policy generator
          </h1>
          <p className="max-w-3xl text-base text-slate-600 sm:text-lg">
            This demo API translates raw O-RAN intents into structured policy output for SMO, RIC,
            CU, DU, and RU control layers. Provide the intent to test the model output.
          </p>
        </header>

        <section className="grid gap-6 rounded-3xl border border-[var(--panel-border)] bg-[var(--panel)] p-6 shadow-[0_20px_60px_rgba(15,23,42,0.08)]">
          <div className="grid gap-4 md:grid-cols-[1fr,auto] md:items-center">
            <div>
              <p className="text-sm font-medium text-slate-700">Feedback loop iterations</p>
              <p className="text-2xl font-semibold text-slate-900">1</p>
            </div>
            <div className="rounded-2xl border border-[var(--panel-border)] bg-slate-50 px-4 py-3 text-sm text-slate-600">
              Paste a raw intent and run inference to see the policy output.
            </div>
          </div>

          <div className="grid gap-6 lg:grid-cols-2">
            <div className="flex flex-col gap-3">
              <label className="text-sm font-semibold text-slate-700" htmlFor="input">
                Raw intent input
              </label>
              <textarea
                id="input"
                className="min-h-[220px] w-full resize-none rounded-2xl border border-[var(--panel-border)] bg-white px-4 py-3 text-sm text-slate-900 shadow-sm outline-none transition focus:border-[var(--accent)] focus:ring-2 focus:ring-[var(--accent-soft)]"
                value={inputText}
                onChange={(event) => setInputText(event.target.value)}
              />
              <div className="flex flex-wrap gap-2">
                {DEMO_SCENARIOS.map((scenario) => (
                  <button
                    key={scenario}
                    className="rounded-full border border-[var(--panel-border)] bg-white px-3 py-1 text-xs text-slate-700 shadow-sm transition hover:border-[var(--accent)]"
                    onClick={() => setInputText(scenario)}
                    type="button"
                  >
                    {scenario}
                  </button>
                ))}
              </div>
              <button
                className="inline-flex w-fit items-center gap-2 rounded-full bg-[var(--accent)] px-6 py-2 text-sm font-semibold text-white shadow-md transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-60"
                onClick={runInference}
                disabled={isLoading}
              >
                {isLoading ? "Running..." : "Generate policy"}
              </button>
              {error ? <p className="text-sm text-red-600">{error}</p> : null}
            </div>

            <div className="flex flex-col gap-3">
              <label className="text-sm font-semibold text-slate-700" htmlFor="output">
                Model output
              </label>
              <pre
                id="output"
                className="min-h-[220px] w-full whitespace-pre-wrap rounded-2xl border border-[var(--panel-border)] bg-slate-50 px-4 py-3 text-xs text-slate-900 shadow-sm outline-none"
              >
                {outputText || "Generated policy output will appear here."}
              </pre>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

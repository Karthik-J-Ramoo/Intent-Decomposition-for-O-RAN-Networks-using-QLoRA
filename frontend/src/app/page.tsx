"use client";

import { useState } from "react";

const DEFAULT_PROMPT =
  "Reduce URLLC latency below 15ms while maintaining eMBB throughput above 40 Mbps.";
const DEFAULT_NETWORK_STATE = `{
  "total_prbs": 103,
  "slices": {
    "URLLC": {
      "queue_size": 104,
      "latency_ms": 27.91,
      "throughput": 40.43,
      "allocated_prbs": 45
    },
    "eMBB": {
      "queue_size": 124,
      "latency_ms": 31.4,
      "throughput": 65.47,
      "allocated_prbs": 39
    },
    "IoT": {
      "queue_size": 55,
      "latency_ms": 21.82,
      "throughput": 19.93,
      "allocated_prbs": 19
    }
  }
}`;
const DEFAULT_SLA_STATUS = `{
  "URLLC": "violated",
  "eMBB": "satisfied",
  "IoT": "satisfied"
}`;

export default function Home() {
  const [inputText, setInputText] = useState(DEFAULT_PROMPT);
  const [networkStateText, setNetworkStateText] = useState(DEFAULT_NETWORK_STATE);
  const [slaStatusText, setSlaStatusText] = useState(DEFAULT_SLA_STATUS);
  const [outputText, setOutputText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  const runInference = async () => {
    setIsLoading(true);
    setError("");

    try {
      JSON.parse(networkStateText);
      JSON.parse(slaStatusText);

      const response = await fetch("/api/infer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          rawText: inputText,
          networkState: networkStateText,
          slaStatus: slaStatusText,
        }),
      });

      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.error || "Request failed");
      }

      const data = (await response.json()) as { output: string };
      setOutputText(data.output || "");
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
            This fine-tuned LLM translates raw O-RAN intents plus network context into structured
            policy output for SMO, RIC, CU, DU, and RU control layers. Provide the intent,
            network state, and SLA status to test the model output.
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
              <textarea
                id="output"
                className="min-h-[220px] w-full resize-none rounded-2xl border border-[var(--panel-border)] bg-slate-50 px-4 py-3 text-sm text-slate-900 shadow-sm outline-none"
                value={outputText}
                readOnly
                placeholder="Generated policy output will appear here."
              />
            </div>
          </div>

          <div className="grid gap-6 lg:grid-cols-2">
            <div className="flex flex-col gap-3">
              <label className="text-sm font-semibold text-slate-700" htmlFor="network-state">
                Network state (JSON)
              </label>
              <textarea
                id="network-state"
                className="min-h-[220px] w-full resize-none rounded-2xl border border-[var(--panel-border)] bg-white px-4 py-3 text-sm text-slate-900 shadow-sm outline-none transition focus:border-[var(--accent)] focus:ring-2 focus:ring-[var(--accent-soft)]"
                value={networkStateText}
                onChange={(event) => setNetworkStateText(event.target.value)}
              />
            </div>

            <div className="flex flex-col gap-3">
              <label className="text-sm font-semibold text-slate-700" htmlFor="sla-status">
                SLA status (JSON)
              </label>
              <textarea
                id="sla-status"
                className="min-h-[220px] w-full resize-none rounded-2xl border border-[var(--panel-border)] bg-white px-4 py-3 text-sm text-slate-900 shadow-sm outline-none transition focus:border-[var(--accent)] focus:ring-2 focus:ring-[var(--accent-soft)]"
                value={slaStatusText}
                onChange={(event) => setSlaStatusText(event.target.value)}
              />
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

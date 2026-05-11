# Synthetic O-RAN Intent Decomposition Dataset Generator

This project generates a synthetic dataset for LLM training on intent-based policy decomposition in programmable 5G/O-RAN environments.

Each sample maps:

Intent + Network State -> Multi-layer O-RAN Policy Output

## Features

- Strict schema-compliant JSON samples
- 3 slices: URLLC, eMBB, IoT
- Deterministic reproducibility with configurable seed (default 42)
- Scenario-aware network simulation with 17 edge-case types + NORMAL
- Rule-based policy meta-scheduler across SMO, RIC, CU, DU, and RU
- Built-in schema and distribution validation
- CLI with dataset size and output path options

## Project Structure

- generate_dataset.py: CLI entrypoint
- src/oran_dataset_generator/generator.py: Core generation logic
- src/oran_dataset_generator/__init__.py: Package exports
- dataset.json: Generated output file (created after run)
- requirements_sft.txt: SFT dependencies
- sft_common.py: Shared utilities for SFT
- sft_train.py: Local SFT training script
- sft_eval.py: Interactive/single-prompt inference

## Dataset Schema

Every sample is generated with the exact top-level shape:

- sample_id
- scenario_type
- intent
- network_state
- sla_status
- policy_output

Nested fields match the required schema with:

- intent.constraints.latency_ms as required float
- intent.constraints.throughput as optional float
- network_state.slices containing URLLC/eMBB/IoT metrics
- policy_output with SMO, RIC, CU, DU, RU sections

## SLA Logic

- URLLC SLA satisfied if latency_ms < 20
- eMBB SLA satisfied if throughput > 30
- IoT SLA relaxed: satisfied if latency_ms < 180 and throughput > 3

## Policy Logic

The generator implements a rule-based meta scheduler:

- Detect SLA violations from current state
- Increase PRBs for violated slices
- Decrease PRBs from non-violated slices when needed
- Enforce PRB sum conservation and minimum slice allocation

Hard constraints:

- sum(RIC.prb_allocation) == network_state.total_prbs
- minimum PRBs per slice >= 5

Scheduler selection:

- priority when URLLC is critical
- proportional_fair otherwise

## Scenario Catalog

NORMAL:

- NORMAL

Edge cases:

- full_congestion
- single_slice_overload
- burst_spike
- idle_network
- prb_scarcity
- excess_resources
- near_threshold
- slight_violation
- severe_violation
- conflicting_intents
- multi_priority
- starvation
- equal_load
- noisy_metrics
- inconsistent_state
- retransmission_failure
- slice_failure

## Distribution

Default dataset size is 2000 samples.

- 40% NORMAL
- 60% edge cases distributed as evenly as possible across 17 edge scenarios

For 2000 samples:

- NORMAL = 800
- Edge total = 1200
- Edge distribution = 70 or 71 per scenario (remainder spread deterministically before shuffling)

## Usage

1. Install dependency:

```bash
python -m pip install -r requirements.txt
```

2. Generate default dataset (2000 samples):

```bash
python generate_dataset.py --size 2000 --output dataset.json
```

3. Example custom run:

```bash
python generate_dataset.py --size 500 --output data/dataset_500.json --seed 42 --progress-every 50
```

## SFT Training (Local)

Install SFT dependencies:

```bash
python -m pip install -r requirements_sft.txt
```

Run a small debug train (recommended on limited hardware):

```bash
python sft_train.py --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 --max-train-samples 50 --max-eval-samples 20 --num-epochs 1
```

Run full training:

```bash
python sft_train.py --model-id mistralai/mistral-7b-v0.1
```

Optional: set `HF_TOKEN` for private datasets or pushing to the Hub.

### Single-Prompt Inference

Run one prompt at a time (faster than full eval):

```bash
python sft_eval.py --model-path ./mistral7b_oran_sft --raw-text "Reduce URLLC latency below 15ms while maintaining eMBB throughput."
```

Interactive mode:

```bash
python sft_eval.py --model-path ./mistral7b_oran_sft --interactive
```

Windows note: if you see `charmap` decode errors in TRL, run with UTF-8 mode:

```bash
$env:PYTHONUTF8="1"; & "c:\\.Code\\.Projects\\Intent Decomposition Dataset\\.venv\\Scripts\\python.exe" sft_train.py --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 --max-train-samples 50 --max-eval-samples 20 --num-epochs 1
```

## Output

The output file is pretty-printed JSON and fully validated before completion.
If validation fails, generation exits with an error.

## Notes

- The script is designed for realistic synthetic behavior, not placeholder values.
- Randomness uses both Python random and NumPy random.
- UUID v4 is used for sample_id.

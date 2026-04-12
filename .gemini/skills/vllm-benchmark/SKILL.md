---
name: vllm-benchmark
description: Use this skill to benchmark KV cache performance vs. no-KV cache performance using the benchmark.py tool. It supports running tests with specific hardware-based constraints for CPU and GPU to ensure efficient execution. No manual file editing should occur when using this skill, as the tool automatically writes results to an output CSV.
---

# VLLM Benchmark

Use this skill to measure the performance of the mini-vllm implementation.

## Critical Constraint

- **DO NOT** edit any source files (e.g., `model.py`, `kv_cache.py`, `block_allocator.py`).
- The benchmark results are automatically written to a CSV file (default: `benchmark_results.csv`).
- Do not manually create or edit the output file.

## Hardware Constraints

Adhere to the following limits based on the execution device:

| Hardware | Max New Tokens (`--new-tokens`) | Max Prompt Length |
| :--- | :--- | :--- |
| **CPU** | 50 - 250 | Under 100 characters |
| **GPU** | Up to 500 | Up to 200 characters |

## Usage

Run the benchmark using `benchmark.py` with the following parameters:

| Parameter | Flag | Default | Description |
| :--- | :--- | :--- | :--- |
| **Output** | `--output` | `benchmark_results.csv` | Path to the results CSV. |
| **Prompt** | `--prompt` | `"Hi my name is"` | The text prompt for benchmarking. |
| **New Tokens** | `--new-tokens` | `200` | Number of tokens to generate. |
| **Total Blocks** | `--total-blocks` | `128` | Max number of KV cache blocks. |
| **Block Size** | `--block-size` | `16` | Tokens per KV cache block. |
| **Warmup Runs** | `--warmup-runs` | `1` | Number of initial untimed runs. |
| **Timed Runs** | `--timed-runs` | `3` | Number of timed runs to average. |

### Example Command (CPU)

```bash
python benchmark.py --prompt "What is the capital of France?" --new-tokens 150 --timed-runs 3
```

## Workflow

1.  **Identify Hardware**: Check if the system is running on CPU or GPU.
2.  **Verify Constraints**: Ensure your prompt and token count fit the hardware limits.
3.  **Run Command**: Execute `python benchmark.py [FLAGS]` with the desired parameters.
4.  **Analyze Results**: Review the CLI output for prompt tokens, tokens/sec, and speedup.
5.  **Confirm Output**: Verify that the results were appended to the output CSV file.

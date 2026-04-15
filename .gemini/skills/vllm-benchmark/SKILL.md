---
name: vllm-benchmark
description: Use this skill to benchmark KV cache performance (paged and non-paged) vs. no-KV cache performance using the benchmark.py tool. It supports running tests with specific hardware-based constraints for CPU and GPU to ensure efficient execution. No manual file editing should occur when using this skill, as the tool automatically writes results to an output CSV.
---

# VLLM Benchmark

Use this skill to measure the performance of the mini-vllm implementation, comparing paged KV cache, non-paged KV cache, and baseline generation.

## Critical Constraint

- **DO NOT** edit any source files (e.g., `model.py`, `kv_cache.py`, `block_allocator.py`).
- The benchmark results are automatically written to a CSV file (defined in `benchmarking/config.py`).
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
| **Output** | `--output` | See config | Path to the results CSV. |
| **Prompt** | `--prompt` | `"Hi my name is"` | The text prompt for benchmarking. |
| **New Tokens** | `--new-tokens` | `200` | Number of tokens to generate. |
| **Total Blocks** | `--total-blocks` | `128` | Max number of KV cache blocks (paged). |
| **Block Size** | `--block-size` | `16` | Tokens per KV cache block (paged). |
| **Non-Paged Block Size** | `--non-paged-block-size` | Auto | Contiguous block size for non-paged mode. |
| **Warmup Runs** | `--warmup-runs` | `1` | Number of initial untimed runs. |
| **Timed Runs** | `--timed-runs` | `3` | Number of timed runs to average. |

### Evaluation Modes

1.  **`kv_cache`**: Paged KV cache implementation (the primary focus).
2.  **`kv_cache_no_paging`**: Standard KV cache using a single contiguous block.
3.  **`no_kv_cache`**: Baseline generation without any caching.

### Example Command (CPU)

```bash
$env:PYTHONPATH = '.'; python benchmarking/benchmark.py --prompt "What is paged attention?" --new-tokens 150 --timed-runs 3
```

## Workflow

1.  **Identify Hardware**: Check if the system is running on CPU or GPU.
2.  **Verify Constraints**: Ensure your prompt and token count fit the hardware limits.
3.  **Run Command**: Execute `python benchmarking/benchmark.py [FLAGS]` with the desired parameters. Ensure `PYTHONPATH` is set to the root directory.
4.  **Analyze Results**: Review the CLI output for prompt tokens, tokens/sec, and speedups for both paged and non-paged modes.
5.  **Confirm Output**: Verify that the results were appended to the output CSV file.

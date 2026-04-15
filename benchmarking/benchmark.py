from __future__ import annotations

import csv
import json
import statistics
import time
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import GPT2Tokenizer

from inference_engine import InferenceEngine
from benchmarking.config import BENCHMARK_RESULTS_CSV_PATH
from benchmarking.nanogpt_model import GPT as NanoGPT


@torch.no_grad()
def generate_without_kv_cache(model: Any, tokenizer: GPT2Tokenizer, prompt: str, max_new_tokens: int) -> torch.Tensor:
	idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0)
	return model.generate(idx, max_new_tokens)


def benchmark_case(label: str, fn, warmup_runs: int = 1, timed_runs: int = 3):
	for _ in range(warmup_runs):
		fn()

	durations = []
	for _ in range(timed_runs):
		start = time.perf_counter()
		fn()
		durations.append(time.perf_counter() - start)

	return {
		"label": label,
		"avg_seconds": statistics.mean(durations),
		"stdev_seconds": statistics.stdev(durations) if len(durations) > 1 else 0.0,
		"runs": timed_runs,
		"durations": durations,
	}


def write_results_csv(output_path: Path, rows: list[dict[str, Any]]) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fieldnames = [
		"timestamp_utc",
		"benchmark_label",
		"execution_device",
		"cuda_available",
		"mode",
		"prompt",
		"prompt_tokens",
		"new_tokens",
		"total_tokens_processed",
		"total_blocks",
		"block_size",
		"warmup_runs",
		"timed_runs",
		"avg_seconds",
		"stdev_seconds",
		"min_seconds",
		"max_seconds",
		"median_seconds",
		"tokens_per_second",
		"speedup_vs_no_cache",
		"run_durations_json",
	]

	file_exists = output_path.exists()
	with output_path.open("a", newline="", encoding="utf-8") as csv_file:
		writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
		if not file_exists:
			writer.writeheader()
		writer.writerows(rows)


def build_summary_row(
	mode: str,
	execution_device: str,
	cuda_available: bool,
	result: dict[str, Any],
	prompt: str,
	prompt_tokens: int,
	new_tokens: int,
	total_blocks: int,
	block_size: int,
	timestamp_utc: str,
	speedup_vs_no_cache: float,
) -> dict[str, Any]:
	durations = result["durations"]
	return {
		"timestamp_utc": timestamp_utc,
		"benchmark_label": "kv_cache_mode_comparison",
		"execution_device": execution_device,
		"cuda_available": cuda_available,
		"mode": mode,
		"prompt": prompt,
		"prompt_tokens": prompt_tokens,
		"new_tokens": new_tokens,
		"total_tokens_processed": prompt_tokens + new_tokens,
		"total_blocks": total_blocks,
		"block_size": block_size,
		"warmup_runs": result["warmup_runs"],
		"timed_runs": result["runs"],
		"avg_seconds": result["avg_seconds"],
		"stdev_seconds": result["stdev_seconds"],
		"min_seconds": min(durations),
		"max_seconds": max(durations),
		"median_seconds": statistics.median(durations),
		"tokens_per_second": new_tokens / result["avg_seconds"] if result["avg_seconds"] else float("inf"),
		"speedup_vs_no_cache": speedup_vs_no_cache,
		"run_durations_json": json.dumps(durations, separators=(",", ":")),
	}


def main():
	parser = ArgumentParser(description="Benchmark paged KV cache, non-paged KV cache, and no KV cache generation performance.")
	parser.add_argument("--output", type=Path, default=BENCHMARK_RESULTS_CSV_PATH, help="CSV file to write benchmark results to.")
	parser.add_argument("--prompt", type=str, default="Hi my name is", help="Prompt to benchmark.")
	parser.add_argument("--new-tokens", type=int, default=200, help="Number of new tokens to generate.")
	parser.add_argument("--total-blocks", type=int, default=128, help="Number of KV cache blocks available.")
	parser.add_argument("--block-size", type=int, default=16, help="Tokens per KV cache block.")
	parser.add_argument(
		"--non-paged-block-size",
		type=int,
		default=None,
		help="Optional contiguous block size for non-paged KV cache mode. Defaults to prompt_tokens + new_tokens + 2.",
	)
	parser.add_argument("--warmup-runs", type=int, default=1, help="Warmup runs before timing.")
	parser.add_argument("--timed-runs", type=int, default=3, help="Timed runs for each benchmark mode.")
	args = parser.parse_args()

	torch.manual_seed(1337)

	prompt = args.prompt
	max_new_tokens = args.new_tokens
	total_blocks = args.total_blocks
	block_size = args.block_size

	engine = InferenceEngine(total_blocks, block_size, max_tokens=max_new_tokens)
	engine.model.eval()
	tokenizer = engine.tokenizer
	prompt_tokens = len(tokenizer.encode(prompt))
	non_paged_block_size = args.non_paged_block_size or (prompt_tokens + max_new_tokens + 2)
	non_paged_engine = InferenceEngine(1, non_paged_block_size, max_tokens=max_new_tokens)
	non_paged_engine.model.eval()
	baseline_model = NanoGPT.from_pretrained("gpt2")
	baseline_model.eval()
	execution_device = next(engine.model.parameters()).device.type
	cuda_available = torch.cuda.is_available()

	cache_result = benchmark_case(
		"kv_cache",
		lambda: engine.generate(prompt),
		warmup_runs=args.warmup_runs,
		timed_runs=args.timed_runs,
	)
	non_paged_cache_result = benchmark_case(
		"kv_cache_no_paging",
		lambda: non_paged_engine.generate(prompt),
		warmup_runs=args.warmup_runs,
		timed_runs=args.timed_runs,
	)
	no_cache_result = benchmark_case(
		"no_kv_cache",
		lambda: generate_without_kv_cache(baseline_model, tokenizer, prompt, max_new_tokens),
		warmup_runs=args.warmup_runs,
		timed_runs=args.timed_runs,
	)

	total_generated_tokens = prompt_tokens + max_new_tokens
	timestamp_utc = datetime.now(timezone.utc).isoformat()

	print(f"Prompt tokens: {prompt_tokens}")
	print(f"New tokens: {max_new_tokens}")
	print(f"Total tokens processed per run: {total_generated_tokens}")
	print()

	for result in (cache_result, non_paged_cache_result, no_cache_result):
		tokens_per_second = max_new_tokens / result["avg_seconds"]
		print(f"{result['label']}: {result['avg_seconds']:.4f}s avg over {result['runs']} runs  |  {tokens_per_second:.2f} new tokens/s")

	speedup = no_cache_result["avg_seconds"] / cache_result["avg_seconds"]
	non_paged_speedup = no_cache_result["avg_seconds"] / non_paged_cache_result["avg_seconds"]
	print()
	print(f"KV cache speedup: {speedup:.2f}x")
	print(f"KV cache (no paging) speedup: {non_paged_speedup:.2f}x")

	csv_rows = [
		build_summary_row(
			"kv_cache",
			execution_device,
			cuda_available,
			{**cache_result, "warmup_runs": args.warmup_runs},
			prompt,
			prompt_tokens,
			max_new_tokens,
			total_blocks,
			block_size,
			timestamp_utc,
			speedup,
		),
		build_summary_row(
			"kv_cache_no_paging",
			execution_device,
			cuda_available,
			{**non_paged_cache_result, "warmup_runs": args.warmup_runs},
			prompt,
			prompt_tokens,
			max_new_tokens,
			1,
			non_paged_block_size,
			timestamp_utc,
			non_paged_speedup,
		),
		build_summary_row(
			"no_kv_cache",
			execution_device,
			cuda_available,
			{**no_cache_result, "warmup_runs": args.warmup_runs},
			prompt,
			prompt_tokens,
			max_new_tokens,
			total_blocks,
			block_size,
			timestamp_utc,
			1.0,
		),
	]

	write_results_csv(args.output, csv_rows)
	print(f"\nWrote CSV results to: {args.output}")


if __name__ == "__main__":
	main()

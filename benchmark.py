from __future__ import annotations

import csv
import json
import statistics
import time
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2Tokenizer

from block_allocator import BlockAllocator
from kv_cache import KVCache
from model import GPT


def _prepare_attention_cache(model: GPT, kv_cache: KVCache, allocator: BlockAllocator, request_id: int, token_pos: int) -> None:
	transformer: Any = model.transformer
	for layer_idx, block in enumerate(transformer.h):
		block.attn.layer_idx = layer_idx
		block.attn.request_id = request_id
		block.attn.kv_cache = kv_cache
		block.attn.block_list = allocator.block_table[request_id]
		block.attn.token_pos = token_pos


def _sample_next_token(logits: torch.Tensor) -> torch.Tensor:
	return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)


def _forward_without_kv_cache(model: GPT, idx: torch.Tensor) -> torch.Tensor:
	device = idx.device
	batch_size, seq_len = idx.size()
	assert seq_len <= model.config.block_size, f"Cannot forward sequence of length {seq_len}, block size is only {model.config.block_size}"

	pos = torch.arange(0, seq_len, dtype=torch.long, device=device)
	transformer: Any = model.transformer
	transformer_wte = transformer.wte
	transformer_wpe = transformer.wpe
	transformer_drop = transformer.drop
	transformer_ln_f = transformer.ln_f
	blocks = transformer.h

	x = transformer_drop(transformer_wte(idx) + transformer_wpe(pos))

	for block in blocks:
		residual = x
		x = block.ln_1(x)

		query, key, value = block.attn.c_attn(x).split(block.attn.n_embd, dim=2)
		key = key.view(batch_size, seq_len, block.attn.n_head, block.attn.n_embd // block.attn.n_head).transpose(1, 2)
		query = query.view(batch_size, seq_len, block.attn.n_head, block.attn.n_embd // block.attn.n_head).transpose(1, 2)
		value = value.view(batch_size, seq_len, block.attn.n_head, block.attn.n_embd // block.attn.n_head).transpose(1, 2)

		if block.attn.flash:
			attended = torch.nn.functional.scaled_dot_product_attention(
				query,
				key,
				value,
				attn_mask=None,
				dropout_p=block.attn.dropout if block.training else 0,
				is_causal=True,
			)
		else:
			attention_scores = (query @ key.transpose(-2, -1)) * (1.0 / (key.size(-1) ** 0.5))
			attention_scores = attention_scores.masked_fill(block.attn.bias[:, :, :seq_len, :seq_len] == 0, float("-inf"))
			attention_scores = F.softmax(attention_scores, dim=-1)
			attention_scores = block.attn.attn_dropout(attention_scores)
			attended = attention_scores @ value

		attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, block.attn.n_embd)
		x = residual + block.attn.resid_dropout(block.attn.c_proj(attended))
		x = x + block.mlp(block.ln_2(x))

	x = transformer_ln_f(x)
	return model.lm_head(x[:, [-1], :])


@torch.no_grad()
def generate_with_kv_cache(model: GPT, tokenizer: GPT2Tokenizer, prompt: str, max_new_tokens: int, total_blocks: int, block_size: int) -> torch.Tensor:
	idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0)
	request_id = 0
	num_prompt_tokens = idx.shape[1]

	allocator = BlockAllocator(total_blocks, block_size)
	kv_cache = KVCache(total_blocks, block_size, model.config.n_layer, model.config.n_head, model.config.n_embd // model.config.n_head)

	allocator.allocate_prefill(request_id, num_prompt_tokens)
	token_pos = num_prompt_tokens - 1
	_prepare_attention_cache(model, kv_cache, allocator, request_id, token_pos)

	logits, _ = model(idx)
	next_token = _sample_next_token(logits)
	idx = torch.cat((idx, next_token), dim=1)

	generated = 1
	while generated < max_new_tokens:
		allocator.allocate_decode(request_id)
		token_pos += 1
		_prepare_attention_cache(model, kv_cache, allocator, request_id, token_pos)
		logits, _ = model(next_token, position_offset=token_pos)
		next_token = _sample_next_token(logits)
		idx = torch.cat((idx, next_token), dim=1)
		generated += 1

	return idx


@torch.no_grad()
def generate_without_kv_cache(model: GPT, tokenizer: GPT2Tokenizer, prompt: str, max_new_tokens: int) -> torch.Tensor:
	idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0)

	for _ in range(max_new_tokens):
		logits = _forward_without_kv_cache(model, idx)
		next_token = _sample_next_token(logits)
		idx = torch.cat((idx, next_token), dim=1)

	return idx


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
		"benchmark_label": "kv_cache_vs_no_kv_cache",
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
	parser = ArgumentParser(description="Benchmark KV cache vs no KV cache generation performance.")
	parser.add_argument("--output", type=Path, default=Path("benchmark_results.csv"), help="CSV file to write benchmark results to.")
	parser.add_argument("--prompt", type=str, default="Hi my name is", help="Prompt to benchmark.")
	parser.add_argument("--new-tokens", type=int, default=200, help="Number of new tokens to generate.")
	parser.add_argument("--total-blocks", type=int, default=128, help="Number of KV cache blocks available.")
	parser.add_argument("--block-size", type=int, default=16, help="Tokens per KV cache block.")
	parser.add_argument("--warmup-runs", type=int, default=1, help="Warmup runs before timing.")
	parser.add_argument("--timed-runs", type=int, default=3, help="Timed runs for each benchmark mode.")
	args = parser.parse_args()

	torch.manual_seed(1337)

	prompt = args.prompt
	max_new_tokens = args.new_tokens
	total_blocks = args.total_blocks
	block_size = args.block_size

	tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
	model = GPT.from_pretrained("gpt2")
	model.eval()
	execution_device = next(model.parameters()).device.type
	cuda_available = torch.cuda.is_available()

	cache_result = benchmark_case(
		"kv_cache",
		lambda: generate_with_kv_cache(model, tokenizer, prompt, max_new_tokens, total_blocks, block_size),
		warmup_runs=args.warmup_runs,
		timed_runs=args.timed_runs,
	)
	no_cache_result = benchmark_case(
		"no_kv_cache",
		lambda: generate_without_kv_cache(model, tokenizer, prompt, max_new_tokens),
		warmup_runs=args.warmup_runs,
		timed_runs=args.timed_runs,
	)

	prompt_tokens = len(tokenizer.encode(prompt))
	total_generated_tokens = prompt_tokens + max_new_tokens
	timestamp_utc = datetime.now(timezone.utc).isoformat()

	print(f"Prompt tokens: {prompt_tokens}")
	print(f"New tokens: {max_new_tokens}")
	print(f"Total tokens processed per run: {total_generated_tokens}")
	print()

	for result in (cache_result, no_cache_result):
		tokens_per_second = max_new_tokens / result["avg_seconds"]
		print(f"{result['label']}: {result['avg_seconds']:.4f}s avg over {result['runs']} runs  |  {tokens_per_second:.2f} new tokens/s")

	speedup = no_cache_result["avg_seconds"] / cache_result["avg_seconds"]
	print()
	print(f"KV cache speedup: {speedup:.2f}x")

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

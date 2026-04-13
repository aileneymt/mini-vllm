# mini-vllm

> A minimal LLM inference engine implementing PagedAttention-style KV cache management on NanoGPT. Based on the "Efficient Memory Management for Large Language Model Serving with PagedAttention" paper.

---

## Description and Motivation

In transformers, each step of token generation requires the computation of query, key, and value tensors for self attention. The key and value tensors must be calculated with respect to all the previous tokens in the input, which scales quadratically if done from scratch for every step. 

To save on computation, KV caches store computed KV tensors, allowing new iterations in the autoregressive stage to only compute a single key and value tensor for the new token generated in the preceding step. However, these KV caches for requests are typically stored in contiguous memory space. Because they grow over their lifetime, this requires that enough memory is initially reserved for the maximum possible generated sequence length. This leads to significant amounts of internal and external fragmentation that waste memory for an already memory-limited process.

Implementing a virtual memory and paging system similar to how the OS manages memory allows KV tensors to be stored in blocks that can be allocated dynamically, do not need to be contiguous, and can be shared between requests. This reduces memory waste significantly; the original vLLM paper reported 2-4x improvements over systems using contiguous allocation.

---

## Architecture

The engine is composed of three core components:

**BlockAllocator**: Maintains state of blocks, allocates for prefill and decode stages, and frees once requests are complete.

![image](https://github.com/aileneymt/mini-vllm/blob/main/images/block_allocator_diagram.png)

**KVCache**: Uses the block table managed by BlockAllocator to determine where in the pre-allocated GPU tensor to read and write KV tensors.

![image](https://github.com/aileneymt/mini-vllm/blob/main/images/kv_cache_diagram1.png)
![image](https://github.com/aileneymt/mini-vllm/blob/main/images/kv_cache_diagram2.png)

**InferenceEngine**: The orchestrator; receives a prompt, runs prefill, decode loop, frees blocks, and returns output.

![images](https://github.com/aileneymt/mini-vllm/blob/main/images/inference_engine_diagram.png)


---


### Benchmarking

The benchmark.py script evaluates the performance impact of the KV cache implementation by comparing two generation methods:

   1. KV Cache (Optimized): Uses the InferenceEngine to generate tokens. This method leverages the custom KVCache and BlockAllocator to avoid redundant computations for previously processed tokens. During the autoregressive/decode phase, self-attention is computed using only one token (the output of the preceding forward pass).
   2. No KV Cache (Baseline): Uses the standard NanoGPT implementation to generate tokens. In this mode, the model performs a full forward pass over the entire growing sequence (prompt + all generated tokens) for every single new token, which is computationally expensive as the sequence length increases.

Key Features
   * Metric Tracking: Measures average generation time, tokens per second (TPS), and calculates the specific speedup ratio provided by the KV cache.
   * Configurable Parameters: Allows testing across different prompt lengths, token counts, and memory configurations (block size and total blocks).
   * Automated Reporting: Appends timing data to a CSV file (benchmark_results.csv) for tracking and analysis.

The [benchmark_results_viz.ipynb](https://github.com/aileneymt/mini-vllm/blob/main/benchmarking/benchmark_results_viz.ipynb) notebook visualizes the data from the output CSV. It graphs Average Runtime vs Generated Tokens, KV Cache Speedup by Trial, and Throughput by Trial (tokens/sec).

On CPU, the KV cache implementation typically provides a **2x to 10x** speedup for moderate generation lengths (200-500 tokens), significantly reducing latency as the sequence grows.

---

### Roadmap

The project has been fully implemented in its most minimal form. Future additions include rewriting BlockAllocator in C++ and enabling prefix-sharing so blocks can be shared between requests.

---

## References

- [PagedAttention paper](https://arxiv.org/abs/2309.06180)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [NanoGPT](https://github.com/karpathy/nanoGPT)

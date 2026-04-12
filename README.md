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

**BlockAllocator** — Maintains state of blocks, allocates for prefill and decode stages, and frees once requests are complete.

![image](https://github.com/aileneymt/mini-vllm/blob/main/images/block_allocator_diagram.png)

**KVCache** — Uses the block table managed by BlockAllocator to determine where in the pre-allocated GPU tensor to read and write KV tensors.

![image](https://github.com/aileneymt/mini-vllm/blob/main/images/kv_cache_diagram1.png)
![image](https://github.com/aileneymt/mini-vllm/blob/main/images/kv_cache_diagram2.png)

**InferenceEngine** — Orchestrator: receives a prompt, runs prefill, decode loop, frees blocks, and returns output.

```
prompt
  └── InferenceEngine
        ├── BlockAllocator   (manages physical memory blocks)
        └── KVCache          (stores/retrieves KV tensors per block)
```


---

### Roadmap

Currently focusing on implementing the project in its most minimal form. Once functioning, I plan on rewriting the BlockAllocator in C++ and enabling prefix-sharing so blocks can be shared between requests.


---

### Benchmarking

Will compare against PyTorch's native caching (which allocates memory contiguously) after implementation complete.

---

## References

- [PagedAttention paper](https://arxiv.org/abs/2309.06180)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [GPT-2 — Hugging Face](https://huggingface.co/gpt2)

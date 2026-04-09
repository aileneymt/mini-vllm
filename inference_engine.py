from model import GPT, GPTConfig
from kv_cache import KVCache
from block_allocator import BlockAllocator

class InferenceEngine:
    # creates GPT object
    # takes prompt, runs prefill, decode loop
    # uses KVCache to manage the cache 
    # can handle multiple requests
    def __init__(self, total_blocks: int, block_size: int):
        self.config = GPTConfig()
        self.model = GPT(self.config)
        self.allocator = BlockAllocator(total_blocks, block_size)
        self.kv_cache = KVCache(total_blocks, block_size, self.config.n_layer, self.config.n_head, self.config.n_embd // self.config.n_head)
        self.requests = dict[int, int] # map request id -> number of tokens in the request

    def __prepare_attention(self, request_id: int):
        for i, dec_block in enumerate(self.model.transformer.h):
            dec_block.attn.layer_idx = i
            dec_block.attn.request_id = request_id
            dec_block.attn.kv_cache = self.kv_cache
            dec_block.attn.block_list = self.allocator.block_table[request_id]
            dec_block.attn.token_pos = self.requests[request_id]

        return None
    
    def generate(self, prompt):
        # handle prefill and decode steps
        return None
    
    
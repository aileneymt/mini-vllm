import torch
from block_allocator import BlockAllocator

class KVCache:
    # holds the KV tensors
    # uses BlockAllocator to know where to store them
    # write and read from the KV cache
    def __init__(self, allocator: BlockAllocator, total_blocks: int, block_size: int, num_layers: int, num_heads: int, head_dim: int):
        shape = (total_blocks, block_size, num_heads, head_dim)
        self.allocator = allocator
        self.k_cache = [torch.zeros(shape, device='cuda') for _ in range(num_layers)]
        self.v_cache = [torch.zeros(shape, device='cuda') for _ in range(num_layers)]
    '''
    Take new key/value tensors for a SINGLE TOKEN
    start_pos captures the offset in the entire k_cache, not just starting from a single block
    token_pos is the token's index in its own request
    '''
    def write(self, request_id: int, layer_idx: int, token_pos: int, key, val):
        # key and val shape [12 (num_heads), 64 (head_dim)] since its for ONE token
        if not self.allocator.request_exists(request_id):
            raise LookupError(f"Request {request_id} does not exist")
        
        block_size = len(self.k_cache[0][0])
        logical_block_idx = token_pos // block_size
        phys_block_idx = self.allocator.block_table[request_id][logical_block_idx]

        offset = token_pos % block_size
        self.k_cache[layer_idx][phys_block_idx][offset] = key
        self.v_cache[layer_idx][phys_block_idx][offset] = val
    
    def read(self, request_id: int, layer_idx: int, token_pos: int):
        if not self.allocator.request_exists(request_id):
            raise LookupError(f"Request {request_id} does not exist")
        
        block_size = len(self.k_cache[0][0])
        logical_block_idx = token_pos // block_size
        phys_block_idx = self.allocator.block_table[request_id][logical_block_idx]
        offset = token_pos % block_size

        return (self.k_cache[layer_idx][phys_block_idx][offset], self.v_cache[layer_idx][phys_block_idx][offset])
    
    '''
    Will be used for writing prefill. instead of writing one token's key val tensors at a time, this will try to fill one block at a time
    delegates to write() if necessary
    '''
    def write_batch(self, request_id: int, layer_idx: int, key, val):
        if not self.allocator.request_exists(request_id):
            raise LookupError(f"Request {request_id} does not exist")
        
        block_size = len(self.k_cache[0][0])
        block_list = self.allocator.block_table[request_id]
        token_pos = 0
        num_tokens = len(key)
        while token_pos < num_tokens:
            logical_block_idx = token_pos // block_size
            phys_block_idx = block_list[logical_block_idx]
            tokens_in_block = min(num_tokens - token_pos, block_size)
            start, end = logical_block_idx*block_size, logical_block_idx*block_size + tokens_in_block
            
            self.k_cache[layer_idx][phys_block_idx][:tokens_in_block] = key[start:end]
            self.v_cache[layer_idx][phys_block_idx][:tokens_in_block] = val[start:end]
            token_pos += tokens_in_block

    def read_batch(self, request_id: int, layer_idx: int):
        if not self.allocator.request_exists(request_id):
            raise LookupError(f"Request {request_id} does not exist")
        
        k_list = []
        v_list = []
        block_size = len(self.k_cache[0][0])
        block_list = self.allocator.block_table[request_id]

        for log_block_idx, phys_block_id in enumerate(block_list):
            tokens_in_block = self.allocator.all_blocks[phys_block_id].num_filled
            end = log_block_idx * block_size + tokens_in_block
            k_list.append(self.k_cache[layer_idx][phys_block_id][:end])
            v_list.append(self.v_cache[layer_idx][phys_block_id][:end])
        k_full = torch.cat(k_list, dim=0)
        v_full = torch.cat(v_list, dim=0)
        return (k_full, v_full)


import torch
from block_allocator import BlockAllocator

class KVCache:
    # holds the KV tensors
    # uses BlockAllocator to know where to store them
    # write and read from the KV cache
    def __init__(self, total_blocks: int, block_size: int, num_layers: int, num_heads: int, head_dim: int):
        shape = (total_blocks, block_size, num_heads, head_dim)
        self.k_cache = [torch.zeros(shape, device='cuda') for _ in range(num_layers)]
        self.v_cache = [torch.zeros(shape, device='cuda') for _ in range(num_layers)]
    '''
    Take new key/value data for a request and a layer

    '''
    def write(self, request_id: int, layer_idx: int, block_ids: list[int], start_offset: int, key, val):
        block_size = len(self.k_cache[0][0])
        for id in block_ids:
            self.k_cache[layer_idx][id][start_offset] = key[:block_size - start_offset]
        return None
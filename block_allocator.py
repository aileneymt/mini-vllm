from math import ceil

class Block:
    # holds the KV tensors for associated tokens
    def __init__(self, block_id:int, size: int):
        self.block_id = block_id
        self.size = size
        self.num_filled = 0 # num tokens we are storing KV tensors for
        self.ref_count = 0
    def is_full(self):
        return self.num_filled == self.size
    def is_shared(self):
        return self.ref_count > 1
    

class BlockAllocator:
    # keeps list of free and used blocks
    # will allocate and free blocks when requests need new blocks and when requests are completed
    def __init__(self, total_blocks: int, block_size: int):
        self.total_blocks = total_blocks
        self.block_size = block_size # tokens per block
        self.all_blocks = [Block(i, block_size) for i in range(total_blocks)]
        self.free_blocks = set(range(total_blocks))

        self.block_table: dict[int, list[int]] = {} # request_id -> list of block_ids in order of allocation (logical order)
    
    # allocate blocks for prompt. return list of block_ids corresponding to the new request
    def allocate_prefill(self, request_id: int, num_tokens: int) -> list[int]:
        blocks_needed = ceil(num_tokens / self.block_size)
        
        if len(self.free_blocks) < blocks_needed:
            raise MemoryError(f"Not enough free blocks for request {request_id}")
        
        self.block_table[request_id] = []
        it = iter(self.free_blocks)
        while num_tokens:
            block_id = next(it)
            alloc_block = self.all_blocks[block_id]
            alloc_block.num_filled = self.block_size if num_tokens >= self.block_size else num_tokens
            alloc_block.ref_count = 1
            num_tokens -= alloc_block.num_filled
            self.block_table[request_id].append(alloc_block.block_id)
        
        self.free_blocks.difference_update(self.block_table[request_id])
        return self.block_table[request_id]

    # return (block_id, slot index) of the newest position allocated
    def allocate_decode(self, request_id: int) -> tuple[int, int]:
        if not self.request_exists(request_id):
            raise LookupError(f"Request {request_id} does not exist")
        '''
        Allocate a new block for this request if:
        - zero blocks allocated so far OR
        - the last block is full OR
        - the last block is not full, but it's ref_count > 1 (sharing with another request)
        '''
        if not self.block_table[request_id] or self.all_blocks[self.block_table[request_id][-1]].is_full() or self.all_blocks[self.block_table[request_id][-1]].is_shared():
            if len(self.free_blocks) == 0:
                raise MemoryError(f"Not enough free blocks for request {request_id}")
            
            it = iter(self.free_blocks)
            block_id = next(it)
            new_block = self.all_blocks[block_id]
            new_block.ref_count = 1
            # if the last block was shared, remove the reference and replace its id in our block table
            
            if self.block_table[request_id] and self.all_blocks[self.block_table[request_id][-1]].ref_count > 1:
                shared_block = self.all_blocks[self.block_table[request_id][-1]]
                shared_block.ref_count -= 1
                new_block.num_filled = shared_block.num_filled
                self.block_table[request_id][-1] = block_id
            else:
                self.block_table[request_id].append(block_id)

            self.free_blocks.remove(block_id)
        
        # increment num_filled for the last block of this request
        last_block_id = self.block_table[request_id][-1]
        self.all_blocks[last_block_id].num_filled += 1
        return (last_block_id, self.all_blocks[last_block_id].num_filled)
    
    # frees the blocks associated with the given request, if they are not being shared
    def free(self, request_id: int) -> bool:
        if not self.request_exists(request_id):
            raise LookupError(f"Request {request_id} does not exist")

        for block_id in self.block_table[request_id]:
            block = self.all_blocks[block_id]
            if block.is_shared():
                block.ref_count -= 1
            else:
                block.num_filled = 0
                block.ref_count = 0
                self.free_blocks.add(block_id)
            
        del self.block_table[request_id]
        return True
    
    def request_exists(self, request_id: int) -> bool:
        return request_id in self.block_table


        
    
        
from model import GPT, GPTConfig
from kv_cache import KVCache
from block_allocator import BlockAllocator
import torch
from transformers import GPT2Tokenizer
from torch.nn import functional as F

class InferenceEngine:
    # creates GPT object
    # takes prompt, runs prefill, decode loop
    # uses KVCache to manage the cache 
    # can handle multiple requests
    def __init__(self, total_blocks: int, block_size: int, max_tokens=50, temp=1.0):
        self.max_tokens = max_tokens
        self.config = GPTConfig()
        self.model = GPT(self.config)
        self.allocator = BlockAllocator(total_blocks, block_size)
        self.kv_cache = KVCache(total_blocks, block_size, self.config.n_layer, self.config.n_head, self.config.n_embd // self.config.n_head)
        self.requests: dict[int, int] = {} # map request id -> number of tokens in the request
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.temp = temp
        self.__reqCtr = 0

    def __prepare_attention(self, request_id: int):
        for i, dec_block in enumerate(self.model.transformer.h):
            dec_block.attn.layer_idx = i
            dec_block.attn.request_id = request_id
            dec_block.attn.kv_cache = self.kv_cache
            dec_block.attn.block_list = self.allocator.block_table[request_id]
            dec_block.attn.token_pos = self.requests[request_id]
    
    def generate(self, prompt):
        # handle prefill and decode steps
        idx = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0)
        req_id = self.__reqCtr
        self.__reqCtr += 1
        num_tokens_prompt = idx.shape[1]
        self.requests[req_id] = num_tokens_prompt - 1

        # prefill stage
        self.allocator.allocate_prefill(req_id, num_tokens_prompt)
        self.__prepare_attention(req_id)
        logits, _ = self.model(idx)
        idx_next = self.__sample(logits)
        idx = torch.cat((idx, idx_next), dim=1)

        # decode stage
        while self.requests[req_id] - num_tokens_prompt < self.max_tokens:
            self.allocator.allocate_decode(req_id)
            self.requests[req_id] += 1
            self.__prepare_attention(req_id)
            logits, _ = self.model(idx_next)
            idx_next = self.__sample(logits)
            idx = torch.cat((idx, idx_next), dim=1)
        
        # free memory associated with request
        self.allocator.free(req_id) # don't need to do anything in kv_cache, we'll just overwrite
        del self.requests[req_id]

        # return the decoded result
        return self.tokenizer.decode(idx[0].tolist())
    
    def __sample(self, logits):
        logits = logits[:, -1, :] / self.temp
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        return idx_next

    
    
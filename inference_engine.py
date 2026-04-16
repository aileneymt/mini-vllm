from model import GPT, GPTConfig
from kv_cache import KVCache
from block_allocator import BlockAllocator
import torch
from transformers import GPT2Tokenizer
from torch.nn import functional as F

class Request:
    def __init__(self, req_id, token_pos, idx):
        self.id = req_id
        self.token_pos = token_pos # the index (0 indexed) of the token that was generated in the preceding decode/prefill step.
        # we write to this position in decode
        self.idx = idx
        self.generated_tokens = 0
    
    def append_token(self, idx_next):
        self.idx = torch.cat((self.idx, idx_next), dim=1)
        self.token_pos += 1
        self.generated_tokens += 1

class InferenceEngine:
    # creates GPT object
    # takes prompt, runs prefill, decode loop
    # uses KVCache to manage the cache 
    # can handle multiple requests
    def __init__(self, total_blocks: int, block_size: int, max_tokens=50, temp=1.0):
        self.max_tokens = max_tokens
        self.model = GPT.from_pretrained('gpt2')
        self.config = self.model.config
        self.allocator = BlockAllocator(total_blocks, block_size)
        self.kv_cache = KVCache(total_blocks, block_size, self.config.n_layer, self.config.n_head, self.config.n_embd // self.config.n_head)
        self.requests: dict[int, Request] = {} # map request id -> the Request object
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.temp = temp
        self.__reqCtr = 0

    def __prepare_attention(self, request: Request):
        for i, dec_block in enumerate(self.model.transformer.h):
            dec_block.attn.layer_idx = i
            dec_block.attn.token_pos = request.token_pos
            dec_block.attn.kv_cache = self.kv_cache
            dec_block.attn.block_list = self.allocator.block_table[request.id]
    
    def prefill_step(self, prompt) -> int:
        idx = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0)
        req_id = self.__reqCtr
        self.__reqCtr += 1
        num_tokens_prompt = idx.shape[1]
        request = Request(req_id, num_tokens_prompt - 1, idx)
        self.requests[req_id] = request

        # prefill stage
        self.allocator.allocate_prefill(req_id, num_tokens_prompt)
        self.__prepare_attention(request)
        logits, _ = self.model(idx)
        idx_next = self.__sample(logits)
        request.append_token(idx_next)

        return req_id

    def decode_step(self, request_id: int) -> bool:
        request = self.requests[request_id]
        if request.generated_tokens >= self.max_tokens: return False
    
        self.allocator.allocate_decode(request_id)
        self.__prepare_attention(request)
        logits, _ = self.model(request.idx[:, -1:], position_offset=request.token_pos)
        idx_next = self.__sample(logits)
        request.append_token(idx_next)
        return True
    
    def complete_request(self, request_id: int) -> str:
        output = self.tokenizer.decode(self.requests[request_id].idx[0].tolist())
        self.allocator.free(request_id) # don't need to do anything in kv_cache, we'll just overwrite
        del self.requests[request_id]

        # return the decoded result
        return output

    # generate a response for a prompt sequentially
    def generate(self, prompt) -> str:
        request_id = self.prefill_step(prompt)
        while self.decode_step(request_id):
            continue
        return self.complete_request(request_id)

    
    def __sample(self, logits, top_k=50):
        logits = logits[:, -1, :] / self.temp
        # keep only top k tokens
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float('-inf')
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        return idx_next

    
    
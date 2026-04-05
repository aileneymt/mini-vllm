class InferenceEngine:
    # loads GPT-2
    # takes prompt, runs prefill, decode loop
    # uses KVCache instead of letting PyTorch manage the cache 
    # can handle multiple requests
    def __init__(self):
        return None
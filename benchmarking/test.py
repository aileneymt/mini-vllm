from inference_engine import InferenceEngine

engine = InferenceEngine(100, 16, 50, 0.8)

output = engine.generate("Hi my name is")
print(output)

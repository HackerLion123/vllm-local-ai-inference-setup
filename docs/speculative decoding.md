# Speculative Decoding

Speculative decoding uses a smaller, faster "draft" model to generate draft tokens and then a larger "target" model to verify them. This can significantly speed up generation, especially for models with high VRAM bandwidth.

**Configuration:**

Specify a `speculative_model` and adjust `num_speculative_tokens`.

**Python (`LLM` class):**
````python
# Use a smaller model for speculative decoding
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    speculative_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    num_speculative_tokens=5,
    tensor_parallel_size=4, # Ensure the main model fits
    trust_remote_code=True,
)
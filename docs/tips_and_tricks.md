# Best Practices to improve performance and memory

This guide provides in-depth tips and configuration examples to optimize your vLLM deployment for performance and memory efficiency.

## 1. Use Quantized Models

Quantization reduces the precision of model weights (e.g., from 16-bit to 4-bit), significantly cutting VRAM usage and often increasing inference speed with minimal impact on accuracy.

-   **AWQ (Activation-aware Weight Quantization):** Recommended for fast inference and good accuracy.
-   **GPTQ (General-purpose Post-Training Quantization):** Another popular format.
-   **SqueezeLLM:** Good for models with large vocabulary sizes.

**Configuration:**

When initializing the model, specify the `quantization` method. vLLM will automatically download the correct model version if available on the Hugging Face Hub.

**Python:**
````python
from vllm import LLM

# Use a 4-bit AWQ quantized model
llm = LLM(
    model="TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
    quantization="awq",
    dtype="half",
    trust_remote_code=True,
)
````


## 2. Optimize KV Cache Usage

The KV cache stores attention keys and values for generated tokens, consuming a large portion of VRAM.

### Control GPU Memory Allocation

Use `gpu_memory_utilization` to control the fraction of GPU VRAM vLLM is allowed to use for the model weights and KV cache. A value between `0.8` and `0.9` is a safe starting point.

**Python (`LLM` class):**
````python
llm = LLM(model="mistralai/Mistral-7B-v0.1", gpu_memory_utilization=0.85)
````


### Limit Maximum Context Length

If your application doesn't require a large context window, reduce `max_model_len` to save a significant amount of KV cache memory.

**Python:**
````python
# Limit context to 2048 tokens instead of the model's default
llm = LLM(model="mistralai/Mistral-7B-v0.1", max_model_len=2048)
````

**OpenAI API Server (CLI):**
````bash
python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-7B-v0.1 \
    --max-model-len 2048
````

### Enable KV Cache CPU Offloading

If you have limited VRAM but ample system RAM, you can offload the KV cache to the CPU. This prevents out-of-memory errors but introduces latency when swapping.

**Python:**
````python
# Allocate 16GB of CPU RAM for the swap space
llm = LLM(
    model="meta-llama/Llama-2-13b-hf",
    swap_space=16, # in GiB
    gpu_memory_utilization=0.8
)
````


## 3. Tune Engine Parameters for Concurrency

Properly configuring the engine can maximize throughput for your specific workload.

### `max_num_seqs`

This sets the maximum number of sequences (requests) that can be processed in a single batch. A higher number increases potential throughput but also VRAM usage. If you encounter OOM errors, try lowering this value.

**Python:**
````python
from vllm import LLM, EngineArgs

engine_args = EngineArgs(model="mistralai/Mistral-7B-v0.1", max_num_seqs=128)
llm = LLM.from_engine_args(engine_args)
````


### `max_num_batched_tokens`

This controls the maximum total number of tokens (prompt + generation) processed in a single forward pass. It helps prevent OOMs with very long sequences. The default is often sufficient, but you can tune it for specific workloads.

**Python:**
````python
llm = LLM(model="mistralai/Mistral-7B-v0.1", max_num_batched_tokens=4096)
````


## 4. Use Streaming for Better Responsiveness

Streaming returns tokens as they are generated, dramatically reducing the perceived latency or time-to-first-token (TTFT).

**Python (`LLM` class with `stream=True`):**
````python
from vllm import LLM, SamplingParams

prompts = ["Write a short story about a robot who dreams."]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)
llm = LLM(model="mistralai/Mistral-7B-v0.1")

# Use stream=True in the generate call
request_id = 0
results_generator = llm.generate(prompts, sampling_params, request_id, stream=True)

# The generator will yield RequestOutput objects as tokens are generated
full_output = ""
for request_output in results_generator:
    # The latest token is in the last entry of the outputs list
    latest_token = request_output.outputs[-1].text
    full_output += latest_token

print(f"Full output: {full_output}")
````

When using the OpenAI-compatible server, simply set `stream: true` in your API request body.

## 5. Advanced: Multi-GPU Inference with Tensor Parallelism

If you are using a large model that does not fit on a single GPU, you can use tensor parallelism to shard the model across multiple GPUs.

**Configuration:**

Set the `tensor_parallel_size` argument to the number of GPUs you want to use.

**Python:**
````python
# Use 2 GPUs for tensor parallelism
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=2,
    trust_remote_code=True,
)
````


## 6. Advanced: Serving Multiple Models with LoRA Adapters

If you need to serve multiple fine-tuned models based on the same base model, using LoRA (Low-Rank Adaptation) adapters is highly efficient. vLLM can serve multiple LoRA adapters on the same base model, switching between them on-the-fly.

**Configuration:**

Enable LoRA and provide the paths to your adapters.

**Python:**
````python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# Initialize the base model with LoRA enabled
llm = LLM(
    model="mistralai/Mistral-7B-v0.1",
    enable_lora=True,
    max_loras=2, # Max number of adapters to load at once
    max_lora_rank=8,
)

# Define sampling parameters and specify the LoRA adapter to use
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=256,
)

# Make a request with a specific LoRA adapter
prompts = ["What is the capital of France?"]
lora_path = "/path/to/your/lora_adapter_1"
outputs = llm.generate(
    prompts,
    sampling_params,
    lora_request=LoRARequest("adapter1", 1, lora_path)
)
````

When making a request to the server, you can specify the LoRA adapter to use in the request body:
````json
{
  "model": "mistralai/Mistral-7B-v0.1",
  "prompt": "What is the capital of France?",
  "lora_request": {
    "lora_name": "adapter1",
    "lora_int_id": 1,
    "lora_local_path": "/path/to/your/lora_adapter_1"
  }
}
````





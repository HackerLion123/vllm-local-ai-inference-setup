# vLLM - Understanding vLLM


## What is vLLM

Fastest inference framework to deploy production level llm for high speed inference. It is the fastest llm inference tool for local or self deployed llms. vLLM has exceled in combining all different optimizitations proposed in papers and have unified them to work together. This is done while retaining perfomance of the model.

### 1. Prefix Caching
Caches attention key/value (KV) pairs for shared prompt prefixes so they aren’t recomputed.

**How it works**
- When two (or more) requests begin with the same token sequence (e.g., a shared system prompt or context), vLLM stores the KV cache for that prefix once and **reuses** it for later requests with the *exact same* prefix.
- At generation time, the model only needs to compute the *new* (suffix) tokens, which is much cheaper than re-running the whole prefix.

**When it helps**
- Chat templates where a long system prompt is reused across many users.
- RAG systems that tack different questions onto the same retrieved background context.

**Tips**
- Keep the system prompt and formatting **identical** (same tokens) across requests to maximize cache hits.
- If you A/B test different prompt wordings, do it **after** the shared prefix so you still benefit from caching.

**Trade-offs**
- The cache consumes memory. Engines evict least-used entries under pressure, so extremely diverse prefixes can reduce hit rate.


### 2. Speculative Decoding
Uses a faster “draft” process to propose multiple tokens, then verifies them with the full model in one step.

**How it works**
- A lightweight drafter (small model or heuristic) proposes the next *k* tokens.
- The target (large) model runs **one** forward pass to verify those *k* tokens together.
- Accepted tokens are committed; rejected ones fall back to standard decoding for that step.

**When it helps**
- Medium/low temperature, predictable outputs (higher acceptance rate).
- Models with a much smaller/faster “draft” available.

**Tips**
- Tune the look-ahead length *k*: too small → less speedup; too large → more rejections.
- Keep sampling settings reasonable (extreme randomness lowers acceptance).

**Trade-offs**
- Extra complexity and some memory for the drafter.
- If outputs are highly stochastic, gains shrink.


### 3. Chunked Prefills
Splits long prompt processing into chunks so one long request doesn’t monopolize the GPU.

**How it works**
- The “prefill” (encoding the input) is broken into N-token chunks.
- Between chunks, the scheduler can interleave work from other requests, improving fairness and time-to-first-token (TTFT) for everyone.

**When it helps**
- Very long inputs (RAG, code, logs).
- Mixed workloads where short queries should not wait behind a single giant prompt.

**Tips**
- Use a chunk size that is large enough for good throughput but small enough to let others in (common defaults work well).
- Combine with streaming input for best perceived latency.

**Trade-offs**
- Slight overhead from more scheduling steps.
- If you only run single long requests, chunking offers limited benefit.


### 4. Disaggregated Serving
Separates **prefill** (compute-heavy) and **decode** (memory/bandwidth-heavy) into different workers or pools.

**How it works**
- One set of workers focuses on ingesting prompts (prefill), another on token generation (decode).
- KV/state can be shared/handed off so decode workers don’t wait on prefill spikes.

**When it helps**
- Traffic with bursts of long prompts alongside many ongoing generations.
- Multi-node clusters where you want to scale each phase independently.

**Tips**
- Size the prefill pool for peak ingest, the decode pool for steady token throughput.
- Use fast interconnects if states move across machines.

**Trade-offs**
- More moving parts (coordination overhead).
- Cross-worker/state transfer can add latency if the network is slow.


### 5. Streaming Prefills
Starts computing while the prompt is still arriving, rather than waiting for the entire input.

**How it works**
- The engine ingests tokens incrementally and updates the KV cache as they come in.
- Generation can begin as soon as enough of the prompt has been processed.

**When it helps**
- Interactive UIs where users paste/type long content.
- APIs that send large payloads in chunks.

**Tips**
- Pair with chunked prefill so others aren’t blocked by a stream.
- If you can, send the **static** prefix first (so it can be cached/reused).

**Trade-offs**
- Slight scheduling overhead.
- If upstream can’t truly stream, benefits are limited.


### 6. Jump-Forward Decoding
Skips over deterministic stretches of output (e.g., fixed JSON keys) in larger steps instead of token-by-token.

**How it works**
- When a grammar or schema implies a **single valid continuation** (no branching), the engine can emit multiple tokens in one go.
- This reduces per-token overhead during known, fixed segments of structured output.

**When it helps**
- Strictly structured formats (JSON with fixed keys, XML tags, function-call scaffolds).
- Templates where long literal spans are predetermined.

**Tips**
- Combine with **guided/grammar decoding** so the engine knows where jumps are valid.
- Keep schemas tight—fewer branches → more jump opportunities.

**Trade-offs**
- Only helps in deterministic regions; free-form text still decodes normally.
- Requires correct schemas; malformed grammars negate benefits.


### 7. Quantization
Lowers numerical precision to save memory and boost speed with minimal accuracy loss.

**What you can quantize**
- **Weights (post-training):** GPTQ, AWQ (often 4–8 bits) → biggest VRAM savings.
- **Activations/KV cache:** INT8/FP8 variants reduce runtime memory footprint.

**When it helps**
- Fitting bigger models or longer context into limited VRAM.
- Increasing batch size and throughput on the same hardware.

**Tips**
- Prefer **weight-only** quant (GPTQ/AWQ) when you need the largest savings with strong accuracy.
- Try KV-cache INT8/FP8 for long conversations where KV dominates memory.
- Validate quality on your task; not all models tolerate aggressive quant equally.

**Trade-offs**
- Potential small accuracy/perplexity hit.
- Some quant schemes add de/quant overhead; real gains depend on kernels and GPU.



### 8. Cascade Attention
Shares attention computation for **common prefixes across a batch**, then branches for the unique suffixes.

**How it works**
- Requests that start with the same tokens get their prefix attention computed once.
- The computation **cascades** to each request’s divergent tail, avoiding duplicated work.

**When it helps**
- High-throughput serving where many requests share a system prompt or header.
- Prompt-templated apps (chat, forms, agent frameworks).

**Tips**
- Batch requests with known shared context to maximize reuse.
- Keep tokenization stable across requests (same template → same tokens).

**Trade-offs**
- Benefits shrink as prefixes diverge.
- Requires scheduler awareness to group similar requests.



### 9. Structured Outputs (Guided Decoding)
Constrains generation to a schema/grammar so outputs are valid by construction (e.g., JSON).

**How it works**
- At each step, the decoder masks out tokens that would violate a provided grammar (JSON schema, regex, EBNF, etc.).
- The model can only pick valid tokens, guaranteeing the final output’s structure.

**When it helps**
- APIs and tools that **must** receive valid JSON/XML.
- Form filling, function-calling, code generation with strict syntax.

**Tips**
- Keep schemas precise but not overly complex—simpler grammars are faster.
- Use with sampling that isn’t too “wild”; extreme randomness can fight constraints.

**Trade-offs**
- Some per-token overhead to compute allowed tokens.
- Very complex grammars can slow decoding; consider simplifying.



### 10. CPU KV Cache Offloading
Moves colder parts of the KV cache from GPU to CPU RAM to stretch effective context/concurrency.

**How it works**
- KV blocks that are unlikely to be needed immediately are evicted to host memory.
- When needed again (e.g., for attention to far-back tokens), they’re prefetched back to GPU.

**When it helps**
- Long-context chats or many parallel sessions that otherwise exceed VRAM.
- Spiky workloads where not all sessions are active at once.

**Tips**
- Works best on systems with fast PCIe/NVLink and ample CPU RAM.
- Tune offload/eviction thresholds to balance transfers vs. recompute.

**Trade-offs**
- Data movement over PCIe adds latency; heavy back-and-forth can erase gains.
- If **every** token needs the whole history (rare), offloading won’t help much.


### 11. Multi-LoRA Serving
Serves many LoRA adapters on a single base model without duplicating the base weights.

**How it works**
- The base model stays loaded once.
- For each request, the specified LoRA adapter’s deltas are applied on-the-fly (or via efficient fused kernels).
- Adapters can be cached and swapped per request.

**When it helps**
- Multi-tenant systems (one persona/task per adapter).
- A/B testing or routing across specialized LoRAs without spinning up many replicas.

**Tips**
- Keep adapter sizes modest and reuse popular adapters to benefit from caching.
- Name/ID adapters clearly and pass the adapter ID per request.

**Trade-offs**
- Small per-token overhead for applying adapters.
- Very large numbers of distinct adapters may pressure memory if all are hot simultaneously.



## Installing vLLM

```
pip install vllm
```



## Simple Getting Started

```
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")

prompts = ["The future of humanity is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

outputs = llm.generate(prompts, sampling_params)

print("Prompt:", outputs[0].prompt)
print("Generated:", outputs[0].outputs[0].text)
```

Start the server.

```
vllm serve meta-llama/Meta-Llama-3-8B-Instruct --port 8000
```

To have open-webui with vllm models

```
docker run -d \
    --name open-webui \
    -p 3000:8080 \
    -v open-webui:/app/backend/data \
    -e OPENAI_API_BASE_URL=http://0.0.0.0:8000/v1 \
    --restart always \
    ghcr.io/open-webui/open-webui:main
```




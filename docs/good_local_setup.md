# Setting up Local LLM Inference Using vLLM

For a **single‑user local setup**, you don’t need vLLM’s full distributed serving stack or complex scaling features.  
The goal here is:

- Pick the **right hardware & model size**
- Install vLLM correctly for that hardware
- Enable a **small subset of engine options** that matter for local use (memory, quantization, context length)
- Run either:
  - **Offline inference from Python**, or  
  - A **local OpenAI‑compatible HTTP server**

---

## 1. What vLLM Actually Gives You (and What We’ll Ignore)

vLLM is a high‑throughput inference engine with:

- **PagedAttention** – treats KV cache like virtual memory to pack many sequences efficiently and avoid fragmentation.:contentReference[oaicite:0]{index=0}  
- **Continuous batching**, CUDA/HIP graphs, and quantization support (AWQ, GPTQ, FP8, INT4/INT8, etc.).:contentReference[oaicite:1]{index=1}  
- **OpenAI‑compatible server** (`vllm serve`) and an **offline `LLM` Python class**.:contentReference[oaicite:2]{index=2}  

For a **local single‑user** setup, we’ll **ignore**:

- Multi‑node deployments, Ray, Kubernetes, Triton, etc.
- Complex scheduling tweaks, advanced speculative decoding configurations
- Multi‑tenant and multi‑LoRA routing setups

We’ll only touch options that give clear benefit on a single machine.

---

## 2. Hardware & OS Planning

### 2.1 Supported Platforms (high level)

vLLM supports:​:contentReference[oaicite:3]{index=3}  

- **GPU**
  - NVIDIA (CUDA)
  - AMD (ROCm)
  - Intel XPU
- **CPU**
  - x86 (Intel/AMD)
  - ARM (AArch64)
  - Apple Silicon

For a **local, interactive LLM**, the realistic options are:

- **Linux + NVIDIA GPU (CUDA)** – _strongly recommended_  
- **Linux + AMD GPU (ROCm)** – good if you’re already on AMD  
- **CPU‑only / Apple Silicon** – fine for **small** models or background jobs; much slower token throughput

> If you have a dGPU and can choose OS: **Ubuntu + NVIDIA** is currently the smoothest path.

### 2.2 VRAM Sizing (Rule of Thumb)

Memory usage ≈ **weights + KV cache + overhead**.

Approx weights sizes:

- 16‑bit (FP16/BF16): `2 bytes * params`
- 8‑bit: `1 byte * params`
- 4‑bit: `0.5 bytes * params`

Approx **weights only**:

| Model size | 4‑bit weights | 8‑bit weights | 16‑bit weights |
| ---------- | ------------- | ------------- | -------------- |
| 7B         | ~3.5 GB       | ~7 GB         | ~14 GB         |
| 8B         | ~4.0 GB       | ~8 GB         | ~16 GB         |
| 13B        | ~6.5 GB       | ~13 GB        | ~26 GB         |
| 33B        | ~16.5 GB      | ~33 GB        | ~66 GB         |

You also need space for:

- KV cache (depends on **context length** and **concurrency**)
- Runtime overhead, temporary buffers

**Practical guidance (single user, modest context):**

- **8–10 GB GPU**  
  - 7B **Q4** model (or smaller FP16)  
  - Context: up to ~4k–8k tokens
- **12–16 GB GPU**  
  - 7–8B **FP16/BF16** _or_ 13B **Q4**  
- **24 GB GPU**  
  - 13B FP16 comfortably  
  - 30B‑class models with quantization (Q4/Q8) if tuned carefully

If your VRAM is tight, **quantization** and **lower max context** are your main tools.

General role of thumb is quantized version of larger parameter models will give better results than a smaller model.

### 2.3 System RAM & Disk

- **RAM**: 32 GB is comfortable, 16 GB workable for smaller models
- **Disk** (NVMe recommended):
  - At least **2× model size**:
    - 1× for the downloaded model
    - 1× for caches / temp / extra versions
- Make sure your Linux swap isn’t tiny; out‑of‑memory kills are annoying mid‑generation.

---

## 3. Installing vLLM

We’ll assume:

- Linux
- Python 3.9–3.12 

### 3.1 Create a Virtual Environment

Pick your favorite: `venv`, `conda`, or `uv`. Example with `venv`:

```bash
python3 -m venv vllm-env
source vllm-env/bin/activate
python -m pip install --upgrade pip
```


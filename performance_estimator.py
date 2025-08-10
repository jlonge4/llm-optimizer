import math
import argparse
import json
from pathlib import Path
from huggingface_hub import hf_hub_download

# --- GPU Data ---
# Added VRAM and filled in missing TFLOP values for consistency
gpu_data = {
    "A100": {
        "FP16_TFLOPs": 624, "FP8_TFLOPs": 1248, "INT4_TOPS": 2496,
        "Memory_Bandwidth_GBs": 2039, "VRAM_GB": 80
    },
    "A40": {
        "FP16_TFLOPs": 299.4, "FP8_TFLOPs": 598.8, "INT4_TOPS": 1197.4,
        "Memory_Bandwidth_GBs": 696, "VRAM_GB": 48
    },
    "A30": {
        "FP16_TFLOPs": 330, "FP8_TFLOPs": 660, "INT4_TOPS": 1320,
        "Memory_Bandwidth_GBs": 933, "VRAM_GB": 24
    },
    "A10": {
        "FP16_TFLOPs": 250, "FP8_TFLOPs": 500, "INT4_TOPS": 1000,
        "Memory_Bandwidth_GBs": 600, "VRAM_GB": 24
    },
    "A16": {
        "FP16_TFLOPs": 143.6, "FP8_TFLOPs": 287.2, "INT4_TOPS": 574.4,
        "Memory_Bandwidth_GBs": 800, "VRAM_GB": 64
    },
    "A2": {
        "FP16_TFLOPs": 36, "FP8_TFLOPs": 72, "INT4_TOPS": 144,
        "Memory_Bandwidth_GBs": 200, "VRAM_GB": 16
    },
    "H100": {
        "FP16_TFLOPs": 989, "FP8_TFLOPs": 1979, "INT4_TOPS": 3958,
        "Memory_Bandwidth_GBs": 3350, "VRAM_GB": 80
    },
    "H200": {
        "FP16_TFLOPs": 1980, "FP8_TFLOPs": 4000, "INT4_TOPS": 8000,
        "Memory_Bandwidth_GBs": 4800, "VRAM_GB": 141
    },
    "B100": {
        "FP16_TFLOPs": 1800, "FP8_TFLOPs": 3500, "INT4_TOPS": 7000,
        "Memory_Bandwidth_GBs": 4000, "VRAM_GB": 192
    },
    "B200": {
        "FP16_TFLOPs": 5000, "FP8_TFLOPs": 10000, "INT4_TOPS": 20000,
        "Memory_Bandwidth_GBs": 8000, "VRAM_GB": 192
    },
}

def get_model_config_from_hf(model_id: str) -> dict:
    """
    Downloads a model's config.json from Hugging Face and calculates its parameters.
    """
    try:
        config_path = hf_hub_download(repo_id=model_id, filename="config.json")
        with open(config_path) as f:
            config = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Could not download or read config.json for {model_id}: {e}")

    try:
        # Extract parameters needed for calculation
        h = config["hidden_size"]
        n_layers = config["num_hidden_layers"]
        i = config["intermediate_size"]
        v = config["vocab_size"]
        n_heads = config.get("num_attention_heads", 0)
        n_kv_heads = config.get("num_key_value_heads", n_heads)

        # Calculate params per layer
        head_dim = h // n_heads
        attention_params = n_layers * (h * (n_heads * head_dim) + h * (n_kv_heads * head_dim) * 2 + h * h)
        
        # FFN params (assuming SwiGLU)
        ffn_params = n_layers * (h * i * 2 + i * h)
        
        # Embedding and output params
        embedding_params = v * h
        output_params = v * h if not config.get("tie_word_embeddings", False) else 0

        total_params = attention_params + ffn_params + embedding_params + output_params
        
        return {
            "num_params": total_params / 1e9, # Return in billions
            "num_layers": n_layers,
            "hidden_dim": h,
        }
    except KeyError as e:
        raise KeyError(f"Could not find required key {e} in config.json for {model_id}")


def estimate_llm_performance(
    num_gpus: int, tflop_per_gpu: float, mem_bw_per_gpu: float, vram_per_gpu: float,
    num_params: float, num_layers: int, hidden_dim: int, precision: str,
    concurrency: int, input_length: int, output_length: int,
    mfu_prefill: float = 0.45, mfu_decode: float = 0.30, vram_util_factor: float = 0.90
) -> dict:
    total_tflops = num_gpus * tflop_per_gpu
    total_mem_bw = num_gpus * mem_bw_per_gpu
    total_usable_vram = num_gpus * vram_per_gpu * vram_util_factor

    bytes_per_param = {"fp16": 2, "fp8": 1, "int4": 0.5}.get(precision, 2)
    num_params_val = num_params * 1e9
    model_size_gb = (num_params_val * bytes_per_param) / 1e9

    kv_cache_per_token_gb = (2 * num_layers * hidden_dim * bytes_per_param) / 1e9
    kv_cache_per_request_gb = kv_cache_per_token_gb * (input_length + output_length)
    total_kv_cache_gb = concurrency * kv_cache_per_request_gb
    total_memory_needed_gb = model_size_gb + total_kv_cache_gb

    if total_memory_needed_gb > total_usable_vram:
        return {
            "error": "Not enough VRAM.",
            "memory_needed_gb": round(total_memory_needed_gb, 2),
            "usable_vram_gb": round(total_usable_vram, 2)
        }

    prefill_flops = 2 * num_params_val * input_length * concurrency
    effective_prefill_flops = total_tflops * 1e12 * mfu_prefill
    ttft_s = prefill_flops / effective_prefill_flops if effective_prefill_flops > 0 else float('inf')

    decode_step_flops = 2 * num_params_val * concurrency
    effective_decode_flops = total_tflops * 1e12 * mfu_decode
    decode_time_compute_s = decode_step_flops / effective_decode_flops if effective_decode_flops > 0 else float('inf')
    
    model_size_bytes = model_size_gb * 1e9
    total_mem_bw_bytes = total_mem_bw * 1e9
    decode_time_memory_s = model_size_bytes / total_mem_bw_bytes if total_mem_bw_bytes > 0 else float('inf')
    
    itl_s = max(decode_time_compute_s, decode_time_memory_s)

    output_throughput_tps = concurrency / itl_s if itl_s > 0 else float('inf')
    input_throughput_tps = (input_length * concurrency) / ttft_s if ttft_s > 0 else float('inf')
    e2e_latency_s = ttft_s + (output_length * itl_s)
    requests_per_sec = concurrency / e2e_latency_s if e2e_latency_s > 0 else float('inf')

    return {
        "ttft_ms": round(ttft_s * 1000, 2),
        "itl_ms": round(itl_s * 1000, 2),
        "e2e_latency_s": round(e2e_latency_s, 2),
        "output_throughput_tps": round(output_throughput_tps, 2),
        "input_throughput_tps": round(input_throughput_tps, 2),
        "requests_per_sec": round(requests_per_sec, 2),
        "bottleneck_is_memory": decode_time_memory_s > decode_time_compute_s,
    }

def main():
    parser = argparse.ArgumentParser(
        description="Estimate LLM Performance by fetching model config from Hugging Face.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--gpu", type=str, required=True, choices=gpu_data.keys(), help="GPU model to use for estimation.")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs.")
    parser.add_argument("--model", type=str, required=True, help="Hugging Face model ID (e.g., 'meta-llama/Meta-Llama-3-8B').")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "fp8", "int4"], help="Precision for weights and KV cache.")
    parser.add_argument("--concurrency", type=int, default=32, help="Number of concurrent requests (batch size).")
    parser.add_argument("--input-len", type=int, default=1024, help="Input sequence length in tokens.")
    parser.add_argument("--output-len", type=int, default=1024, help="Output sequence length to generate.")
    
    args = parser.parse_args()

    try:
        # --- Get Specs from Dictionaries and Hugging Face ---
        gpu_specs = gpu_data[args.gpu]
        print(f"Fetching model config for '{args.model}'...")
        model_specs = get_model_config_from_hf(args.model)
        print(f"Successfully loaded model config: {model_specs['num_params']:.2f}B parameters.")

        tflops_map = {"fp16": "FP16_TFLOPs", "fp8": "FP8_TFLOPs", "int4": "INT4_TOPS"}
        tflops = gpu_specs[tflops_map[args.precision]]

        if tflops is None:
            print(f"Error: {args.precision.upper()} performance not available for {args.gpu}")
            return

        print(f"\n--- Estimating: {args.model} on {args.num_gpus}x {args.gpu} ({args.precision}) ---")
        print(f"Workload: Concurrency={args.concurrency}, Input={args.input_len}, Output={args.output_len}\n")

        performance = estimate_llm_performance(
            num_gpus=args.num_gpus,
            tflop_per_gpu=tflops,
            mem_bw_per_gpu=gpu_specs["Memory_Bandwidth_GBs"],
            vram_per_gpu=gpu_specs["VRAM_GB"],
            num_params=model_specs["num_params"],
            num_layers=model_specs["num_layers"],
            hidden_dim=model_specs["hidden_dim"],
            precision=args.precision,
            concurrency=args.concurrency,
            input_length=args.input_len,
            output_length=args.output_len
        )

        for key, val in performance.items():
            print(f"{key:<22}: {val}")
        print("-" * 40)

    except (RuntimeError, KeyError) as e:
        print(f"\nError: {e}")

if __name__ == '__main__':
    main()

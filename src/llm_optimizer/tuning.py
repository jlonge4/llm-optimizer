"""
Parameter tuning configurations for different LLM frameworks.

This module generates server and client arguments for optimal performance
based on hardware specifications and workload requirements.
"""

import math
from dataclasses import dataclass

from llm_optimizer.performance import (
    ModelConfig,
    get_parameter_conservativeness_for_stat_type,
)
from llm_optimizer.predefined.gpus import get_gpu_specs, get_precision_tflops


@dataclass
class TuningConfig:
    """Configuration for parameter tuning."""

    framework: str
    server_args: list[str]
    client_args: list[str]
    description: str


def get_precision_bytes_per_param(precision: str) -> int:
    """Get bytes per parameter for different precisions."""
    precision_map = {
        "fp16": 2,  # 16 bits = 2 bytes
        "fp8": 1,   # 8 bits = 1 byte
        "bf16": 2,  # bfloat16 = 2 bytes
    }
    if precision not in precision_map:
        raise ValueError(f"Unsupported precision: {precision}. Use {list(precision_map.keys())}")
    return precision_map[precision]


def calculate_model_memory_gb(
    model_config: ModelConfig,
    precision: str,
    safety_factor: float = 1.2
) -> float:
    """Calculate model memory usage in GB including safety margin."""
    bytes_per_param = get_precision_bytes_per_param(precision)
    model_memory_gb = (model_config.num_params * bytes_per_param * safety_factor) / 1e9
    return model_memory_gb


def calculate_kv_cache_memory_per_token(
    model_config: ModelConfig,
    precision: str
) -> float:
    """Calculate KV cache memory usage per token in bytes."""
    # KV cache stores key and value vectors for each layer
    # Shape: [num_layers, 2 (K+V), hidden_dim] per token
    bytes_per_param = get_precision_bytes_per_param(precision)

    # For GQA/MQA, we use num_kv_heads instead of num_heads
    kv_heads = model_config.num_kv_heads
    head_dim = model_config.hidden_dim // model_config.num_heads

    # KV cache size per token = layers * 2 (K+V) * kv_heads * head_dim * bytes_per_param
    kv_memory_per_token = (
        model_config.num_layers * 2 * kv_heads * head_dim * bytes_per_param
    )

    return kv_memory_per_token


def calculate_activation_memory_per_token(
    model_config: ModelConfig,
    precision: str
) -> float:
    """Calculate activation memory per token in bytes (rough estimate)."""
    bytes_per_param = get_precision_bytes_per_param(precision)

    # Rough estimate: activations ≈ 2x hidden_dim per layer
    # This includes attention outputs, MLP activations, residuals
    activation_memory = (
        model_config.num_layers * 2 * model_config.hidden_dim * bytes_per_param
    )

    return activation_memory


def calculate_optimal_batch_tokens(
    gpu_specs: dict,
    model_config: ModelConfig,
    precision: str,
    sequence_length: int,
    memory_utilization: float = 0.8
) -> int:
    """Calculate optimal batch tokens based on GPU memory and bandwidth."""

    # Available GPU memory after model weights
    model_memory_gb = calculate_model_memory_gb(model_config, precision)
    available_memory_gb = gpu_specs["VRAM_GB"] * memory_utilization - model_memory_gb

    if available_memory_gb <= 0:
        # Model doesn't fit, return minimum
        return 1024

    # Memory per token (KV cache + activations)
    kv_memory_per_token = calculate_kv_cache_memory_per_token(model_config, precision)
    activation_memory_per_token = calculate_activation_memory_per_token(model_config, precision)
    total_memory_per_token = kv_memory_per_token + activation_memory_per_token

    # Calculate max batch tokens based on available memory
    available_memory_bytes = available_memory_gb * 1e9
    max_batch_tokens = int(available_memory_bytes / total_memory_per_token)

    # Apply bandwidth-based heuristics
    # Higher bandwidth GPUs can handle larger batches more efficiently
    bandwidth_factor = min(2.0, gpu_specs["Memory_Bandwidth_GBs"] / 2000)  # Normalize around 2TB/s

    # Suggest batch sizes that are powers of 2 or multiples of 1024 for efficiency
    optimal_batch = int(max_batch_tokens * bandwidth_factor)

    # Round to nearest efficient batch size
    efficient_sizes = [1024, 2048, 4096, 6144, 8192, 12288, 16384, 24576, 32768]
    optimal_batch = min(efficient_sizes, key=lambda x: abs(x - optimal_batch))

    return max(1024, optimal_batch)  # Minimum 1024 tokens


def calculate_optimal_max_seqs(
    gpu_specs: dict,
    model_config: ModelConfig,
    precision: str,
    sequence_length: int,
    optimal_concurrency: int,
    target_throughput: bool = True
) -> dict:
    """Calculate optimal max_seqs values for different performance targets."""

    # Base calculations using GPU specs
    tflops = get_precision_tflops(gpu_specs, precision)
    memory_bandwidth = gpu_specs["Memory_Bandwidth_GBs"]

    # Compute capacity limit (based on FLOPS)
    # Higher FLOPS GPUs can handle more concurrent sequences
    compute_scale = min(2.0, tflops / 300)  # Normalize around 300 TFLOPS

    # Memory bandwidth limit (higher bandwidth = more concurrent sequences)
    bandwidth_scale = min(2.0, memory_bandwidth / 2000)  # Normalize around 2TB/s

    # Combine factors
    capacity_factor = (compute_scale + bandwidth_scale) / 2

    # Calculate different configurations
    configs = {
        "conservative": max(16, int(optimal_concurrency * 0.25)),
        "balanced": max(32, int(optimal_concurrency * 0.5 * capacity_factor)),
        "aggressive": max(64, int(optimal_concurrency * capacity_factor)),
        "latency_optimized": min(32, int(optimal_concurrency * 0.3)),
        "memory_efficient": max(8, int(optimal_concurrency * 0.125))
    }

    return configs


def calculate_chunked_prefill_size(
    gpu_specs: dict,
    model_config: ModelConfig,
    precision: str,
    target_throughput: bool = True
) -> int:
    """Calculate optimal chunked prefill size based on GPU specs."""

    # Larger, more powerful GPUs can handle bigger prefill chunks
    tflops = get_precision_tflops(gpu_specs, precision)
    memory_bandwidth = gpu_specs["Memory_Bandwidth_GBs"]

    # Scale based on compute and memory capabilities
    compute_factor = min(2.0, tflops / 300)  # Normalize around 300 TFLOPS
    memory_factor = min(2.0, memory_bandwidth / 2000)  # Normalize around 2TB/s

    # Base chunk size
    base_chunk = 2048 if target_throughput else 1024

    # Apply scaling
    scaled_chunk = int(base_chunk * (compute_factor + memory_factor) / 2)

    # Round to powers of 2
    chunk_sizes = [1024, 2048, 4096, 8192, 16384]
    optimal_chunk = min(chunk_sizes, key=lambda x: abs(x - scaled_chunk))

    return optimal_chunk


def calculate_memory_fraction(
    gpu_specs: dict,
    model_config: ModelConfig,
    precision: str,
    conservative: bool = False
) -> float:
    """Calculate optimal GPU memory utilization fraction."""

    # Calculate model memory requirements
    model_memory_gb = calculate_model_memory_gb(model_config, precision)
    total_vram_gb = gpu_specs["VRAM_GB"]

    # Model memory ratio
    model_ratio = model_memory_gb / total_vram_gb

    if conservative:
        # Conservative: leave more headroom
        if model_ratio > 0.6:  # Large model
            return 0.8
        elif model_ratio > 0.3:  # Medium model
            return 0.85
        else:  # Small model
            return 0.9
    else:
        # Aggressive: use more memory
        if model_ratio > 0.6:  # Large model
            return 0.9
        elif model_ratio > 0.3:  # Medium model
            return 0.95
        else:  # Small model
            return 0.95


def get_precision_tflops(gpu_specs: dict, precision: str) -> float:
    """Get TFLOPS for the specified precision from GPU specs."""
    if precision == "fp16" or precision == "bf16":
        return gpu_specs["FP16_TFLOPS"]
    elif precision == "fp8":
        if gpu_specs["FP8_TFLOPS"] is None:
            raise ValueError(f"FP8 not supported on this GPU architecture: {gpu_specs.get('Architecture', 'Unknown')}")
        return gpu_specs["FP8_TFLOPS"]
    else:
        raise ValueError(f"Unsupported precision: {precision}")


def generate_concurrency_range(optimal_concurrency: int, variation_factor: float = 0.3) -> list[int]:
    """
    Generate a range of concurrency values around the optimal for tuning.

    Args:
        optimal_concurrency: Base concurrency value
        variation_factor: Percentage variation around optimal (0.3 = ±30%)

    Returns:
        List of concurrency values to test
    """
    max(1, int(optimal_concurrency * (1 - variation_factor)))
    int(optimal_concurrency * (1 + variation_factor))

    # Generate 3-5 values in the range
    if optimal_concurrency <= 8:
        return [max(1, optimal_concurrency - 2), optimal_concurrency, optimal_concurrency + 2]
    elif optimal_concurrency <= 32:
        step = max(4, optimal_concurrency // 4)
        return [max(1, optimal_concurrency - step), optimal_concurrency, optimal_concurrency + step]
    else:
        step = max(8, optimal_concurrency // 5)
        return [
            max(1, optimal_concurrency - step),
            max(1, optimal_concurrency - step // 2),
            optimal_concurrency,
            optimal_concurrency + step // 2,
            optimal_concurrency + step
        ]


def generate_concurrency_range_3_values(optimal_concurrency: int) -> list[int]:
    """
    Generate exactly 3 concurrency values for simplified tuning.

    Args:
        optimal_concurrency: Base concurrency value

    Returns:
        List of exactly 3 concurrency values: [n/2, n, n+n/2]
    """
    low = max(1, optimal_concurrency // 2)
    mid = optimal_concurrency
    high = optimal_concurrency + (optimal_concurrency // 2)

    return [low, mid, high]


def generate_3_value_range(optimal_value: int, min_val: int = 1, max_val: int = None) -> list[int]:
    """
    Generate exactly 3 values around an optimal value for any parameter.

    Args:
        optimal_value: Base value
        min_val: Minimum allowed value
        max_val: Maximum allowed value (optional)

    Returns:
        List of exactly 3 values around optimal
    """
    if optimal_value <= 4:
        # For small values, use simple ±1 or ±2
        low = max(min_val, optimal_value - 1)
        mid = optimal_value
        high = optimal_value + 1
    else:
        # For larger values, use percentage-based variation
        variation = max(1, optimal_value // 4)  # 25% variation
        low = max(min_val, optimal_value - variation)
        mid = optimal_value
        high = optimal_value + variation

    if max_val:
        high = min(max_val, high)

    # Ensure we have 3 unique values
    values = [low, mid, high]
    return sorted(set(values))


def generate_tp_dp_combinations(num_gpus: int, min_tp_size: int = 1) -> list[tuple[int, int]]:
    """
    Generate tensor parallel (TP) and data parallel (DP) combinations for multi-GPU setups.

    Args:
        num_gpus: Total number of GPUs
        min_tp_size: Minimum TP size required (based on model size)

    Returns:
        List of (tp_size, dp_size) tuples where tp_size * dp_size = num_gpus
    """
    combinations = []

    # Find all divisor pairs of num_gpus
    for tp_size in range(min_tp_size, num_gpus + 1):
        if num_gpus % tp_size == 0:
            dp_size = num_gpus // tp_size
            combinations.append((tp_size, dp_size))

    # Sort by preference: smaller TP size first (better for throughput)
    combinations.sort(key=lambda x: x[0])

    return combinations


def calculate_min_tensor_parallel_size(
    model_config: ModelConfig,
    gpu_specs: dict,
    precision: str,
    safety_factor: float = 1.2
) -> int:
    """
    Calculate minimum tensor parallel size required to fit the model.

    Args:
        model_config: Model configuration
        gpu_specs: GPU specifications
        precision: Model precision
        safety_factor: Safety factor for memory calculation

    Returns:
        Minimum TP size required (1 if model fits on single GPU)
    """
    model_memory_gb = calculate_model_memory_gb(model_config, precision, safety_factor)
    single_gpu_vram = gpu_specs["VRAM_GB"]

    if model_memory_gb <= single_gpu_vram * 0.9:  # 90% utilization limit
        return 1
    else:
        # Calculate how many GPUs needed for tensor parallel
        min_tp_size = int(math.ceil(model_memory_gb / (single_gpu_vram * 0.9)))
        return min_tp_size


def generate_batch_token_range(optimal_batch_tokens: int) -> list[int]:
    """
    Generate a range of batch token values for tuning.

    Args:
        optimal_batch_tokens: Base batch token value

    Returns:
        List of batch token values to test
    """
    # Generate variants: smaller for latency, larger for throughput
    variants = [
        optimal_batch_tokens // 2,
        optimal_batch_tokens,
        min(optimal_batch_tokens * 2, 32768)  # Cap at 32K tokens
    ]

    # Round to nearest efficient sizes and remove duplicates
    efficient_sizes = [1024, 2048, 4096, 6144, 8192, 12288, 16384, 24576, 32768]
    result = []

    for variant in variants:
        closest = min(efficient_sizes, key=lambda x: abs(x - variant))
        if closest not in result:
            result.append(closest)

    return sorted(result)


def generate_sglang_configs(
    num_gpus: int,
    gpu_name: str,
    model_config: ModelConfig,
    optimal_concurrency: int,
    target_throughput: bool = True,
    precision: str = "fp16",
    sequence_length: int = 2048,
) -> list[TuningConfig]:
    """
    Generate SGLang server and client configurations for tuning using GPU specifications.

    Args:
        num_gpus: Number of GPUs available
        gpu_name: GPU model name
        model_config: Model configuration
        optimal_concurrency: Optimal concurrency level from performance estimation
        target_throughput: Whether to optimize for throughput (True) or latency (False)
        precision: Model precision ("fp16" or "fp8")
        sequence_length: Typical sequence length for calculations

    Returns:
        List of tuning configurations to try
    """
    configs = []

    # Get GPU specifications
    gpu_specs = get_gpu_specs(gpu_name)

    # Calculate optimal parameters based on GPU specs and precision
    max_seqs_configs = calculate_optimal_max_seqs(
        gpu_specs, model_config, precision, sequence_length, optimal_concurrency, target_throughput
    )
    chunked_prefill_size = calculate_chunked_prefill_size(
        gpu_specs, model_config, precision, target_throughput
    )
    memory_fraction = calculate_memory_fraction(gpu_specs, model_config, precision, conservative=True)

    # Base configuration - conservative settings
    base_server_args = []
    base_client_args = ["num_prompts=1000"]

    if num_gpus > 1:
        # For multi-GPU: prefer data parallelism for throughput
        if target_throughput:
            base_server_args.extend([f"dp_size={num_gpus}", "tp_size=1"])
        else:
            # For latency: use tensor parallelism
            base_server_args.extend(
                [
                    "dp_size=1",
                    f"tp_size={min(num_gpus, 8)}",  # Cap TP size
                ]
            )

    # Configuration 1: Conservative baseline
    configs.append(
        TuningConfig(
            framework="sglang",
            server_args=base_server_args
            + [
                "schedule_conservativeness=1.0",
                f"chunked_prefill_size={chunked_prefill_size}",
                f"max_running_requests={max_seqs_configs['conservative']}",
                f"mem_fraction_static={memory_fraction:.2f}",
            ],
            client_args=base_client_args + [f"max_concurrency={optimal_concurrency // 2}"],
            description=f"Conservative baseline - {chunked_prefill_size} prefill size, stable performance",
        )
    )

    # Configuration 2: Aggressive throughput
    if target_throughput:
        aggressive_memory = calculate_memory_fraction(gpu_specs, model_config, precision, conservative=False)
        aggressive_prefill = min(chunked_prefill_size * 2, 16384)

        configs.append(
            TuningConfig(
                framework="sglang",
                server_args=base_server_args
                + [
                    "schedule_conservativeness=0.3",
                    f"chunked_prefill_size={aggressive_prefill}",
                    f"max_running_requests={max_seqs_configs['aggressive']}",
                    "schedule_policy=fcfs",
                    f"mem_fraction_static={aggressive_memory:.2f}",
                ],
                client_args=base_client_args + [f"max_concurrency={optimal_concurrency}"],
                description=f"Aggressive throughput - {aggressive_prefill} prefill size, maximum request intake",
            )
        )

    # Configuration 3: Memory optimized
    memory_prefill = max(1024, chunked_prefill_size // 2)
    configs.append(
        TuningConfig(
            framework="sglang",
            server_args=base_server_args
            + [
                "schedule_conservativeness=1.2",
                f"chunked_prefill_size={memory_prefill}",
                f"mem_fraction_static={memory_fraction * 0.9:.2f}",
                f"max_running_requests={max_seqs_configs['memory_efficient']}",
            ],
            client_args=base_client_args + [f"max_concurrency={optimal_concurrency // 4}"],
            description=f"Memory optimized - {memory_prefill} prefill size, reduced OOM risk",
        )
    )

    # Configuration 4: Latency optimized
    if not target_throughput:
        configs.append(
            TuningConfig(
                framework="sglang",
                server_args=base_server_args
                + [
                    "schedule_conservativeness=0.8",
                    f"chunked_prefill_size={chunked_prefill_size}",
                    f"max_running_requests={max_seqs_configs['latency_optimized']}",
                    "schedule_policy=lpm",  # Better prefix matching for latency
                    f"mem_fraction_static={memory_fraction:.2f}",
                ],
                client_args=["num_prompts=1000", f"max_concurrency={max_seqs_configs['latency_optimized']}"],
                description=f"Latency optimized - {chunked_prefill_size} prefill size, minimal queue waiting",
            )
        )

    return configs


def generate_vllm_configs(
    num_gpus: int,
    gpu_name: str,
    model_config: ModelConfig,
    optimal_concurrency: int,
    target_throughput: bool = True,
    precision: str = "fp16",
    sequence_length: int = 2048,
) -> list[TuningConfig]:
    """
    Generate vLLM server and client configurations for tuning using GPU specifications.

    Args:
        num_gpus: Number of GPUs available
        gpu_name: GPU model name
        model_config: Model configuration
        optimal_concurrency: Optimal concurrency level from performance estimation
        target_throughput: Whether to optimize for throughput (True) or latency (False)
        precision: Model precision ("fp16" or "fp8")
        sequence_length: Typical sequence length for calculations

    Returns:
        List of tuning configurations to try
    """
    configs = []

    # Get GPU specifications
    gpu_specs = get_gpu_specs(gpu_name)

    # Calculate optimal parameters based on GPU specs and precision
    optimal_batch_tokens = calculate_optimal_batch_tokens(
        gpu_specs, model_config, precision, sequence_length
    )
    max_seqs_configs = calculate_optimal_max_seqs(
        gpu_specs, model_config, precision, sequence_length, optimal_concurrency, target_throughput
    )

    # Calculate different batch token sizes
    large_batch_tokens = optimal_batch_tokens
    medium_batch_tokens = max(1024, optimal_batch_tokens // 2)
    small_batch_tokens = max(1024, optimal_batch_tokens // 4)

    # Base configuration
    conservative_memory = calculate_memory_fraction(gpu_specs, model_config, precision, conservative=True)
    aggressive_memory = calculate_memory_fraction(gpu_specs, model_config, precision, conservative=False)

    base_server_args = []
    base_client_args = ["num_prompts=1000"]

    if num_gpus > 1:
        base_server_args.append(f"tensor_parallel_size={num_gpus}")

    # Configuration 1: High throughput
    if target_throughput:
        configs.append(
            TuningConfig(
                framework="vllm",
                server_args=base_server_args
                + [
                    f"max_num_batched_tokens={large_batch_tokens}",
                    f"max_num_seqs={max_seqs_configs['aggressive']}",
                    f"gpu_memory_utilization={aggressive_memory:.2f}",
                ],
                client_args=base_client_args + [f"max_concurrency={optimal_concurrency}"],
                description=f"High throughput - {large_batch_tokens} batch tokens, aggressive memory usage",
            )
        )

    # Configuration 2: Balanced performance
    configs.append(
        TuningConfig(
            framework="vllm",
            server_args=base_server_args
            + [
                f"max_num_batched_tokens={medium_batch_tokens}",
                f"max_num_seqs={max_seqs_configs['balanced']}",
                f"gpu_memory_utilization={conservative_memory:.2f}",
            ],
            client_args=base_client_args + [f"max_concurrency={optimal_concurrency // 2}"],
            description=f"Balanced - {medium_batch_tokens} batch tokens, moderate memory usage",
        )
    )

    # Configuration 3: Low latency
    if not target_throughput:
        configs.append(
            TuningConfig(
                framework="vllm",
                server_args=base_server_args
                + [
                    f"max_num_batched_tokens={small_batch_tokens}",
                    f"max_num_seqs={max_seqs_configs['latency_optimized']}",
                    f"gpu_memory_utilization={conservative_memory * 0.9:.2f}",
                ],
                client_args=["num_prompts=1000", f"max_concurrency={max_seqs_configs['latency_optimized']}"],
                description=f"Low latency - {small_batch_tokens} batch tokens for quick response",
            )
        )

    # Configuration 4: Memory efficient
    configs.append(
        TuningConfig(
            framework="vllm",
            server_args=base_server_args
            + [
                f"max_num_batched_tokens={small_batch_tokens}",
                f"max_num_seqs={max_seqs_configs['memory_efficient']}",
                f"gpu_memory_utilization={conservative_memory * 0.9:.2f}",
            ],
            client_args=base_client_args + [f"max_concurrency={optimal_concurrency // 8}"],
            description=f"Memory efficient - {small_batch_tokens} batch tokens, conservative usage",
        )
    )

    # Configuration 5: For large models (if parameters suggest it)
    if model_config.num_params > 30:  # Models larger than 30B
        if num_gpus >= 2:
            configs.append(
                TuningConfig(
                    framework="vllm",
                    server_args=[
                        f"tensor_parallel_size={num_gpus}",
                        f"gpu_memory_utilization={aggressive_memory:.2f}",
                        f"max_num_batched_tokens={small_batch_tokens}",
                        f"max_num_seqs={min(max_seqs_configs['conservative'], 64)}",
                    ],
                    client_args=base_client_args
                    + [f"max_concurrency={optimal_concurrency // 4}"],
                    description=f"Large model optimized - {small_batch_tokens} batch tokens with TP",
                )
            )

    return configs


def generate_simple_tuning_configs(
    framework: str,
    num_gpus: int,
    gpu_name: str,
    model_config: ModelConfig,
    optimal_concurrency: int,
    precision: str = "fp16",
    sequence_length: int = 2048,
) -> list[TuningConfig]:
    """
    Generate simplified tuning configurations focusing only on critical parameters:
    - max_concurrency for client_args: 3 values [n/2, n, n+n/2]
    - tp/dp for server_args: Only if num_gpus > 1
    - Let frameworks use their default memory utilization

    Args:
        framework: Framework name ("sglang" or "vllm")
        num_gpus: Number of GPUs available
        gpu_name: GPU model name
        model_config: Model configuration
        optimal_concurrency: Optimal concurrency level
        precision: Model precision
        sequence_length: Typical sequence length

    Returns:
        List of simplified tuning configurations
    """
    configs = []

    # Generate 3 concurrency values
    concurrency_values = generate_concurrency_range_3_values(optimal_concurrency)

    if num_gpus > 1:
        # Generate TP/DP combinations for multi-GPU
        gpu_specs = get_gpu_specs(gpu_name)
        min_tp_size = calculate_min_tensor_parallel_size(model_config, gpu_specs, precision)
        tp_dp_combinations = generate_tp_dp_combinations(num_gpus, min_tp_size)

        # One config per TP/DP combination
        for tp_size, dp_size in tp_dp_combinations:
            server_args = []

            if framework.lower() == "sglang":
                server_args.extend([f"tp_size={tp_size}", f"dp_size={dp_size}"])
                config_desc = f"Simple - TP:{tp_size}/DP:{dp_size}"
            elif framework.lower() == "vllm":
                server_args.append(f"tensor_parallel_size={tp_size}")
                config_desc = f"Simple - TP:{tp_size}"

            client_args = [
                "num_prompts=1000",
                f"max_concurrency=[{','.join(map(str, concurrency_values))}]"
            ]

            configs.append(TuningConfig(
                framework=framework,
                server_args=server_args,
                client_args=client_args,
                description=config_desc
            ))
    else:
        # Single GPU - only vary concurrency
        client_args = [
            "num_prompts=1000",
            f"max_concurrency=[{','.join(map(str, concurrency_values))}]"
        ]

        configs.append(TuningConfig(
            framework=framework,
            server_args=[],  # No server args for single GPU
            client_args=client_args,
            description="Simple - Single GPU"
        ))

    return configs


def generate_advanced_tuning_configs(
    framework: str,
    num_gpus: int,
    gpu_name: str,
    model_config: ModelConfig,
    optimal_concurrency: int,
    precision: str = "fp16",
    sequence_length: int = 2048,
) -> list[TuningConfig]:
    """
    Generate advanced tuning configurations with additional server parameters.

    Stage 2: Advanced tuning adds more server parameters with 3 values each:
    - Includes all simple tuning parameters
    - Adds key server parameters with 3-value ranges
    - SGLang: chunked_prefill_size (max_running_requests fixed at 2048)
    - vLLM: max_num_batched_tokens (max_num_seqs fixed at 2048)
    - Memory utilization uses framework defaults (no variation)

    Args:
        framework: Framework name ("sglang" or "vllm")
        num_gpus: Number of GPUs available
        gpu_name: GPU model name
        model_config: Model configuration
        optimal_concurrency: Optimal concurrency level
        precision: Model precision
        sequence_length: Typical sequence length

    Returns:
        List of advanced tuning configurations
    """
    configs = []
    gpu_specs = get_gpu_specs(gpu_name)

    # Generate 3 concurrency values
    concurrency_values = generate_concurrency_range_3_values(optimal_concurrency)

    # Generate TP/DP combinations (use just the first one for advanced tuning to limit combinations)
    min_tp_size = calculate_min_tensor_parallel_size(model_config, gpu_specs, precision)
    tp_dp_combinations = generate_tp_dp_combinations(num_gpus, min_tp_size) if num_gpus > 1 else [(1, 1)]
    # Use only the first (best) TP/DP combination for advanced tuning
    tp_size, dp_size = tp_dp_combinations[0]

    if framework.lower() == "sglang":
        # Calculate optimal values and generate 3-value ranges
        optimal_chunked_prefill = calculate_chunked_prefill_size(gpu_specs, model_config, precision, target_throughput=True)

        prefill_values = generate_3_value_range(optimal_chunked_prefill, min_val=1024, max_val=16384)

        base_server_args = [
            "schedule_conservativeness=0.3",
            "schedule_policy=fcfs",
            "max_running_requests=2048",  # Fixed large value
        ]

        if num_gpus > 1:
            base_server_args.extend([f"tp_size={tp_size}", f"dp_size={dp_size}"])

        # Generate parameter ranges for advanced tuning (no memory variation)
        server_args = base_server_args + [
            f"chunked_prefill_size=[{','.join(map(str, prefill_values))}]"
        ]

        client_args = [
            "num_prompts=1000",
            f"max_concurrency=[{','.join(map(str, concurrency_values))}]"
        ]

        config_desc = f"Advanced tuning - prefill: {prefill_values}, concurrency: {concurrency_values}"

        configs.append(TuningConfig(
            framework="sglang",
            server_args=server_args,
            client_args=client_args,
            description=config_desc
        ))

    elif framework.lower() == "vllm":
        # Calculate optimal values and generate 3-value ranges
        optimal_batch_tokens = calculate_optimal_batch_tokens(gpu_specs, model_config, precision, sequence_length)

        batch_values = generate_3_value_range(optimal_batch_tokens, min_val=1024, max_val=32768)

        base_server_args = [
            "max_num_seqs=2048",  # Fixed large value
        ]
        if num_gpus > 1:
            base_server_args.append(f"tensor_parallel_size={tp_size}")

        # Generate parameter ranges for advanced tuning (no memory variation)
        server_args = base_server_args + [
            f"max_num_batched_tokens=[{','.join(map(str, batch_values))}]"
        ]

        client_args = [
            "num_prompts=1000",
            f"max_concurrency=[{','.join(map(str, concurrency_values))}]"
        ]

        config_desc = f"Advanced tuning - batch: {batch_values}, concurrency: {concurrency_values}"

        configs.append(TuningConfig(
            framework="vllm",
            server_args=server_args,
            client_args=client_args,
            description=config_desc
        ))

    return configs


def generate_llm_optimizer_commands(
    configs: list[TuningConfig],
    model_id: str,
    input_length: int,
    output_length: int,
    num_gpus: int = 1,
    host: str = "127.0.0.1",
    output_dir: str = "tuning_results",
    constraints: str = None,
) -> list[str]:
    """
    Generate llm-optimizer CLI commands for the given configurations with parameter ranges.

    Args:
        configs: List of tuning configurations
        model_id: HuggingFace model identifier
        input_length: Input sequence length
        output_length: Output sequence length
        num_gpus: Number of GPUs
        host: Server host
        output_dir: Output directory for results
        constraints: SLO constraints string to include in commands

    Returns:
        List of CLI commands to run
    """
    commands = []

    for i, config in enumerate(configs):
        # Separate fixed args from tunable args
        fixed_server_args = []
        tunable_server_args = []
        fixed_client_args = [
            "num_prompts=1000",
            "dataset_name=sharegpt",
            f"random_input={input_length}",
            f"random_output={output_length}",
        ]
        tunable_client_args = []

        # Parse server args to identify tunable parameters
        for arg in config.server_args:
            if "max_num_seqs=" in arg and "[" in arg:
                tunable_server_args.append(arg)
            elif "max_num_batched_tokens=" in arg and "[" in arg:
                tunable_server_args.append(arg)
            elif "chunked_prefill_size=" in arg and "[" in arg:
                tunable_server_args.append(arg)
            elif ("tp_size=" in arg or "dp_size=" in arg) and "[" in arg:
                tunable_server_args.append(arg)
            else:
                fixed_server_args.append(arg)

        # Parse client args to identify tunable parameters
        for arg in config.client_args:
            if "max_concurrency=" in arg and "[" in arg:
                tunable_client_args.append(arg)
            else:
                # Skip if it's already in fixed_client_args
                if not any(arg.startswith(fixed_arg.split("=")[0]) for fixed_arg in fixed_client_args):
                    fixed_client_args.append(arg)

        # Build command parts
        fixed_server_args_str = ";".join(fixed_server_args)
        fixed_client_args_str = ";".join(fixed_client_args)

        # Start building command
        cmd_parts = [
            "llm-optimizer",
            f"--framework {config.framework}",
            f"--model {model_id}",
            f"--gpus {num_gpus}",
            f"--host {host}",
        ]

        # Add server args
        if fixed_server_args_str:
            cmd_parts.append(f'--server-args "{fixed_server_args_str}"')

        # Add tunable server args
        for tunable_arg in tunable_server_args:
            cmd_parts.append(f'--server-args "{tunable_arg}"')

        # Add client args
        if fixed_client_args_str:
            cmd_parts.append(f'--client-args "{fixed_client_args_str}"')

        # Add tunable client args
        for tunable_arg in tunable_client_args:
            cmd_parts.append(f'--client-args "{tunable_arg}"')

        # Add output options
        cmd_parts.extend([
            f"--output-dir {output_dir}",
            f"--output-json {output_dir}/config_{i + 1}_{config.framework}.json"
        ])

        # Add constraints if provided
        if constraints:
            cmd_parts.append(f'--constraints "{constraints}"')

        cmd = " ".join(cmd_parts)
        commands.append(cmd)

    return commands


def get_framework_tuning_configs(
    framework: str,
    num_gpus: int,
    gpu_name: str,
    model_config: ModelConfig,
    optimal_concurrency: int,
    target_throughput: bool = True,
    precision: str = "fp16",
    sequence_length: int = 2048,
) -> list[TuningConfig]:
    """
    Get tuning configurations for a specific framework using GPU specifications.

    Args:
        framework: Framework name ("sglang" or "vllm")
        num_gpus: Number of GPUs
        gpu_name: GPU model name
        model_config: Model configuration
        optimal_concurrency: Optimal concurrency level
        target_throughput: Whether to optimize for throughput
        precision: Model precision ("fp16" or "fp8")
        sequence_length: Typical sequence length for calculations

    Returns:
        List of tuning configurations

    Raises:
        ValueError: If framework is not supported
    """
    if framework.lower() == "sglang":
        return generate_sglang_configs(
            num_gpus=num_gpus,
            gpu_name=gpu_name,
            model_config=model_config,
            optimal_concurrency=optimal_concurrency,
            target_throughput=target_throughput,
            precision=precision,
            sequence_length=sequence_length,
        )
    elif framework.lower() == "vllm":
        return generate_vllm_configs(
            num_gpus=num_gpus,
            gpu_name=gpu_name,
            model_config=model_config,
            optimal_concurrency=optimal_concurrency,
            target_throughput=target_throughput,
            precision=precision,
            sequence_length=sequence_length,
        )
    else:
        raise ValueError(f"Unsupported framework: {framework}. Use 'sglang' or 'vllm'.")


def generate_simplified_throughput_configs(
    framework: str,
    num_gpus: int,
    gpu_name: str,
    model_config: ModelConfig,
    optimal_concurrency: int,
    precision: str = "fp16",
    sequence_length: int = 2048,
    constraints: list = None,
) -> list[TuningConfig]:
    """
    Generate simplified high-throughput configurations with parameter ranges.

    This generates only one optimized configuration per framework with parameter ranges
    for crucial tuning parameters, as specified in the improvement plan.

    Args:
        framework: Framework name ("sglang" or "vllm")
        num_gpus: Number of GPUs
        gpu_name: GPU model name
        model_config: Model configuration
        optimal_concurrency: Optimal concurrency level
        precision: Model precision
        sequence_length: Typical sequence length
        constraints: SLO constraints (for parameter conservativeness)

    Returns:
        List with single optimized TuningConfig containing parameter ranges
    """
    # Get GPU specifications
    gpu_specs = get_gpu_specs(gpu_name)

    # Calculate optimal parameters
    optimal_batch_tokens = calculate_optimal_batch_tokens(
        gpu_specs, model_config, precision, sequence_length
    )

    # Determine conservativeness based on constraints
    if constraints:
        # Use most restrictive constraint's stat_type
        max(
            get_parameter_conservativeness_for_stat_type(c.stat_type)
            for c in constraints
        )

    # Generate parameter ranges
    concurrency_range = generate_concurrency_range(optimal_concurrency)
    batch_token_range = generate_batch_token_range(optimal_batch_tokens)

    # Multi-GPU TP/DP combinations
    min_tp_size = calculate_min_tensor_parallel_size(model_config, gpu_specs, precision)
    tp_dp_combinations = generate_tp_dp_combinations(num_gpus, min_tp_size) if num_gpus > 1 else [(1, 1)]

    configs = []

    if framework.lower() == "vllm":
        # Memory fractions
        aggressive_memory = calculate_memory_fraction(gpu_specs, model_config, precision, conservative=False)

        # Base server args
        server_args = []

        # Add TP/DP combinations if multi-GPU
        if num_gpus > 1:
            tp_options = [str(tp) for tp, dp in tp_dp_combinations]
            server_args.append(f"tensor_parallel_size=[{','.join(tp_options)}]")

        # Add parameter ranges
        server_args.extend([
            f"max_num_batched_tokens=[{','.join(map(str, batch_token_range))}]",
            f"max_num_seqs=[{','.join(map(str, concurrency_range))}]",
            f"gpu_memory_utilization={aggressive_memory:.2f}",
        ])

        # Client args with concurrency range
        client_args = [f"max_concurrency=[{','.join(map(str, concurrency_range))}]"]

        configs.append(TuningConfig(
            framework="vllm",
            server_args=server_args,
            client_args=client_args,
            description=f"High-throughput optimized - batch_tokens: {batch_token_range}, concurrency: {concurrency_range}"
        ))

    elif framework.lower() == "sglang":
        # Calculate chunked prefill size
        chunked_prefill_size = calculate_chunked_prefill_size(
            gpu_specs, model_config, precision, target_throughput=True
        )
        prefill_range = [chunked_prefill_size // 2, chunked_prefill_size, chunked_prefill_size * 2]
        prefill_range = [p for p in prefill_range if 1024 <= p <= 16384]  # Keep in reasonable range

        aggressive_memory = calculate_memory_fraction(gpu_specs, model_config, precision, conservative=False)

        # Base server args
        server_args = []

        # Add TP/DP combinations if multi-GPU
        if num_gpus > 1:
            tp_options = [str(tp) for tp, dp in tp_dp_combinations]
            dp_options = [str(dp) for tp, dp in tp_dp_combinations]
            server_args.extend([
                f"tp_size=[{','.join(tp_options)}]",
                f"dp_size=[{','.join(dp_options)}]",
            ])

        # Add parameter ranges
        server_args.extend([
            "schedule_conservativeness=0.3",  # Aggressive for throughput
            f"chunked_prefill_size=[{','.join(map(str, prefill_range))}]",
            f"max_running_requests=[{','.join(map(str, concurrency_range))}]",
            f"mem_fraction_static={aggressive_memory:.2f}",
            "schedule_policy=fcfs",
        ])

        # Client args with concurrency range
        client_args = [f"max_concurrency=[{','.join(map(str, concurrency_range))}]"]

        configs.append(TuningConfig(
            framework="sglang",
            server_args=server_args,
            client_args=client_args,
            description=f"High-throughput optimized - prefill: {prefill_range}, concurrency: {concurrency_range}"
        ))

    return configs

"""
Parameter tuning configurations for different LLM frameworks.

This module generates server and client arguments for optimal performance
based on hardware specifications and workload requirements.
"""

from dataclasses import dataclass

from llm_optimizer.args import ArgScope, ArgSet, arg_sets_to_arg_str
from llm_optimizer.common import (
    ModelConfig,
    calculate_min_tensor_parallel_size,
    calculate_model_memory_bytes,
    generate_concurrency_range_3_values,
    generate_parameter_range,
    generate_tp_dp_combinations,
)
from llm_optimizer.performance import get_parameter_conservativeness_for_stat_type
from llm_optimizer.predefined import PARAMETER_MAPPINGS
from llm_optimizer.predefined.gpus import get_gpu_specs, get_precision_tflops
from llm_optimizer.resources import (
    GPUResourceManager,
    ModelMemoryCalculator,
)


@dataclass
class TuningConfig:
    """Configuration for parameter tuning using args.py framework."""

    framework: str
    server_arg_sets: list[ArgSet]  # List of server argument sets
    client_arg_sets: list[ArgSet]  # List of client argument sets
    description: str

    @property
    def server_args_str(self) -> str:
        """Convert server ArgSets to argument string for backward compatibility."""
        return arg_sets_to_arg_str(self.server_arg_sets)

    @property
    def client_args_str(self) -> str:
        """Convert client ArgSets to argument string for backward compatibility."""
        return arg_sets_to_arg_str(self.client_arg_sets)


def calculate_optimal_batch_tokens(
    gpu_specs: dict,
    model_config: ModelConfig,
    precision: str,
    sequence_length: int,
    memory_utilization: float = 0.8
) -> int:
    """Calculate optimal batch tokens based on GPU memory and bandwidth."""

    # Initialize memory calculator
    memory_calculator = ModelMemoryCalculator()

    # Calculate model memory
    model_memory_gb = memory_calculator.calculate_model_memory(model_config, precision)
    available_memory_gb = gpu_specs["VRAM_GB"] * memory_utilization - model_memory_gb

    if available_memory_gb <= 0:
        # Model doesn't fit, return minimum
        return 1024

    # Calculate memory breakdown for one token
    memory_breakdown = memory_calculator.calculate_total_memory_needed(
        model_config,
        batch_size=1,
        sequence_length=1,
        model_precision=precision
    )

    # Get per-token memory (KV cache + activations)
    total_memory_per_token = memory_breakdown.kv_cache_per_token_gb * 1e9 + \
                           (memory_breakdown.activation_memory_gb * 1e9 / sequence_length)

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
    model_memory_bytes = calculate_model_memory_bytes(model_config, precision)
    model_memory_gb = model_memory_bytes / (1024**3)
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


def generate_common_base_configs(
    framework: str,
    num_gpus: int,
    gpu_name: str,
    model_config: ModelConfig,
    optimal_concurrency: int,
    target_throughput: bool = True,
    precision: str = "fp16",
    sequence_length: int = 2048,
) -> list[dict]:
    """
    Generate common base configurations that work for both SGLang and vLLM.

    Returns dictionaries with server_arg_sets and client_arg_sets directly.

    Args:
        framework: Framework name ("sglang" or "vllm")
        num_gpus: Number of GPUs available
        gpu_name: GPU model name
        model_config: Model configuration
        optimal_concurrency: Optimal concurrency level
        target_throughput: Whether to optimize for throughput (True) or latency (False)
        precision: Model precision
        sequence_length: Typical sequence length for calculations

    Returns:
        List of dictionaries with server_arg_sets, client_arg_sets, and description
    """
    # Initialize resource manager and get GPU specifications
    gpu_manager = GPUResourceManager()
    gpu_manager.get_total_resources(num_gpus, gpu_name, precision)

    # For backward compatibility, extract gpu_specs dict
    gpu_specs = get_gpu_specs(gpu_name)

    mapping = PARAMETER_MAPPINGS[framework.lower()]

    # Calculate optimal parameters
    max_seqs_configs = calculate_optimal_max_seqs(
        gpu_specs, model_config, precision, sequence_length, optimal_concurrency, target_throughput
    )
    calculate_memory_fraction(gpu_specs, model_config, precision, conservative=True)

    configs = []

    # Configuration 1: Conservative baseline
    server_arg_sets = []
    client_arg_sets = []

    # Add base client args
    client_arg_sets.extend([
        ArgSet(scope=ArgScope.CLIENT, name="num_prompts", arg_type=int, values=[1000]),
        ArgSet(scope=ArgScope.CLIENT, name="max_concurrency", arg_type=int, values=[optimal_concurrency // 2])
    ])

    # Add server args using framework mapping
    max_seqs_param = mapping.get("max_concurrent_requests", "max_concurrent_requests")

    if max_seqs_param:
        server_arg_sets.append(ArgSet(
            scope=ArgScope.SERVER,
            name=max_seqs_param,
            arg_type=int,
            values=[max_seqs_configs['conservative']]
        ))

    # Add multi-GPU parallelization for baseline
    if num_gpus > 1:
        min_tp_size = calculate_min_tensor_parallel_size(model_config, gpu_specs, precision)
        tp_param = mapping.get("tensor_parallel", "tensor_parallel")
        dp_param = mapping.get("data_parallel", "data_parallel")

        if target_throughput:
            # Prefer data parallelism for throughput
            tp_value, dp_value = 1, num_gpus
        else:
            # Use tensor parallelism for latency
            tp_value = min(min_tp_size, num_gpus, 8)  # Cap TP size
            dp_value = 1

        # Create composite ArgSet for TP/DP combination
        if tp_param and dp_param:
            server_arg_sets.append(ArgSet(
                scope=ArgScope.SERVER,
                name=(tp_param, dp_param),
                arg_type=(int, int),
                values=[(tp_value, dp_value)]
            ))

    configs.append({
        "server_arg_sets": server_arg_sets,
        "client_arg_sets": client_arg_sets,
        "description": "Conservative baseline - stable performance",
    })

    # Configuration 2: Aggressive throughput (if targeting throughput)
    if target_throughput:
        server_arg_sets = []
        client_arg_sets = []

        # Add base client args with higher concurrency
        client_arg_sets.extend([
            ArgSet(scope=ArgScope.CLIENT, name="num_prompts", arg_type=int, values=[1000]),
            ArgSet(scope=ArgScope.CLIENT, name="max_concurrency", arg_type=int, values=[optimal_concurrency])
        ])

        # Add server args
        if max_seqs_param:
            server_arg_sets.append(ArgSet(
                scope=ArgScope.SERVER,
                name=max_seqs_param,
                arg_type=int,
                values=[max_seqs_configs['aggressive']]
            ))


        # Add same parallelization strategy as config1
        if num_gpus > 1 and tp_param and dp_param:
            if target_throughput:
                tp_value, dp_value = 1, num_gpus
            else:
                tp_value = min(min_tp_size, num_gpus, 8)
                dp_value = 1

            server_arg_sets.append(ArgSet(
                scope=ArgScope.SERVER,
                name=(tp_param, dp_param),
                arg_type=(int, int),
                values=[(tp_value, dp_value)]
            ))

        configs.append({
            "server_arg_sets": server_arg_sets,
            "client_arg_sets": client_arg_sets,
            "description": "Aggressive throughput - maximum request intake",
        })

    # Configuration 3: Memory optimized
    server_arg_sets = []
    client_arg_sets = []

    # Add base client args with lower concurrency
    client_arg_sets.extend([
        ArgSet(scope=ArgScope.CLIENT, name="num_prompts", arg_type=int, values=[1000]),
        ArgSet(scope=ArgScope.CLIENT, name="max_concurrency", arg_type=int, values=[optimal_concurrency // 4])
    ])

    # Add server args
    if max_seqs_param:
        server_arg_sets.append(ArgSet(
            scope=ArgScope.SERVER,
            name=max_seqs_param,
            arg_type=int,
            values=[max_seqs_configs['memory_efficient']]
        ))


    # Use conservative parallelization for memory config
    if num_gpus > 1 and tp_param and dp_param:
        # Prefer smaller parallelization for memory efficiency
        dp_value = min(2, num_gpus)
        tp_value = num_gpus // dp_value

        server_arg_sets.append(ArgSet(
            scope=ArgScope.SERVER,
            name=(tp_param, dp_param),
            arg_type=(int, int),
            values=[(tp_value, dp_value)]
        ))

    configs.append({
        "server_arg_sets": server_arg_sets,
        "client_arg_sets": client_arg_sets,
        "description": "Memory efficient - conservative memory usage",
    })

    return configs




def add_sglang_specific_params(
    base_configs: list[dict],
    gpu_specs: dict,
    model_config: ModelConfig,
    precision: str,
    target_throughput: bool,
) -> list[dict]:
    """
    Add SGLang-specific parameters to base configurations.

    Args:
        base_configs: Base configurations with server_arg_sets and client_arg_sets
        gpu_specs: GPU specifications
        model_config: Model configuration
        precision: Model precision
        target_throughput: Whether targeting throughput

    Returns:
        Enhanced configurations with SGLang-specific ArgSets added
    """
    enhanced_configs = []

    # Calculate SGLang-specific parameters
    chunked_prefill_size = calculate_chunked_prefill_size(
        gpu_specs, model_config, precision, target_throughput
    )

    for _i, config in enumerate(base_configs):
        enhanced = config.copy()
        # Deep copy the ArgSets lists
        enhanced["server_arg_sets"] = config["server_arg_sets"].copy()
        enhanced["client_arg_sets"] = config["client_arg_sets"].copy()

        if "Conservative" in config["description"]:
            # Conservative config gets standard prefill and conservative scheduling
            enhanced["server_arg_sets"].extend([
                ArgSet(scope=ArgScope.SERVER, name="chunked_prefill_size", arg_type=int, values=[chunked_prefill_size]),
                ArgSet(scope=ArgScope.SERVER, name="schedule_conservativeness", arg_type=float, values=[1.0])
            ])

        elif "Aggressive" in config["description"]:
            # Aggressive config gets larger prefill and aggressive scheduling
            aggressive_prefill = min(chunked_prefill_size * 2, 16384)
            enhanced["server_arg_sets"].extend([
                ArgSet(scope=ArgScope.SERVER, name="chunked_prefill_size", arg_type=int, values=[aggressive_prefill]),
                ArgSet(scope=ArgScope.SERVER, name="schedule_conservativeness", arg_type=float, values=[0.3]),
                ArgSet(scope=ArgScope.SERVER, name="schedule_policy", arg_type=str, values=["fcfs"])
            ])

        elif "Memory" in config["description"]:
            # Memory efficient config gets smaller prefill
            memory_prefill = max(1024, chunked_prefill_size // 2)
            enhanced["server_arg_sets"].extend([
                ArgSet(scope=ArgScope.SERVER, name="chunked_prefill_size", arg_type=int, values=[memory_prefill]),
                ArgSet(scope=ArgScope.SERVER, name="schedule_conservativeness", arg_type=float, values=[1.2])
            ])

        enhanced_configs.append(enhanced)

    return enhanced_configs


def add_vllm_specific_params(
    base_configs: list[dict],
    gpu_specs: dict,
    model_config: ModelConfig,
    precision: str,
    sequence_length: int,
) -> list[dict]:
    """
    Add vLLM-specific parameters to base configurations.

    Args:
        base_configs: Base configurations with server_arg_sets and client_arg_sets
        gpu_specs: GPU specifications
        model_config: Model configuration
        precision: Model precision
        sequence_length: Sequence length for calculations

    Returns:
        Enhanced configurations with vLLM-specific ArgSets added
    """
    enhanced_configs = []

    # Calculate vLLM-specific parameters
    optimal_batch_tokens = calculate_optimal_batch_tokens(
        gpu_specs, model_config, precision, sequence_length
    )

    for _i, config in enumerate(base_configs):
        enhanced = config.copy()
        # Deep copy the ArgSets lists
        enhanced["server_arg_sets"] = config["server_arg_sets"].copy()
        enhanced["client_arg_sets"] = config["client_arg_sets"].copy()

        if "Conservative" in config["description"]:
            # Conservative config gets moderate batch size
            batch_tokens = max(1024, optimal_batch_tokens // 2)
            enhanced["server_arg_sets"].append(
                ArgSet(scope=ArgScope.SERVER, name="max_num_batched_tokens", arg_type=int, values=[batch_tokens])
            )

        elif "Aggressive" in config["description"]:
            # Aggressive config gets large batch size
            enhanced["server_arg_sets"].append(
                ArgSet(scope=ArgScope.SERVER, name="max_num_batched_tokens", arg_type=int, values=[optimal_batch_tokens])
            )

        elif "Memory" in config["description"]:
            # Memory efficient config gets small batch size
            batch_tokens = max(1024, optimal_batch_tokens // 4)
            enhanced["server_arg_sets"].append(
                ArgSet(scope=ArgScope.SERVER, name="max_num_batched_tokens", arg_type=int, values=[batch_tokens])
            )

        enhanced_configs.append(enhanced)

    return enhanced_configs


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
    Generate SGLang server and client configurations for tuning using common base plus SGLang-specific enhancements.

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
    # Get GPU specifications
    gpu_specs = get_gpu_specs(gpu_name)

    # Generate common base configurations
    base_configs = generate_common_base_configs(
        framework="sglang",
        num_gpus=num_gpus,
        gpu_name=gpu_name,
        model_config=model_config,
        optimal_concurrency=optimal_concurrency,
        target_throughput=target_throughput,
        precision=precision,
        sequence_length=sequence_length,
    )

    # Add SGLang-specific parameters
    enhanced_configs = add_sglang_specific_params(
        base_configs=base_configs,
        gpu_specs=gpu_specs,
        model_config=model_config,
        precision=precision,
        target_throughput=target_throughput,
    )

    # Convert to TuningConfig objects
    tuning_configs = []
    for config in enhanced_configs:
        tuning_configs.append(TuningConfig(
            framework="sglang",
            server_arg_sets=config["server_arg_sets"],
            client_arg_sets=config["client_arg_sets"],
            description=config["description"],
        ))

    return tuning_configs



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
    Generate vLLM server and client configurations for tuning using common base plus vLLM-specific enhancements.

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
    # Get GPU specifications
    gpu_specs = get_gpu_specs(gpu_name)

    # Generate common base configurations
    base_configs = generate_common_base_configs(
        framework="vllm",
        num_gpus=num_gpus,
        gpu_name=gpu_name,
        model_config=model_config,
        optimal_concurrency=optimal_concurrency,
        target_throughput=target_throughput,
        precision=precision,
        sequence_length=sequence_length,
    )

    # Add vLLM-specific parameters
    enhanced_configs = add_vllm_specific_params(
        base_configs=base_configs,
        gpu_specs=gpu_specs,
        model_config=model_config,
        precision=precision,
        sequence_length=sequence_length,
    )

    # Convert to TuningConfig objects
    tuning_configs = []
    for config in enhanced_configs:
        tuning_configs.append(TuningConfig(
            framework="vllm",
            server_arg_sets=config["server_arg_sets"],
            client_arg_sets=config["client_arg_sets"],
            description=config["description"],
        ))

    return tuning_configs


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
    Generate simplified tuning configurations using args.py framework:
    - max_concurrency for client_args: 3 values [n/2, n, n+n/2]
    - tp*dp for server_args: Composite argument for multi-GPU
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
        List of simplified tuning configurations using ArgSet format
    """
    configs = []

    # Generate 3 concurrency values
    concurrency_values = generate_concurrency_range_3_values(optimal_concurrency)

    # Client ArgSets - max_concurrency is universal across frameworks
    client_arg_sets = [
        ArgSet(scope=ArgScope.CLIENT, name="num_prompts", arg_type=int, values=[1000]),
        ArgSet(scope=ArgScope.CLIENT, name="max_concurrency", arg_type=int, values=concurrency_values)
    ]

    if num_gpus > 1:
        # Generate TP/DP combinations for multi-GPU
        gpu_specs = get_gpu_specs(gpu_name)
        min_tp_size = calculate_min_tensor_parallel_size(model_config, gpu_specs, precision)
        tp_dp_combinations = generate_tp_dp_combinations(num_gpus, min_tp_size)

        # Create composite ArgSet using framework-specific parameter names
        mapping = PARAMETER_MAPPINGS[framework.lower()]
        tp_param = mapping.get("tensor_parallel", "tensor_parallel")
        dp_param = mapping.get("data_parallel", "data_parallel")

        server_arg_sets = [
            ArgSet(
                scope=ArgScope.SERVER,
                name=(tp_param, dp_param),
                arg_type=(int, int),
                values=tp_dp_combinations
            )
        ]

        config_desc = f"Simple - {framework.upper()} TP/DP: {tp_dp_combinations}"

        configs.append(TuningConfig(
            framework=framework,
            server_arg_sets=server_arg_sets,
            client_arg_sets=client_arg_sets,
            description=config_desc
        ))
    else:
        # Single GPU - no server args needed
        configs.append(TuningConfig(
            framework=framework,
            server_arg_sets=[],  # No server args for single GPU
            client_arg_sets=client_arg_sets,
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
    Generate advanced tuning configurations that inherit from simple configs and add more parameters.

    Stage 2: Advanced tuning inherits all simple tuning base settings and adds more server parameters:
    - Starts with simple tuning configurations as base
    - Adds key server parameters with 3-value ranges
    - SGLang: chunked_prefill_size, schedule_conservativeness, schedule_policy
    - vLLM: max_num_batched_tokens

    Args:
        framework: Framework name ("sglang" or "vllm")
        num_gpus: Number of GPUs available
        gpu_name: GPU model name
        model_config: Model configuration
        optimal_concurrency: Optimal concurrency level
        precision: Model precision
        sequence_length: Typical sequence length

    Returns:
        List of advanced tuning configurations that inherit from simple configs
    """
    # Start by getting the simple configs as the base
    simple_configs = generate_simple_tuning_configs(
        framework=framework,
        num_gpus=num_gpus,
        gpu_name=gpu_name,
        model_config=model_config,
        optimal_concurrency=optimal_concurrency,
        precision=precision,
        sequence_length=sequence_length,
    )

    if not simple_configs:
        return []

    # Use the first simple config as base (they all have the same TP/DP and concurrency structure)
    base_config = simple_configs[0]
    gpu_specs = get_gpu_specs(gpu_name)

    advanced_configs = []

    if framework.lower() == "sglang":
        # Calculate optimal values and generate 3-value ranges for advanced parameters
        optimal_chunked_prefill = calculate_chunked_prefill_size(gpu_specs, model_config, precision, target_throughput=True)
        prefill_values = generate_parameter_range(optimal_chunked_prefill, min_val=1024, max_val=16384)
        conservativeness_values = [0.3, 0.6, 1.0]  # Aggressive to conservative

        # Build advanced server args by extending the base config
        mapping = PARAMETER_MAPPINGS[framework.lower()]
        prefill_param = mapping.get("prefill_chunk_size", "prefill_chunk_size")

        # Copy base ArgSets and add advanced parameters
        server_arg_sets = base_config.server_arg_sets.copy()
        client_arg_sets = base_config.client_arg_sets.copy()

        # Add advanced server parameters
        server_arg_sets.extend([
            ArgSet(scope=ArgScope.SERVER, name=prefill_param, arg_type=int, values=prefill_values),
            ArgSet(scope=ArgScope.SERVER, name="schedule_conservativeness", arg_type=float, values=conservativeness_values),
            ArgSet(scope=ArgScope.SERVER, name="schedule_policy", arg_type=str, values=["fcfs"])
        ])

        advanced_configs.append(TuningConfig(
            framework="sglang",
            server_arg_sets=server_arg_sets,
            client_arg_sets=client_arg_sets,
            description=f"Advanced - SGLang with prefill tuning: {prefill_values}"
        ))

    elif framework.lower() == "vllm":
        # Calculate optimal values and generate 3-value ranges for advanced parameters
        optimal_batch_tokens = calculate_optimal_batch_tokens(gpu_specs, model_config, precision, sequence_length)
        batch_values = generate_parameter_range(optimal_batch_tokens, min_val=1024, max_val=32768)

        # Build advanced server args by extending the base config
        mapping = PARAMETER_MAPPINGS[framework.lower()]
        batch_param = mapping.get("batch_size", "batch_size")

        # Copy base ArgSets and add advanced parameters
        server_arg_sets = base_config.server_arg_sets.copy()
        client_arg_sets = base_config.client_arg_sets.copy()

        # Add advanced server parameters
        server_arg_sets.append(
            ArgSet(scope=ArgScope.SERVER, name=batch_param, arg_type=int, values=batch_values)
        )

        advanced_configs.append(TuningConfig(
            framework="vllm",
            server_arg_sets=server_arg_sets,
            client_arg_sets=client_arg_sets,
            description=f"Advanced - vLLM with batch tuning: {batch_values}"
        ))

    return advanced_configs


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
    Generate llm-optimizer CLI commands using args.py format.

    Args:
        configs: List of tuning configurations with args.py format strings
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
        # Build basic command structure
        cmd_parts = [
            "llm-optimizer",
            f"--framework {config.framework}",
            f"--model {model_id}",
            f"--gpus {num_gpus}",
            f"--host {host}",
        ]

        # Add server args if present
        if config.server_args_str.strip():
            cmd_parts.append(f'--server-args "{config.server_args_str}"')

        # Build client args with fixed parameters
        fixed_client_args = [
            "num_prompts=1000",
            "dataset_name=sharegpt",
            f"random_input={input_length}",
            f"random_output={output_length}",
        ]

        # Combine fixed and tunable client args
        if config.client_args_str.strip():
            client_args_combined = ";".join(fixed_client_args + [config.client_args_str])
        else:
            client_args_combined = ";".join(fixed_client_args)

        cmd_parts.append(f'--client-args "{client_args_combined}"')

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
    concurrency_range = generate_parameter_range(optimal_concurrency)
    batch_token_range = generate_parameter_range(optimal_batch_tokens, min_val=1024, max_val=32768)

    # Multi-GPU TP/DP combinations
    min_tp_size = calculate_min_tensor_parallel_size(model_config, gpu_specs, precision)
    tp_dp_combinations = generate_tp_dp_combinations(num_gpus, min_tp_size) if num_gpus > 1 else [(1, 1)]

    configs = []

    if framework.lower() == "vllm":
        # Base server args
        server_args = []

        # Get parameter mapping
        mapping = PARAMETER_MAPPINGS[framework.lower()]

        # Add TP/DP combinations if multi-GPU
        if num_gpus > 1:
            tp_options = [str(tp) for tp, dp in tp_dp_combinations]
            tp_param = mapping.get("tensor_parallel", "tensor_parallel")
            server_args.append(f"{tp_param}=[{','.join(tp_options)}]")

        # Add parameter ranges
        batch_param = mapping.get("batch_size", "batch_size")
        max_seqs_param = mapping.get("max_concurrent_requests", "max_concurrent_requests")
        server_args.extend([
            f"{batch_param}=[{','.join(map(str, batch_token_range))}]",
            f"{max_seqs_param}=[{','.join(map(str, concurrency_range))}]",
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

        # Base server args
        server_args = []

        # Get parameter mapping
        mapping = PARAMETER_MAPPINGS[framework.lower()]

        # Add TP/DP combinations if multi-GPU
        if num_gpus > 1:
            tp_options = [str(tp) for tp, dp in tp_dp_combinations]
            dp_options = [str(dp) for tp, dp in tp_dp_combinations]
            tp_param = mapping.get("tensor_parallel", "tensor_parallel")
            dp_param = mapping.get("data_parallel", "data_parallel")
            server_args.extend([
                f"{tp_param}=[{','.join(tp_options)}]",
                f"{dp_param}=[{','.join(dp_options)}]",
            ])

        # Add parameter ranges using mapping
        prefill_param = mapping.get("prefill_chunk_size", "prefill_chunk_size")
        max_seqs_param = mapping.get("max_concurrent_requests", "max_concurrent_requests")

        server_args.extend([
            "schedule_conservativeness=0.3",  # Aggressive for throughput
            f"{prefill_param}=[{','.join(map(str, prefill_range))}]",
            f"{max_seqs_param}=[{','.join(map(str, concurrency_range))}]",
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

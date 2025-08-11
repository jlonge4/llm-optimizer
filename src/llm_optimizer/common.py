"""
Common functionality shared between performance estimation and tuning configuration modules.

This module contains shared calculations, utilities, and data structures to avoid
code duplication and ensure consistency across the codebase.
"""

import json
from dataclasses import dataclass
from typing import Optional

from huggingface_hub import hf_hub_download

from llm_optimizer.predefined.gpus import get_gpu_specs, get_precision_tflops


@dataclass
class ModelConfig:
    """Model configuration data structure."""
    num_params: float  # in billions
    num_layers: int
    hidden_dim: int
    num_heads: int
    num_kv_heads: Optional[int] = None
    vocab_size: int = 32000
    inferred_precision: str = "fp16"  # Inferred model precision

    def __post_init__(self):
        # Default num_kv_heads to num_heads if not specified
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads


@dataclass
class GPUResources:
    """GPU resource totals for multi-GPU configurations."""
    total_tflops: float
    total_memory_bandwidth_gb_s: float
    total_vram_gb: float
    single_gpu_vram_gb: float
    num_gpus: int
    gpu_name: str
    precision: str


def get_precision_bytes_per_param(precision: str) -> int:
    """
    Get bytes per parameter for different precisions.

    Args:
        precision: Model precision ("fp16" or "fp8")

    Returns:
        Number of bytes per parameter

    Raises:
        ValueError: If precision is not supported
    """
    precision_map = {
        "fp16": 2,
        "bf16": 2,  # bf16 uses same memory as fp16
        "fp8": 1,
    }

    if precision not in precision_map:
        raise ValueError(f"Unsupported precision: {precision}. Use {list(precision_map.keys())}")

    return precision_map[precision]


def get_precision_multiplier(precision: str) -> float:
    """
    Get multiplier for KV cache calculations based on precision.

    Args:
        precision: Model precision

    Returns:
        Multiplier for KV cache memory calculations
    """
    return 1.0 if precision in ["fp16", "bf16"] else 0.5  # fp8 uses half the memory


def get_total_gpu_resources(num_gpus: int, gpu_name: str, precision: str) -> GPUResources:
    """
    Calculate total GPU resources for multi-GPU configuration.

    Args:
        num_gpus: Number of GPUs
        gpu_name: GPU model name
        precision: Model precision

    Returns:
        GPUResources object with totals
    """
    gpu_specs = get_gpu_specs(gpu_name)
    total_tflops = num_gpus * get_precision_tflops(gpu_name, precision)

    return GPUResources(
        total_tflops=total_tflops,
        total_memory_bandwidth_gb_s=num_gpus * gpu_specs["Memory_Bandwidth_GBs"],
        total_vram_gb=num_gpus * gpu_specs["VRAM_GB"],
        single_gpu_vram_gb=gpu_specs["VRAM_GB"],
        num_gpus=num_gpus,
        gpu_name=gpu_name,
        precision=precision
    )


def get_head_dimension(model_config: ModelConfig) -> int:
    """Get the head dimension for attention calculations."""
    return model_config.hidden_dim // model_config.num_heads


def get_kv_heads_dimension(model_config: ModelConfig) -> int:
    """Get the KV heads dimension for grouped query attention."""
    return model_config.hidden_dim // model_config.num_heads * model_config.num_kv_heads


def calculate_model_memory_gb(
    model_config: ModelConfig,
    precision: str,
    safety_factor: float = 1.0
) -> float:
    """
    Calculate model memory usage in GB.

    Args:
        model_config: Model configuration
        precision: Model precision
        safety_factor: Safety multiplier for overhead

    Returns:
        Model memory usage in GB
    """
    bytes_per_param = get_precision_bytes_per_param(precision)
    model_memory_gb = (model_config.num_params * 1e9 * bytes_per_param) / (1024**3)
    return model_memory_gb * safety_factor


def calculate_kv_cache_memory_per_token(model_config: ModelConfig, precision: str) -> float:
    """
    Calculate KV cache memory per token in bytes.

    Args:
        model_config: Model configuration
        precision: Model precision

    Returns:
        KV cache memory per token in bytes
    """
    head_dim = get_head_dimension(model_config)
    bytes_per_param = get_precision_bytes_per_param(precision)

    # KV cache: 2 (key + value) * num_layers * num_kv_heads * head_dim * bytes_per_param
    kv_cache_per_token = (
        2 * model_config.num_layers * model_config.num_kv_heads * head_dim * bytes_per_param
    )

    return kv_cache_per_token


def calculate_activation_memory_per_token(model_config: ModelConfig, precision: str) -> float:
    """
    Calculate activation memory per token in bytes.

    Args:
        model_config: Model configuration
        precision: Model precision

    Returns:
        Activation memory per token in bytes
    """
    bytes_per_param = get_precision_bytes_per_param(precision)

    # Approximate activation memory per token
    # This is a rough estimate: hidden_dim * layers * multiplier
    activation_multiplier = 4  # Rough estimate for transformer activations
    activation_per_token = (
        model_config.hidden_dim * model_config.num_layers * activation_multiplier * bytes_per_param
    )

    return activation_per_token


def calculate_total_memory_needed_gb(
    model_config: ModelConfig,
    precision: str,
    concurrency: int,
    sequence_length: int,
    safety_factor: float = 1.2
) -> float:
    """
    Calculate total memory needed for model execution.

    Args:
        model_config: Model configuration
        precision: Model precision
        concurrency: Number of concurrent requests
        sequence_length: Average sequence length
        safety_factor: Safety multiplier for overhead

    Returns:
        Total memory needed in GB
    """
    model_memory = calculate_model_memory_gb(model_config, precision)
    kv_cache_per_token = calculate_kv_cache_memory_per_token(model_config, precision)
    activation_per_token = calculate_activation_memory_per_token(model_config, precision)

    # Total KV cache and activation memory
    total_tokens = concurrency * sequence_length
    dynamic_memory_gb = (
        (kv_cache_per_token + activation_per_token) * total_tokens
    ) / (1024**3)

    total_memory = (model_memory + dynamic_memory_gb) * safety_factor
    return total_memory


def calculate_hardware_ops_per_byte(total_tflops: float, total_memory_bandwidth_gb_s: float) -> float:
    """
    Calculate hardware operations per byte for roofline analysis.

    Args:
        total_tflops: Total TFLOPS across all GPUs
        total_memory_bandwidth_gb_s: Total memory bandwidth in GB/s

    Returns:
        Hardware operations per byte
    """
    return (total_tflops * 1e12) / (total_memory_bandwidth_gb_s * 1e9)


def calculate_min_tensor_parallel_size(
    model_config: ModelConfig,
    gpu_specs: dict,
    precision: str,
    safety_factor: float = 1.2
) -> int:
    """
    Calculate minimum tensor parallel size needed to fit model.

    Args:
        model_config: Model configuration
        gpu_specs: GPU specifications dictionary
        precision: Model precision
        safety_factor: Safety factor for memory calculations

    Returns:
        Minimum tensor parallel size
    """
    model_memory_gb = calculate_model_memory_gb(model_config, precision, safety_factor)
    single_gpu_vram = gpu_specs["VRAM_GB"]

    min_tp_size = max(1, int(model_memory_gb // single_gpu_vram) + 1)
    return min_tp_size


def generate_parameter_range(
    optimal_value: int,
    num_values: int = 3,
    variation_factor: float = 0.3,
    min_val: int = 1,
    max_val: Optional[int] = None
) -> list[int]:
    """
    Generate a range of parameter values around an optimal value.

    Args:
        optimal_value: Base value to generate range around
        num_values: Number of values to generate
        variation_factor: How much to vary (as fraction of optimal_value)
        min_val: Minimum allowed value
        max_val: Maximum allowed value (optional)

    Returns:
        List of parameter values
    """
    if num_values == 1:
        return [optimal_value]

    if optimal_value <= 4:
        # For small values, use simple ±1 variation
        variation = 1
    else:
        # For larger values, use percentage-based variation
        variation = max(1, int(optimal_value * variation_factor))

    # Generate symmetric range around optimal
    half_range = (num_values - 1) // 2
    values = []

    for i in range(-half_range, half_range + 1):
        value = optimal_value + i * variation
        value = max(min_val, value)
        if max_val:
            value = min(max_val, value)
        values.append(value)

    # Ensure we have the requested number of unique values
    values = sorted(set(values))

    # If we don't have enough values, add more around the optimal
    while len(values) < num_values and len(values) < 10:  # Prevent infinite loop
        # Add values between existing ones
        new_values = []
        for i in range(len(values) - 1):
            mid = (values[i] + values[i + 1]) // 2
            if mid not in values and mid != values[i] and mid != values[i + 1]:
                new_values.append(mid)

        if new_values:
            values.extend(new_values)
            values = sorted(set(values))
        else:
            break

    return values[:num_values]


def generate_concurrency_range_3_values(optimal_concurrency: int) -> list[int]:
    """
    Generate exactly 3 concurrency values for tuning: [n/2, n, n+n/2].

    Args:
        optimal_concurrency: Base concurrency value

    Returns:
        List of exactly 3 concurrency values
    """
    low = max(1, optimal_concurrency // 2)
    mid = optimal_concurrency
    high = optimal_concurrency + (optimal_concurrency // 2)

    return [low, mid, high]


def generate_tp_dp_combinations(num_gpus: int, min_tp_size: int = 1) -> list[tuple[int, int]]:
    """
    Generate tensor parallel (TP) and data parallel (DP) combinations.

    Args:
        num_gpus: Total number of GPUs
        min_tp_size: Minimum tensor parallel size needed

    Returns:
        List of (tp_size, dp_size) combinations
    """
    combinations = []

    # Generate all valid TP/DP combinations
    for tp_size in range(min_tp_size, num_gpus + 1):
        if num_gpus % tp_size == 0:  # TP must divide evenly
            dp_size = num_gpus // tp_size
            combinations.append((tp_size, dp_size))

    # Sort by TP size (prefer smaller TP for efficiency)
    return sorted(combinations)


def validate_precision(precision: str) -> None:
    """
    Validate precision parameter.

    Args:
        precision: Model precision to validate

    Raises:
        ValueError: If precision is not supported
    """
    valid_precisions = ["fp16", "bf16", "fp8"]
    if precision not in valid_precisions:
        raise ValueError(f"Unsupported precision: {precision}. Use {valid_precisions}")


def validate_model_config(model_config: ModelConfig) -> None:
    """
    Validate model configuration parameters.

    Args:
        model_config: Model configuration to validate

    Raises:
        ValueError: If configuration is invalid
    """
    if model_config.num_params <= 0:
        raise ValueError("Model must have positive number of parameters")

    if model_config.num_layers <= 0:
        raise ValueError("Model must have positive number of layers")

    if model_config.hidden_dim <= 0:
        raise ValueError("Model must have positive hidden dimension")

    if model_config.num_heads <= 0:
        raise ValueError("Model must have positive number of heads")

    if model_config.hidden_dim % model_config.num_heads != 0:
        raise ValueError("Hidden dimension must be divisible by number of heads")

    if model_config.num_kv_heads and model_config.num_kv_heads > model_config.num_heads:
        raise ValueError("KV heads cannot exceed total heads")


def validate_gpu_compatibility(gpu_name: str, precision: str) -> None:
    """
    Validate GPU compatibility with precision.

    Args:
        gpu_name: GPU model name
        precision: Model precision

    Raises:
        ValueError: If combination is not supported
    """
    try:
        get_precision_tflops(gpu_name, precision)
    except ValueError as e:
        raise ValueError(f"GPU {gpu_name} does not support {precision} precision: {e}")


def infer_precision_from_config(config: dict) -> str:
    """
    Infer model precision from HuggingFace config.

    Args:
        config: HuggingFace model config dictionary

    Returns:
        str: Inferred precision ("fp16", "bf16", or "fp8")
    """
    # Check quantization_config field first (highest priority for quantized models)
    quantization_config = config.get("quantization_config")
    if quantization_config:
        # Check for FP8 quantization
        if isinstance(quantization_config, dict):
            # Look for compression method indicating FP8
            quant_method = quantization_config.get("quant_method", "").lower()
            format_name = quantization_config.get("format", "").lower()

            # Common FP8 quantization indicators
            fp8_indicators = [
                "compressed-tensors",
                "fp8",
                "float8",
                "e4m3", "e5m2",  # FP8 formats
                "fbgemm_fp8",
                "float-quantized"
            ]

            if any(indicator in quant_method or indicator in format_name for indicator in fp8_indicators):
                return "fp8"

            # Check for bit configuration indicating FP8
            bits = quantization_config.get("bits")
            weight_bits = quantization_config.get("weight_bits")
            activation_bits = quantization_config.get("activation_bits")

            # 8-bit weights + 8-bit activations often indicates FP8
            if bits == 8 or (weight_bits == 8 and activation_bits == 8):
                return "fp8"

            # Check config groups for bit specifications
            config_groups = quantization_config.get("config_groups", {})
            if isinstance(config_groups, dict):
                for group_config in config_groups.values():
                    if isinstance(group_config, dict):
                        input_acts = group_config.get("input_activations", {})
                        weights = group_config.get("weights", {})

                        # Check if both weights and activations use 8-bit
                        if (isinstance(input_acts, dict) and input_acts.get("num_bits") == 8 and
                            isinstance(weights, dict) and weights.get("num_bits") == 8):
                            return "fp8"

    # Check torch_dtype field
    torch_dtype = config.get("torch_dtype")
    if torch_dtype:
        # Map torch dtypes to our precision names
        dtype_mapping = {
            "float16": "fp16",
            "bfloat16": "bf16",
            "torch.float16": "fp16",
            "torch.bfloat16": "bf16",
            "fp8": "fp8"
        }
        if torch_dtype in dtype_mapping:
            return dtype_mapping[torch_dtype]

    # Check model name/repo for precision hints
    model_name = config.get("_name_or_path", "").lower()
    if "fp8" in model_name:
        return "fp8"
    elif "bf16" in model_name or "bfloat16" in model_name:
        return "bf16"
    elif "fp16" in model_name or "float16" in model_name:
        return "fp16"

    # Check model architecture for precision hints
    config.get("model_type", "").lower()
    architectures = config.get("architectures", [])

    # Some models specify precision in their config content
    config_str = str(config).lower()
    if "fp8" in config_str or "float8" in config_str:
        return "fp8"
    elif "bf16" in config_str or "bfloat16" in config_str:
        return "bf16"

    # Default fallback based on model characteristics
    # Newer/larger models often use bf16, older ones fp16
    if any(arch for arch in architectures if arch and ("llama" in arch.lower() or "mistral" in arch.lower())):
        # Modern LLMs often default to bf16
        return "bf16"

    # Default to fp16 if we can't determine
    return "fp16"


def get_model_config_from_hf(model_id: str) -> ModelConfig:
    """
    Downloads a model's config.json from Hugging Face and extracts configuration.

    Args:
        model_id: HuggingFace model identifier

    Returns:
        ModelConfig object with extracted parameters

    Raises:
        RuntimeError: If config cannot be downloaded or parsed
        KeyError: If required keys are missing from config
    """
    return get_model_config_and_precision_from_hf(model_id)


def get_model_config_and_precision_from_hf(model_id: str) -> ModelConfig:
    """
    Downloads a model's config.json from Hugging Face and extracts configuration with inferred precision.

    Args:
        model_id: HuggingFace model identifier

    Returns:
        ModelConfig object with extracted parameters and inferred precision

    Raises:
        RuntimeError: If config cannot be downloaded or parsed
        KeyError: If required keys are missing from config
    """
    try:
        config_path = hf_hub_download(repo_id=model_id, filename="config.json")
        with open(config_path) as f:
            config = json.load(f)
    except Exception as e:
        raise RuntimeError(
            f"Could not download or read config.json for {model_id}: {e}"
        )

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
        attention_params = n_layers * (
            h * (n_heads * head_dim) + h * (n_kv_heads * head_dim) * 2 + h * h
        )

        # FFN params (assuming SwiGLU)
        ffn_params = n_layers * (h * i * 2 + i * h)

        # Embedding and output params
        embedding_params = v * h
        output_params = v * h if not config.get("tie_word_embeddings", False) else 0

        total_params = attention_params + ffn_params + embedding_params + output_params

        # Infer precision from config
        precision = infer_precision_from_config(config)

        model_config = ModelConfig(
            num_params=total_params / 1e9,  # In billions
            num_layers=n_layers,
            hidden_dim=h,
            vocab_size=v,
            num_heads=n_heads,
            num_kv_heads=n_kv_heads,
            inferred_precision=precision,
        )

        return model_config

    except KeyError as e:
        raise KeyError(f"Could not find required key {e} in config.json for {model_id}")


"""
GPU specifications for LLM performance estimation.
Data compiled from official NVIDIA specifications and technical documentation.
"""

# GPU specifications from official sources and datasheets
# NVIDIA specs researched and verified from NVIDIA official documentation (December 2024)
# AWS Trainium/Inferentia specs verified from AWS Neuron documentation (October 2025)
# Note: AWS specs represent per-chip/device values (each chip contains 2 NeuronCore-v2 cores)
#       Use --num_gpus to specify number of chips/devices for scaling; it defaults to
#       "Chips_Per_Instance", the accelerator count published for that instance SKU.
GPU_SPECS = {
    # AWS Trainium1 - 1 chip
    "TRN1.2XLARGE": {
        "FP16_TFLOPS": 190,
        "BF16_TFLOPS": 190,
        "FP8_TFLOPS": 190,
        "Memory_Bandwidth_GBs": 820,
        "VRAM_GB": 32,
        "Architecture": "Trainium (1st Gen Accelerator)",
        "Memory_Type": "HBM",
        "Chips_Per_Instance": 1,
    },
    # AWS Trainium1 - 16 chips
    "TRN1.32XLARGE": {
        "FP16_TFLOPS": 190,
        "BF16_TFLOPS": 190,
        "FP8_TFLOPS": 190,
        "Memory_Bandwidth_GBs": 820,
        "VRAM_GB": 32,
        "Architecture": "Trainium (1st Gen Accelerator)",
        "Memory_Type": "HBM",
        "Chips_Per_Instance": 16,
    },
    # AWS Trainium1 - 16 chips, 1,600 Gbps EFA
    "TRN1N.32XLARGE": {
        "FP16_TFLOPS": 190,
        "BF16_TFLOPS": 190,
        "FP8_TFLOPS": 190,
        "Memory_Bandwidth_GBs": 820,
        "VRAM_GB": 32,
        "Architecture": "Trainium (1st Gen Accelerator)",
        "Memory_Type": "HBM",
        "Chips_Per_Instance": 16,
    },
    # AWS Inferentia2 - 1 chip
    "INF2.XLARGE": {
        "FP16_TFLOPS": 190,
        "BF16_TFLOPS": 190,
        "FP8_TFLOPS": 190,
        "Memory_Bandwidth_GBs": 820,
        "VRAM_GB": 32,
        "Architecture": "Inferentia2 (2nd Gen Inference Accelerator)",
        "Memory_Type": "HBM",
        "Chips_Per_Instance": 1,
    },
    # AWS Inferentia2 - 1 chip
    "INF2.8XLARGE": {
        "FP16_TFLOPS": 190,
        "BF16_TFLOPS": 190,
        "FP8_TFLOPS": 190,
        "Memory_Bandwidth_GBs": 820,
        "VRAM_GB": 32,
        "Architecture": "Inferentia2 (2nd Gen Inference Accelerator)",
        "Memory_Type": "HBM",
        "Chips_Per_Instance": 1,
    },
    # AWS Inferentia2 - 6 chips
    "INF2.24XLARGE": {
        "FP16_TFLOPS": 190,
        "BF16_TFLOPS": 190,
        "FP8_TFLOPS": 190,
        "Memory_Bandwidth_GBs": 820,
        "VRAM_GB": 32,
        "Architecture": "Inferentia2 (2nd Gen Inference Accelerator)",
        "Memory_Type": "HBM",
        "Chips_Per_Instance": 6,
    },
    # AWS Inferentia2 - 12 chips
    "INF2.48XLARGE": {
        "FP16_TFLOPS": 190,
        "BF16_TFLOPS": 190,
        "FP8_TFLOPS": 190,
        "Memory_Bandwidth_GBs": 820,
        "VRAM_GB": 32,
        "Architecture": "Inferentia2 (2nd Gen Inference Accelerator)",
        "Memory_Type": "HBM",
        "Chips_Per_Instance": 12,
    },
    # AWS Trainium2 - 1 chip
    "TRN2.3XLARGE": {
        "FP16_TFLOPS": 668.75,
        "BF16_TFLOPS": 668.75,
        "FP8_TFLOPS": 1300,
        "Memory_Bandwidth_GBs": 2900,
        "VRAM_GB": 96,
        "Architecture": "Trainium2 (2nd Gen Accelerator)",
        "Memory_Type": "HBM3",
        "Chips_Per_Instance": 1,
    },
    # AWS Trainium2 - 16 chips
    "TRN2.48XLARGE": {
        "FP16_TFLOPS": 668.75,
        "BF16_TFLOPS": 668.75,
        "FP8_TFLOPS": 1300,
        "Memory_Bandwidth_GBs": 2900,
        "VRAM_GB": 96,
        "Architecture": "Trainium2 (2nd Gen Accelerator)",
        "Memory_Type": "HBM3",
        "Chips_Per_Instance": 16,
    },
    # AWS Trainium2 - 16 chips, UltraServer node
    "TRN2U.48XLARGE": {
        "FP16_TFLOPS": 668.75,
        "BF16_TFLOPS": 668.75,
        "FP8_TFLOPS": 1300,
        "Memory_Bandwidth_GBs": 2900,
        "VRAM_GB": 96,
        "Architecture": "Trainium2 (2nd Gen Accelerator)",
        "Memory_Type": "HBM3",
        "Chips_Per_Instance": 16,
    },
    # AWS Trainium2 - 4x trn2u.48xlarge over NeuronLink-v3
    "TRN2-ULTRASERVER": {
        "FP16_TFLOPS": 668.75,
        "BF16_TFLOPS": 668.75,
        "FP8_TFLOPS": 1300,
        "Memory_Bandwidth_GBs": 2900,
        "VRAM_GB": 96,
        "Architecture": "Trainium2 (2nd Gen Accelerator)",
        "Memory_Type": "HBM3",
        "Chips_Per_Instance": 64,  # 42.8 BF16 PFLOPS, 6,144 GiB total
    },
    # AWS Trainium3 - Gen1 UltraServer
    "TRN3U.GEN1": {
        "FP16_TFLOPS": 671,
        "BF16_TFLOPS": 671,
        "FP8_TFLOPS": 2517,
        "Memory_Bandwidth_GBs": 4900,
        "VRAM_GB": 144,
        "Architecture": "Trainium3 (3rd Gen Accelerator)",
        "Memory_Type": "HBM3e",
        "Chips_Per_Instance": 64,  # 42,944 BF16 TFLOPS, 9,216 GiB total
    },
    # AWS Trainium3 - Gen2 UltraServer, 36 servers x 4 devices
    "TRN3U.GEN2": {
        "FP16_TFLOPS": 671,
        "BF16_TFLOPS": 671,
        "FP8_TFLOPS": 2517,
        "Memory_Bandwidth_GBs": 4900,
        "VRAM_GB": 144,
        "Architecture": "Trainium3 (3rd Gen Accelerator)",
        "Memory_Type": "HBM3e",
        "Chips_Per_Instance": 144,  # 96,624 BF16 TFLOPS, 20,736 GiB total
    },
    # NVIDIA Hopper Architecture GPUs
    "H100": {
        "FP16_TFLOPS": 989.5,  # Official NVIDIA specification for SXM5 variant
        "FP8_TFLOPS": 1978.0,  # 4th gen Tensor Cores with FP8 support
        "Memory_Bandwidth_GBs": 3350,  # 3.35 TB/s HBM3 memory bandwidth
        "VRAM_GB": 80,
        "Architecture": "Hopper",
        "Memory_Type": "HBM3",
        "Chips_Per_Instance": 1,
    },
    "H200": {
        "FP16_TFLOPS": 989.0,  # Official NVIDIA specification
        "FP8_TFLOPS": 1978.0,  # Official NVIDIA specification
        "Memory_Bandwidth_GBs": 4800,  # 4.8 TB/s HBM3e memory bandwidth
        "VRAM_GB": 141,
        "Architecture": "Hopper",
        "Memory_Type": "HBM3e",
        "Chips_Per_Instance": 1,
    },
    # NVIDIA Ampere Architecture GPUs
    "A100": {
        "FP16_TFLOPS": 312.0,  # Official NVIDIA specification for 80GB variant
        "FP8_TFLOPS": None,  # Not supported on Ampere architecture
        "Memory_Bandwidth_GBs": 2039,  # 2.039 TB/s HBM2e memory bandwidth
        "VRAM_GB": 80,
        "Architecture": "Ampere",
        "Memory_Type": "HBM2e",
        "Chips_Per_Instance": 1,
    },
    "A100-40GB": {
        "FP16_TFLOPS": 312.0,  # Same compute as 80GB variant
        "FP8_TFLOPS": None,  # Not supported on Ampere architecture
        "Memory_Bandwidth_GBs": 1555,  # 1.555 TB/s HBM2 memory bandwidth
        "VRAM_GB": 40,
        "Architecture": "Ampere",
        "Memory_Type": "HBM2",
        "Chips_Per_Instance": 1,
    },
    # NVIDIA Ada Lovelace Architecture GPUs
    "L20": {
        "FP16_TFLOPS": 119.5,  # Official NVIDIA specification
        "FP8_TFLOPS": 239.0,  # Ada Lovelace supports FP8
        "Memory_Bandwidth_GBs": 864,  # GDDR6 memory bandwidth
        "VRAM_GB": 48,
        "Architecture": "Ada Lovelace",
        "Memory_Type": "GDDR6",
        "Chips_Per_Instance": 1,
    },
    "L40": {
        "FP16_TFLOPS": 181.0,  # Official NVIDIA specification (dense)
        "FP8_TFLOPS": 362.0,  # Ada Lovelace 4th gen Tensor Cores
        "Memory_Bandwidth_GBs": 864,  # GDDR6 memory bandwidth
        "VRAM_GB": 48,
        "Architecture": "Ada Lovelace",
        "Memory_Type": "GDDR6",
        "Chips_Per_Instance": 1,
    },
    # NVIDIA Blackwell Architecture GPUs (Future)
    "B100": {
        "FP16_TFLOPS": 1800.0,  # Official NVIDIA specification (dense)
        "FP8_TFLOPS": 3500.0,  # 5th gen Tensor Cores
        "Memory_Bandwidth_GBs": 8000,  # 8 TB/s HBM3e memory bandwidth
        "VRAM_GB": 192,  # 2x96GB HBM3e stacks
        "Architecture": "Blackwell",
        "Memory_Type": "HBM3e",
        "Chips_Per_Instance": 1,
    },
    "B200": {
        "FP16_TFLOPS": 2250.0,  # Official NVIDIA specification (dense)
        "FP8_TFLOPS": 4500.0,  # 5th gen Tensor Cores enhanced
        "Memory_Bandwidth_GBs": 8000,  # 8 TB/s HBM3e memory bandwidth
        "VRAM_GB": 192,  # 2x96GB HBM3e stacks
        "Architecture": "Blackwell",
        "Memory_Type": "HBM3e",
        "Chips_Per_Instance": 1,
    },
}


def get_gpu_specs(gpu_name: str) -> dict:
    """
    Get GPU specifications by name.

    Args:
        gpu_name: Name of the GPU (e.g., "H100", "h100", "A100", "a100")

    Returns:
        Dictionary containing GPU specifications

    Raises:
        ValueError: If GPU name is not found
    """
    # Normalize to uppercase for lookup
    normalized_name = gpu_name.upper()

    if normalized_name not in GPU_SPECS:
        available = ", ".join(GPU_SPECS.keys())
        available_lower = ", ".join([name.lower() for name in GPU_SPECS.keys()])
        raise ValueError(f"GPU '{gpu_name}' not found. Available GPUs: {available} (case-insensitive: {available_lower})")

    return GPU_SPECS[normalized_name].copy()


def get_chips_per_instance(gpu_name: str) -> int:
    """
    Get the number of accelerators attached to an instance SKU.

    This is 1 for individual GPUs and the chip/device count of the instance or
    UltraServer for AWS Neuron SKUs (e.g. 16 for "trn2.48xlarge").

    Args:
        gpu_name: Name of the GPU or instance SKU (case-insensitive)

    Returns:
        Number of accelerators per instance
    """
    return get_gpu_specs(gpu_name).get("Chips_Per_Instance", 1)


def list_available_gpus() -> list[str]:
    """Return list of available GPU names."""
    return list(GPU_SPECS.keys())


def list_available_gpus_with_lowercase() -> list[str]:
    """Return list of available GPU names including lowercase variants."""
    gpus = list(GPU_SPECS.keys())
    # Add lowercase variants for better tab completion
    gpus.extend([name.lower() for name in GPU_SPECS.keys()])
    return sorted(gpus)


def get_precision_tflops(gpu_name: str, precision: str) -> float:
    """
    Get TFLOPS for a specific precision.

    Args:
        gpu_name: Name of the GPU (case-insensitive)
        precision: "bf16", "fp16" or "fp8"

    Returns:
        TFLOPS value for the precision

    Raises:
        ValueError: If GPU or precision not supported
    """
    specs = get_gpu_specs(gpu_name)

    if precision in ("fp16", "bf16"):
        return specs["FP16_TFLOPS"]
    elif precision == "fp8":
        if specs["FP8_TFLOPS"] is None:
            raise ValueError(f"FP8 precision not supported on {gpu_name}")
        return specs["FP8_TFLOPS"]
    else:
        raise ValueError(f"Unsupported precision: {precision}. Use 'fp16' or 'fp8'")

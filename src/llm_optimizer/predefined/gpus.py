"""
GPU specifications for LLM performance estimation.
Data compiled from official NVIDIA specifications and technical documentation.
"""

# GPU specifications from official NVIDIA sources and datasheets
# Researched and verified from NVIDIA official documentation (December 2024)
GPU_SPECS = {
    # AWS Trainium 1
    "Trn1_Chip": {
        "FP16_TFLOPS": 190,
        "BF16_TFLOPS": 190,
        "cFP8_TFLOPS": 190,
        "TF32_TFLOPS": 190,
        "FP32_TFLOPS": 47.5,
        "INT8_TOPS": 380,
        "Memory_GiB": 32,
        "Memory_Bandwidth_GBs": 820,
        "DMA_Bandwidth_GBs": 1000,
        "Architecture": "Trainium (1st Gen Accelerator)",
        "Memory_Type": "HBM",
        "Cores_Per_Chip": 2,
    },
    # AWS Trainium 1 - trn1.2xlarge Instance
    "TRN1_32XL": {
        "FP16_TFLOPS": 6080,  # 190 * 2 cores * 16 chips
        "cFP8_TFLOPS": 6080,
        "TF32_TFLOPS": 6080,
        "FP32_TFLOPS": 1520,  # 47.5 * 2 cores * 16 chips
        "INT8_TOPS": 12160,  # 380 * 2 cores * 16 chips
        "VRAM_GB": 512,  # 32 * 16 chips
        "Memory_Bandwidth_GBs": 13120,  # 820 * 16 chips
        "Total_DMA_Bandwidth_GBs": 16000,  # 1000 * 16 chips
        "NeuronLink_Bandwidth_GBs": 1000,  # Per chip-to-chip interconnect
        "EFA_Bandwidth_Gbps": 800,  # Inter-node networking
        "Architecture": "Trainium (1st Gen Accelerator)",
        "Memory_Type": "HBM",
        "Total_Chips": 16,
        "Total_Cores": 32,  # 2 cores * 16 chips
        "vCPUs": 128,
    },
    # AWS Trainium 1 - trn1.2xlarge Instance
    "TRN1_2XL": {
        "FP16_TFLOPS": 380,  # 190 * 2 cores * 1 chip
        "BF16_TFLOPS": 380,
        "cFP8_TFLOPS": 380,
        "TF32_TFLOPS": 380,
        "FP32_TFLOPS": 95,  # 47.5 * 2 cores * 1 chip
        "INT8_TOPS": 760,  # 380 * 2 cores * 1 chip
        "VRAM_GB": 32,  # 1 chip
        "Memory_Bandwidth_GBs": 820,  # 1 chip
        "Total_DMA_Bandwidth_GBs": 1000,  # 1 chip
        "Architecture": "Trainium (1st Gen Accelerator)",
        "Memory_Type": "HBM",
        "Total_Chips": 1,
        "Total_Cores": 2,  # 2 cores * 1 chip
        "vCPUs": 8,
        "Instance_Memory_GiB": 32,
    },
    # AWS Inferentia 2
    "INF2_CHIP": {
        "FP16_TFLOPS": 190,
        "BF16_TFLOPS": 190,
        "cFP8_TFLOPS": 190,
        "TF32_TFLOPS": 190,
        "FP32_TFLOPS": 47.5,
        "INT8_TOPS": 380,
        "VRAM_GB": 32,
        "Memory_Bandwidth_GBs": 820,
        "DMA_Bandwidth_GBs": 1000,
        "Architecture": "Inferentia2 (2nd Gen Inference Accelerator)",
        "Memory_Type": "HBM",
        "Cores_Per_Chip": 2,
    },
    # AWS Inferentia 2 - inf2.xlarge Instance
    "INF2_XL": {
        "FP16_TFLOPS": 380,  # 190 * 2 cores * 1 chip
        "BF16_TFLOPS": 380,
        "cFP8_TFLOPS": 380,
        "TF32_TFLOPS": 380,
        "FP32_TFLOPS": 95,  # 47.5 * 2 cores * 1 chip
        "INT8_TOPS": 760,  # 380 * 2 cores * 1 chip
        "VRAM_GB": 32,  # 1 chip
        "Memory_Bandwidth_GBs": 820,  # 1 chip
        "Total_DMA_Bandwidth_GBs": 1000,  # 1 chip
        "Architecture": "Inferentia2 (2nd Gen Inference Accelerator)",
        "Memory_Type": "HBM",
        "Total_Chips": 1,
        "Total_Cores": 2,
        "vCPUs": 4,
        "Instance_Memory_GiB": 16,
        "Network_Bandwidth_Gbps": 15,
        "EBS_Bandwidth_Gbps": 10,
    },
    # AWS Inferentia 2 - inf2.8xlarge Instance
    "INF2_8XL": {
        "FP16_TFLOPS": 380,  # 190 * 2 cores * 1 chip
        "BF16_TFLOPS": 380,
        "cFP8_TFLOPS": 380,
        "TF32_TFLOPS": 380,
        "FP32_TFLOPS": 95,  # 47.5 * 2 cores * 1 chip
        "INT8_TOPS": 760,  # 380 * 2 cores * 1 chip
        "VRAM_GB": 32,  # 1 chip
        "Memory_Bandwidth_GBs": 820,  # 1 chip
        "Total_DMA_Bandwidth_GBs": 1000,  # 1 chip
        "Architecture": "Inferentia2 (2nd Gen Inference Accelerator)",
        "Memory_Type": "HBM",
        "Total_Chips": 1,
        "Total_Cores": 2,
        "vCPUs": 32,
        "Instance_Memory_GiB": 128,
        "Network_Bandwidth_Gbps": 25,
        "EBS_Bandwidth_Gbps": 10,
    },
    # AWS Inferentia 2 - inf2.24xlarge Instance
    "INF2_24XL": {
        "FP16_TFLOPS": 2280,  # 190 * 2 cores * 6 chips
        "BF16_TFLOPS": 2280,
        "cFP8_TFLOPS": 2280,
        "TF32_TFLOPS": 2280,
        "FP32_TFLOPS": 570,  # 47.5 * 2 cores * 6 chips
        "INT8_TOPS": 4560,  # 380 * 2 cores * 6 chips
        "VRAM_GB": 192,  # 32 * 6 chips
        "Memory_Bandwidth_GBs": 4920,  # 820 * 6 chips
        "Total_DMA_Bandwidth_GBs": 6000,  # 1000 * 6 chips
        "NeuronLink_Bandwidth_GBs": 1000,  # Per chip-to-chip interconnect
        "Architecture": "Inferentia2 (2nd Gen Inference Accelerator)",
        "Memory_Type": "HBM",
        "Total_Chips": 6,
        "Total_Cores": 12,
        "vCPUs": 96,
        "Instance_Memory_GiB": 384,
        "Network_Bandwidth_Gbps": 50,
        "EBS_Bandwidth_Gbps": 30,
    },
    # AWS Inferentia 2 - inf2.48xlarge Instance
    "INF2_48XL": {
        "FP16_TFLOPS": 4560,  # 190 * 2 cores * 12 chips
        "BF16_TFLOPS": 4560,
        "cFP8_TFLOPS": 4560,
        "TF32_TFLOPS": 4560,
        "FP32_TFLOPS": 1140,  # 47.5 * 2 cores * 12 chips
        "INT8_TOPS": 9120,  # 380 * 2 cores * 12 chips
        "VRAM_GB": 384,  # 32 * 12 chips
        "Memory_Bandwidth_GBs": 9840,  # 820 * 12 chips
        "Total_DMA_Bandwidth_GBs": 12000,  # 1000 * 12 chips
        "NeuronLink_Bandwidth_GBs": 1000,  # Per chip-to-chip interconnect
        "Architecture": "Inferentia2 (2nd Gen Inference Accelerator)",
        "Memory_Type": "HBM",
        "Total_Chips": 12,
        "Total_Cores": 24,
        "vCPUs": 192,
        "Instance_Memory_GiB": 768,
        "Network_Bandwidth_Gbps": 100,
        "EBS_Bandwidth_Gbps": 60,
    },
   # NVIDIA Hopper Architecture GPUs
    "H100": {
        "FP16_TFLOPS": 989.5,  # Official NVIDIA specification for SXM5 variant
        "FP8_TFLOPS": 1978.0,  # 4th gen Tensor Cores with FP8 support
        "Memory_Bandwidth_GBs": 3350,  # 3.35 TB/s HBM3 memory bandwidth
        "VRAM_GB": 80,
        "Architecture": "Hopper",
        "Memory_Type": "HBM3",
    },
    "H200": {
        "FP16_TFLOPS": 989.0,  # Official NVIDIA specification
        "FP8_TFLOPS": 1978.0,  # Official NVIDIA specification
        "Memory_Bandwidth_GBs": 4800,  # 4.8 TB/s HBM3e memory bandwidth
        "VRAM_GB": 141,
        "Architecture": "Hopper",
        "Memory_Type": "HBM3e",
    },
    # NVIDIA Ampere Architecture GPUs
    "A100": {
        "FP16_TFLOPS": 312.0,  # Official NVIDIA specification for 80GB variant
        "FP8_TFLOPS": None,  # Not supported on Ampere architecture
        "Memory_Bandwidth_GBs": 2039,  # 2.039 TB/s HBM2e memory bandwidth
        "VRAM_GB": 80,
        "Architecture": "Ampere",
        "Memory_Type": "HBM2e",
    },
    "A100-40GB": {
        "FP16_TFLOPS": 312.0,  # Same compute as 80GB variant
        "FP8_TFLOPS": None,  # Not supported on Ampere architecture
        "Memory_Bandwidth_GBs": 1555,  # 1.555 TB/s HBM2 memory bandwidth
        "VRAM_GB": 40,
        "Architecture": "Ampere",
        "Memory_Type": "HBM2",
    },
    # NVIDIA Ada Lovelace Architecture GPUs
    "L20": {
        "FP16_TFLOPS": 119.5,  # Official NVIDIA specification
        "FP8_TFLOPS": 239.0,  # Ada Lovelace supports FP8
        "Memory_Bandwidth_GBs": 864,  # GDDR6 memory bandwidth
        "VRAM_GB": 48,
        "Architecture": "Ada Lovelace",
        "Memory_Type": "GDDR6",
    },
    "L40": {
        "FP16_TFLOPS": 181.0,  # Official NVIDIA specification (dense)
        "FP8_TFLOPS": 362.0,  # Ada Lovelace 4th gen Tensor Cores
        "Memory_Bandwidth_GBs": 864,  # GDDR6 memory bandwidth
        "VRAM_GB": 48,
        "Architecture": "Ada Lovelace",
        "Memory_Type": "GDDR6",
    },
    # NVIDIA Blackwell Architecture GPUs (Future)
    "B100": {
        "FP16_TFLOPS": 1800.0,  # Official NVIDIA specification (dense)
        "FP8_TFLOPS": 3500.0,  # 5th gen Tensor Cores
        "Memory_Bandwidth_GBs": 8000,  # 8 TB/s HBM3e memory bandwidth
        "VRAM_GB": 192,  # 2x96GB HBM3e stacks
        "Architecture": "Blackwell",
        "Memory_Type": "HBM3e",
    },
    "B200": {
        "FP16_TFLOPS": 2250.0,  # Official NVIDIA specification (dense)
        "FP8_TFLOPS": 4500.0,  # 5th gen Tensor Cores enhanced
        "Memory_Bandwidth_GBs": 8000,  # 8 TB/s HBM3e memory bandwidth
        "VRAM_GB": 192,  # 2x96GB HBM3e stacks
        "Architecture": "Blackwell",
        "Memory_Type": "HBM3e",
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

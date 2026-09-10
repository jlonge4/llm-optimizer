"""Framework-specific strategy implementations for parameter generation."""

from abc import ABC, abstractmethod
from typing import Any

from llm_optimizer.args import ArgScope, ArgSet
from llm_optimizer.common import ModelConfig
from llm_optimizer.tuning.core import (
    calculate_chunked_prefill_size,
    calculate_optimal_batch_tokens,
)


class FrameworkStrategy(ABC):
    """Abstract strategy for framework-specific parameter generation."""

    @property
    @abstractmethod
    def framework_name(self) -> str:
        """Get the framework name."""
        pass

    @abstractmethod
    def calculate_optimal_parameters(
        self,
        gpu_specs: dict[str, Any],
        model_config: ModelConfig,
        precision: str,
        sequence_length: int,
        target_throughput: bool = True
    ) -> dict[str, Any]:
        """Calculate framework-specific optimal parameters."""
        pass

    @abstractmethod
    def create_conservative_args(
        self,
        optimal_params: dict[str, Any],
        **kwargs
    ) -> list[ArgSet]:
        """Create conservative server arguments."""
        pass

    @abstractmethod
    def create_aggressive_args(
        self,
        optimal_params: dict[str, Any],
        **kwargs
    ) -> list[ArgSet]:
        """Create aggressive server arguments."""
        pass

    @abstractmethod
    def create_memory_efficient_args(
        self,
        optimal_params: dict[str, Any],
        **kwargs
    ) -> list[ArgSet]:
        """Create memory-efficient server arguments."""
        pass

    @abstractmethod
    def create_advanced_args(
        self,
        optimal_params: dict[str, Any],
        **kwargs
    ) -> list[ArgSet]:
        """Create advanced tuning arguments with parameter ranges."""
        pass

    def enhance_config_by_type(
        self,
        config_dict: dict[str, Any],
        optimal_params: dict[str, Any],
        **kwargs
    ) -> dict[str, Any]:
        """Enhance configuration based on its description type."""
        enhanced = config_dict.copy()
        enhanced["server_arg_sets"] = config_dict["server_arg_sets"].copy()
        enhanced["client_arg_sets"] = config_dict["client_arg_sets"].copy()

        description = config_dict.get("description", "")

        if "Conservative" in description:
            enhanced["server_arg_sets"].extend(
                self.create_conservative_args(optimal_params, **kwargs)
            )
        elif "Aggressive" in description:
            enhanced["server_arg_sets"].extend(
                self.create_aggressive_args(optimal_params, **kwargs)
            )
        elif "Memory" in description:
            enhanced["server_arg_sets"].extend(
                self.create_memory_efficient_args(optimal_params, **kwargs)
            )

        return enhanced


class SGLangStrategy(FrameworkStrategy):
    """Strategy for SGLang-specific parameter generation."""

    @property
    def framework_name(self) -> str:
        return "sglang"

    def calculate_optimal_parameters(
        self,
        gpu_specs: dict[str, Any],
        model_config: ModelConfig,
        precision: str,
        sequence_length: int,
        target_throughput: bool = True
    ) -> dict[str, Any]:
        """Calculate SGLang-specific optimal parameters."""
        chunked_prefill_size = calculate_chunked_prefill_size(
            gpu_specs, model_config, precision, target_throughput
        )

        return {
            "chunked_prefill_size": chunked_prefill_size,
            "conservative_schedule": 1.0,
            "aggressive_schedule": 0.3,
            "memory_schedule": 1.2,
            "schedule_policy": "fcfs"
        }

    def create_conservative_args(
        self,
        optimal_params: dict[str, Any],
        **kwargs
    ) -> list[ArgSet]:
        """Create conservative SGLang server arguments."""
        return [
            ArgSet(
                scope=ArgScope.SERVER,
                name="chunked_prefill_size",
                arg_type=int,
                values=[optimal_params["chunked_prefill_size"]]
            ),
            ArgSet(
                scope=ArgScope.SERVER,
                name="schedule_conservativeness",
                arg_type=float,
                values=[optimal_params["conservative_schedule"]]
            )
        ]

    def create_aggressive_args(
        self,
        optimal_params: dict[str, Any],
        **kwargs
    ) -> list[ArgSet]:
        """Create aggressive SGLang server arguments."""
        aggressive_prefill = min(optimal_params["chunked_prefill_size"] * 2, 16384)

        return [
            ArgSet(
                scope=ArgScope.SERVER,
                name="chunked_prefill_size",
                arg_type=int,
                values=[aggressive_prefill]
            ),
            ArgSet(
                scope=ArgScope.SERVER,
                name="schedule_conservativeness",
                arg_type=float,
                values=[optimal_params["aggressive_schedule"]]
            ),
            ArgSet(
                scope=ArgScope.SERVER,
                name="schedule_policy",
                arg_type=str,
                values=[optimal_params["schedule_policy"]]
            )
        ]

    def create_memory_efficient_args(
        self,
        optimal_params: dict[str, Any],
        **kwargs
    ) -> list[ArgSet]:
        """Create memory-efficient SGLang server arguments."""
        memory_prefill = max(1024, optimal_params["chunked_prefill_size"] // 2)

        return [
            ArgSet(
                scope=ArgScope.SERVER,
                name="chunked_prefill_size",
                arg_type=int,
                values=[memory_prefill]
            ),
            ArgSet(
                scope=ArgScope.SERVER,
                name="schedule_conservativeness",
                arg_type=float,
                values=[optimal_params["memory_schedule"]]
            )
        ]

    def create_advanced_args(
        self,
        optimal_params: dict[str, Any],
        **kwargs
    ) -> list[ArgSet]:
        """Create advanced SGLang tuning arguments with parameter ranges."""
        from llm_optimizer.common import generate_parameter_range
        from llm_optimizer.predefined import PARAMETER_MAPPINGS

        prefill_values = generate_parameter_range(
            optimal_params["chunked_prefill_size"],
            min_val=1024,
            max_val=16384
        )
        conservativeness_values = [0.3, 0.6, 1.0]

        # Get parameter mapping
        mapping = PARAMETER_MAPPINGS[self.framework_name]
        prefill_param = mapping.get("prefill_chunk_size", "chunked_prefill_size")

        return [
            ArgSet(
                scope=ArgScope.SERVER,
                name=prefill_param,
                arg_type=int,
                values=prefill_values
            ),
            ArgSet(
                scope=ArgScope.SERVER,
                name="schedule_conservativeness",
                arg_type=float,
                values=conservativeness_values
            ),
            ArgSet(
                scope=ArgScope.SERVER,
                name="schedule_policy",
                arg_type=str,
                values=["fcfs"]
            )
        ]


class VLLMStrategy(FrameworkStrategy):
    """Strategy for vLLM-specific parameter generation."""

    @property
    def framework_name(self) -> str:
        return "vllm"

    def calculate_optimal_parameters(
        self,
        gpu_specs: dict[str, Any],
        model_config: ModelConfig,
        precision: str,
        sequence_length: int,
        target_throughput: bool = True
    ) -> dict[str, Any]:
        """Calculate vLLM-specific optimal parameters."""
        optimal_batch_tokens = calculate_optimal_batch_tokens(
            gpu_specs, model_config, precision, sequence_length
        )

        return {
            "optimal_batch_tokens": optimal_batch_tokens,
            "conservative_batch": max(1024, optimal_batch_tokens // 2),
            "aggressive_batch": optimal_batch_tokens,
            "memory_batch": max(1024, optimal_batch_tokens // 4)
        }

    def create_conservative_args(
        self,
        optimal_params: dict[str, Any],
        **kwargs
    ) -> list[ArgSet]:
        """Create conservative vLLM server arguments."""
        return [
            ArgSet(
                scope=ArgScope.SERVER,
                name="max_num_batched_tokens",
                arg_type=int,
                values=[optimal_params["conservative_batch"]]
            )
        ]

    def create_aggressive_args(
        self,
        optimal_params: dict[str, Any],
        **kwargs
    ) -> list[ArgSet]:
        """Create aggressive vLLM server arguments."""
        return [
            ArgSet(
                scope=ArgScope.SERVER,
                name="max_num_batched_tokens",
                arg_type=int,
                values=[optimal_params["aggressive_batch"]]
            )
        ]

    def create_memory_efficient_args(
        self,
        optimal_params: dict[str, Any],
        **kwargs
    ) -> list[ArgSet]:
        """Create memory-efficient vLLM server arguments."""
        return [
            ArgSet(
                scope=ArgScope.SERVER,
                name="max_num_batched_tokens",
                arg_type=int,
                values=[optimal_params["memory_batch"]]
            )
        ]

    def create_advanced_args(
        self,
        optimal_params: dict[str, Any],
        **kwargs
    ) -> list[ArgSet]:
        """Create advanced vLLM tuning arguments with parameter ranges."""
        from llm_optimizer.common import generate_parameter_range
        from llm_optimizer.predefined import PARAMETER_MAPPINGS

        batch_values = generate_parameter_range(
            optimal_params["optimal_batch_tokens"],
            min_val=1024,
            max_val=32768
        )

        # Get parameter mapping
        mapping = PARAMETER_MAPPINGS[self.framework_name]
        batch_param = mapping.get("batch_size", "max_num_batched_tokens")

        return [
            ArgSet(
                scope=ArgScope.SERVER,
                name=batch_param,
                arg_type=int,
                values=batch_values
            )
        ]


def _last_single_value(arg_sets: list[ArgSet], name: str) -> Any:
    """Return the value of a single-valued ArgSet by name, or None.

    Later arg sets win, matching how the server applies repeated arguments.
    Multi-valued (swept) arg sets are skipped -- they have no settled value.
    """
    for arg_set in reversed(arg_sets):
        if arg_set.name == name and len(arg_set.values) == 1:
            return arg_set.values[0]
    return None


class VLLMNeuronStrategy(VLLMStrategy):
    """Strategy for the vLLM Neuron plugin on AWS Trainium/Inferentia.

    The plugin keeps vLLM's CLI, so this inherits vLLM's batch-token tuning and
    layers on what Neuron needs: the compiler builds one graph per bucket ahead
    of time, so every config must declare the prefill-token and decode-batch
    buckets it will actually use, and the largest bucket has to match the
    corresponding maximum.
    """

    @property
    def framework_name(self) -> str:
        return "vllm-neuron"

    def enhance_config_by_type(
        self,
        config_dict: dict[str, Any],
        optimal_params: dict[str, Any],
        **kwargs
    ) -> dict[str, Any]:
        """Add the Neuron additional-config on top of the vLLM arguments.

        This runs after the base class has picked the batch-token size and the
        common generator has set max_num_seqs, because the compiled buckets
        have to top out at exactly those two values.
        """
        from llm_optimizer.predefined.vllm_neuron import (
            DEFAULT_BLOCK_SIZE,
            build_additional_config,
        )

        enhanced = super().enhance_config_by_type(
            config_dict, optimal_params, **kwargs
        )
        arg_sets = enhanced["server_arg_sets"]

        max_batched_tokens = _last_single_value(arg_sets, "max_num_batched_tokens")
        max_num_seqs = _last_single_value(arg_sets, "max_num_seqs")

        # Without both maxima the bucket lists can't be pinned to them, so
        # leave the config alone and let the plugin derive its own defaults.
        if max_batched_tokens is None or max_num_seqs is None:
            return enhanced

        arg_sets.extend([
            ArgSet(
                scope=ArgScope.SERVER,
                name="block_size",
                arg_type=int,
                values=[DEFAULT_BLOCK_SIZE]
            ),
            ArgSet(
                scope=ArgScope.SERVER,
                name="additional_config",
                arg_type=str,
                values=[build_additional_config(max_batched_tokens, max_num_seqs)]
            ),
        ])

        return enhanced

    def create_advanced_args(
        self,
        optimal_params: dict[str, Any],
        **kwargs
    ) -> list[ArgSet]:
        """Create advanced Neuron tuning arguments.

        Neuron compiles a graph per bucket, so the token sweep is restricted to
        powers of two -- the shapes the plugin's own bucketing uses. Block size
        is swept over both values Neuron accepts.
        """
        from llm_optimizer.predefined.vllm_neuron import SUPPORTED_BLOCK_SIZES

        arg_sets = super().create_advanced_args(optimal_params, **kwargs)

        for arg_set in arg_sets:
            if arg_set.name == "max_num_batched_tokens":
                arg_set.values = [
                    v for v in arg_set.values if v > 0 and (v & (v - 1)) == 0
                ] or [optimal_params["optimal_batch_tokens"]]

        arg_sets.append(ArgSet(
            scope=ArgScope.SERVER,
            name="block_size",
            arg_type=int,
            values=list(SUPPORTED_BLOCK_SIZES)
        ))

        return arg_sets


def get_strategy_for_framework(framework: str) -> FrameworkStrategy:
    """Get the appropriate strategy for the given framework."""
    framework_lower = framework.lower()

    if framework_lower == "sglang":
        return SGLangStrategy()
    elif framework_lower == "vllm":
        return VLLMStrategy()
    elif framework_lower == "vllm-neuron":
        return VLLMNeuronStrategy()
    else:
        raise ValueError(
            f"Unsupported framework: {framework}. "
            f"Use 'sglang', 'vllm' or 'vllm-neuron'."
        )


def generate_tuning_configs_with_strategy(
    framework: str,
    strategy: FrameworkStrategy,
    model_config: "ModelConfig",
    num_gpus: int,
    gpu_name: str,
    optimal_concurrency: int,
    target_throughput: bool,
    precision: str,
    sequence_length: int,
) -> list["TuningConfig"]:
    """Generate tuning configurations using strategy pattern.

    This is the core template method that orchestrates the 5-step workflow
    for generating framework-specific configurations.
    """
    from llm_optimizer.predefined.gpus import get_gpu_specs
    from llm_optimizer.tuning.core import TuningConfig
    from llm_optimizer.tuning.generation import generate_common_base_configs

    # Step 1: Get GPU specifications
    gpu_specs = get_gpu_specs(gpu_name)

    # Step 2: Generate common base configurations
    base_configs = generate_common_base_configs(
        framework=framework,
        num_gpus=num_gpus,
        gpu_name=gpu_name,
        model_config=model_config,
        optimal_concurrency=optimal_concurrency,
        target_throughput=target_throughput,
        precision=precision,
        sequence_length=sequence_length,
    )

    # Step 3: Calculate framework-specific optimal parameters
    optimal_params = strategy.calculate_optimal_parameters(
        gpu_specs=gpu_specs,
        model_config=model_config,
        precision=precision,
        sequence_length=sequence_length,
        target_throughput=target_throughput
    )

    # Step 4: Enhance configurations with framework-specific parameters
    enhanced_configs = []
    for config in base_configs:
        enhanced = strategy.enhance_config_by_type(
            config_dict=config,
            optimal_params=optimal_params,
            target_throughput=target_throughput,
            sequence_length=sequence_length
        )
        enhanced_configs.append(enhanced)

    # Step 5: Convert to TuningConfig objects
    tuning_configs = []
    for config in enhanced_configs:
        tuning_configs.append(TuningConfig(
            framework=framework,
            server_arg_sets=config["server_arg_sets"],
            client_arg_sets=config["client_arg_sets"],
            description=config["description"],
        ))

    return tuning_configs


def generate_tuning_configs(
    framework: str,
    num_gpus: int,
    gpu_name: str,
    model_config: "ModelConfig",
    optimal_concurrency: int,
    target_throughput: bool = True,
    precision: str = "fp16",
    sequence_length: int = 2048,
) -> list["TuningConfig"]:
    """Generate tuning configurations for any framework using the strategy pattern.

    This is the main entry point for configuration generation.
    """
    # On AWS Neuron devices vLLM serves through the vLLM Neuron plugin
    from llm_optimizer.predefined import framework_for_device

    framework = framework_for_device(framework, gpu_name)

    # Get strategy for framework and generate configurations
    strategy = get_strategy_for_framework(framework)

    return generate_tuning_configs_with_strategy(
        framework=framework,
        strategy=strategy,
        model_config=model_config,
        num_gpus=num_gpus,
        gpu_name=gpu_name,
        optimal_concurrency=optimal_concurrency,
        target_throughput=target_throughput,
        precision=precision,
        sequence_length=sequence_length,
    )

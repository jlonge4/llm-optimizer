# =========================================================================
#  vLLM Neuron Plugin Argument Configurations
# =========================================================================
# The vLLM Neuron plugin (https://github.com/vllm-project/vllm-neuron) is the
# recommended serving path for AWS Trainium and Inferentia. It is a *plugin*
# rather than a fork: it keeps the `vllm serve` CLI and every upstream engine
# argument, and adds Neuron-specific settings under `--additional-config`.
#
# So the server configs are vLLM's, minus the arguments the Neuron platform
# does not implement, plus the knobs that only exist on Neuron.

import json
import typing as t

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llm_optimizer.predefined.vllm import PARAMETER_MAPPING as VLLM_PARAMETER_MAPPING
from llm_optimizer.predefined.vllm import SERVER_CONFIGS as VLLM_SERVER_CONFIGS

# Arguments the Neuron platform does not support. Passing them either raises at
# startup or is silently dropped, so they are removed from the tunable surface.
#   - pipeline parallelism: not implemented by the plugin
#   - chunked prefill: only supported at batch size 1, no prefill/decode mixing
#   - LoRA / sleep mode / KV cache offloading: unsupported features
UNSUPPORTED_ARGS = frozenset({
    "pipeline_parallel_size",
    "enable_chunked_prefill",
    "enable_lora",
    "max_loras",
    "max_lora_rank",
    "lora_extra_vocab_size",
    "lora_dtype",
    "enable_sleep_mode",
    "cpu_offload_gb",
    "swap_space",
})

SERVER_CONFIGS = {
    name: config
    for name, config in VLLM_SERVER_CONFIGS.items()
    if name not in UNSUPPORTED_ARGS
}

PARAMETER_MAPPING = {
    **VLLM_PARAMETER_MAPPING,
    # Neuron compiles a fixed set of shapes, so there is no chunked prefill to
    # size; prefill token counts are controlled by max_num_batched_tokens.
    "prefill_chunk_size": None,
}

# KV cache block size in tokens. Neuron supports 16 and 32 and defaults to 32,
# unlike vLLM's default of 16. Any kv_segment_size_buckets must stay divisible
# by whichever is chosen.
SUPPORTED_BLOCK_SIZES = (16, 32)
DEFAULT_BLOCK_SIZE = 32

# Smallest compiled prefill bucket, per the plugin's bucketing defaults.
MIN_TOKENS_BUCKET = 128


def _validate_buckets(buckets: list[int], max_value: int, field: str) -> list[int]:
    """Apply the plugin's bucket contract.

    Mirrors vllm_neuron.utils.bucket_utils.validate_num_batched_tokens_buckets
    and validate_num_seqs_buckets: non-empty, positive ints, strictly
    ascending, and a last bucket equal to the corresponding maximum.
    """
    if not buckets:
        raise ValueError(f"{field} cannot be empty")

    for i, bucket in enumerate(buckets):
        if bucket <= 0:
            raise ValueError(f"{field}[{i}] must be positive, got {bucket}")

    for i in range(1, len(buckets)):
        if buckets[i] <= buckets[i - 1]:
            raise ValueError(
                f"{field} must be in strictly ascending order, got {buckets}"
            )

    if buckets[-1] != max_value:
        raise ValueError(
            f"Last bucket in {field} must equal {max_value}, got {buckets[-1]}"
        )

    return buckets


class OnDeviceSamplingConfig(BaseModel):
    """Neuron on-device sampling settings."""

    model_config = ConfigDict(extra="forbid")

    all_greedy: t.Optional[bool] = None
    temperature: t.Optional[str] = None
    top_k: t.Optional[int] = None
    top_p: t.Optional[float] = None


class NeuronConfig(BaseModel):
    """The `neuron_config` block of the plugin's `--additional-config`.

    Only the fields llm-optimizer tunes are modelled; the plugin accepts more.
    Validation deliberately matches the plugin's own so a bad config fails here
    rather than after a multi-minute compile on the instance.
    """

    model_config = ConfigDict(extra="allow")

    num_batched_tokens_buckets: list[int] = Field(min_length=1)
    num_seqs_buckets: list[int] = Field(min_length=1)
    on_device_sampling_config: t.Optional[OnDeviceSamplingConfig] = None

    def validate_against(self, max_num_batched_tokens: int, max_num_seqs: int) -> None:
        """Check the buckets against the serving maxima they must top out at.

        Args:
            max_num_batched_tokens: The server's --max-num-batched-tokens
            max_num_seqs: The server's --max-num-seqs

        Raises:
            ValueError: If either bucket list violates the plugin's contract
        """
        _validate_buckets(
            self.num_batched_tokens_buckets,
            max_num_batched_tokens,
            "num_batched_tokens_buckets",
        )
        _validate_buckets(self.num_seqs_buckets, max_num_seqs, "num_seqs_buckets")


class NeuronAdditionalConfig(BaseModel):
    """The full `--additional-config` payload for the vLLM Neuron plugin."""

    model_config = ConfigDict(extra="allow")

    neuron_config: NeuronConfig
    vision_neuron_config: t.Optional[dict[str, t.Any]] = None
    mm_language_model_only: t.Optional[bool] = None

    @model_validator(mode="after")
    def _check_buckets_ascending(self) -> "NeuronAdditionalConfig":
        """Catch ordering problems even without the serving maxima at hand."""
        for field in ("num_batched_tokens_buckets", "num_seqs_buckets"):
            buckets = getattr(self.neuron_config, field)
            _validate_buckets(buckets, buckets[-1], field)
        return self

    @classmethod
    def from_json(cls, raw: str) -> "NeuronAdditionalConfig":
        """Parse an --additional-config JSON string, as the plugin would."""
        return cls.model_validate(json.loads(raw))

    def to_json(self) -> str:
        """Serialize back to the compact JSON the CLI expects."""
        return self.model_dump_json(exclude_none=True)


def build_buckets(max_value: int, min_value: int) -> list[int]:
    """
    Build a power-of-2 bucket list for the Neuron compiler.

    Neuron compiles one graph per bucket and pads inputs up to the nearest one,
    so the list trades startup time against padding waste. The largest bucket
    must equal the corresponding max (`max_num_batched_tokens` for prefill
    tokens, `max_num_seqs` for decode batches) or the plugin rejects it.

    Args:
        max_value: The maximum the buckets must top out at
        min_value: The smallest bucket to compile

    Returns:
        Ascending power-of-2 buckets ending at exactly max_value
    """
    buckets = []
    value = min_value
    while value < max_value:
        buckets.append(value)
        value *= 2
    buckets.append(max_value)
    return buckets


def build_additional_config(
    max_num_batched_tokens: int,
    max_num_seqs: int,
    greedy_sampling: bool = True,
) -> str:
    """
    Build the `--additional-config` JSON for the Neuron plugin.

    Args:
        max_num_batched_tokens: Prefill token budget the buckets must top out at
        max_num_seqs: Decode batch size the buckets must top out at
        greedy_sampling: Sample on device. Async scheduling, speculative
            decoding and structured output all require on-device sampling, so
            this is on by default for benchmarking.

    Returns:
        Compact JSON string suitable for `--additional-config`
    """
    neuron_config = NeuronConfig(
        num_batched_tokens_buckets=build_buckets(
            max_num_batched_tokens, MIN_TOKENS_BUCKET
        ),
        num_seqs_buckets=build_buckets(max_num_seqs, 1),
        on_device_sampling_config=(
            OnDeviceSamplingConfig(all_greedy=True) if greedy_sampling else None
        ),
    )
    neuron_config.validate_against(max_num_batched_tokens, max_num_seqs)

    return NeuronAdditionalConfig(neuron_config=neuron_config).to_json()

"""Tests for the vLLM Neuron plugin serving path.

The point of these tests is the round trip: generate configurations the way the
llm-optimizer CLI does, parse the emitted `--server-args` back out, and check
the result against what the vLLM Neuron plugin actually accepts. Where the
plugin is installed the tests call its own validators, so a drift in its
contract shows up here rather than after a multi-minute compile on a Trainium
instance.
"""

import importlib.util
import json
import os
import pathlib
import shlex
import unittest

import pytest

from llm_optimizer.args import ArgScope, parse_args_str
from llm_optimizer.common import ModelConfig
from llm_optimizer.predefined import SERVER_CONFIGS, framework_for_device
from llm_optimizer.predefined.vllm_neuron import (
    SUPPORTED_BLOCK_SIZES,
    UNSUPPORTED_ARGS,
    NeuronAdditionalConfig,
    build_additional_config,
    build_buckets,
)
from llm_optimizer.tuning.commands import generate_llm_optimizer_commands
from llm_optimizer.tuning.strategy import generate_tuning_configs


def _load_plugin_bucket_utils():
    """Load the plugin's bucket validators, if they can be reached.

    Prefers the installed package. Failing that, falls back to loading
    bucket_utils.py straight out of a source checkout pointed at by
    VLLM_NEURON_SRC -- the module itself only needs the standard library, but
    importing it through the package pulls in vllm, which is not installed on a
    developer laptop.
    """
    try:
        from vllm_neuron.utils import bucket_utils

        return bucket_utils
    except ImportError:
        pass

    src = os.environ.get("VLLM_NEURON_SRC")
    if not src:
        return None

    path = pathlib.Path(src) / "vllm_neuron" / "utils" / "bucket_utils.py"
    if not path.is_file():
        return None

    spec = importlib.util.spec_from_file_location("_neuron_bucket_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PLUGIN = _load_plugin_bucket_utils()

requires_plugin = pytest.mark.skipif(
    PLUGIN is None,
    reason="vllm-neuron not installed and VLLM_NEURON_SRC not set",
)

LLAMA_8B = ModelConfig(
    num_params=8_000_000_000,
    num_layers=32,
    hidden_dim=4096,
    num_heads=32,
    num_kv_heads=8,
)

# (SKU, chips) pairs spanning both Neuron generations and a non-power-of-two
# chip count, which is where invalid TP degrees used to leak through.
NEURON_SKUS = [
    ("inf2.xlarge", 1),
    ("inf2.24xlarge", 6),
    ("inf2.48xlarge", 12),
    ("trn1.32xlarge", 16),
    ("trn2.3xlarge", 1),
    ("trn2.48xlarge", 16),
    ("trn3u.gen1", 64),
]


def generate_for(gpu_name, num_gpus, target_throughput=True):
    """Generate tuning configs the way the estimate command does."""
    return generate_tuning_configs(
        framework="vllm",
        num_gpus=num_gpus,
        gpu_name=gpu_name,
        model_config=LLAMA_8B,
        optimal_concurrency=32,
        target_throughput=target_throughput,
        precision="bf16",
        sequence_length=4096,
    )


def server_args_of(config):
    """Parse a config's server args string back into a name -> value dict.

    This is the receiving half of the round trip: whatever the generator wrote
    has to survive being serialized into the `--server-args` string and parsed
    back, which is exactly what happens between the two CLI invocations.
    """
    arg_sets = parse_args_str(
        config.server_args_str,
        scope=ArgScope.SERVER,
        configs=SERVER_CONFIGS["vllm-neuron"],
        strict=False,
    )
    return {
        arg_set.name: arg_set.values[0]
        for arg_set in arg_sets
        if not isinstance(arg_set.name, tuple) and len(arg_set.values) == 1
    }


class TestFrameworkResolution(unittest.TestCase):
    """A Neuron device has to route to the plugin, and only the plugin."""

    def test_vllm_resolves_to_plugin_on_neuron(self):
        for gpu_name, _ in NEURON_SKUS:
            with self.subTest(gpu=gpu_name):
                self.assertEqual(framework_for_device("vllm", gpu_name), "vllm-neuron")

    def test_vllm_unchanged_on_nvidia(self):
        for gpu_name in ("H100", "A100", "B200"):
            with self.subTest(gpu=gpu_name):
                self.assertEqual(framework_for_device("vllm", gpu_name), "vllm")

    def test_non_neuron_framework_rejected(self):
        for framework in ("sglang", "max"):
            with self.subTest(framework=framework):
                with self.assertRaises(ValueError) as ctx:
                    framework_for_device(framework, "trn2.48xlarge")
                self.assertIn("Neuron", str(ctx.exception))

    def test_generated_configs_carry_plugin_framework(self):
        for config in generate_for("trn2.48xlarge", 16):
            self.assertEqual(config.framework, "vllm-neuron")


class TestBuckets(unittest.TestCase):
    """The bucket lists we emit must satisfy the plugin's contract."""

    def test_buckets_are_powers_of_two_ending_at_max(self):
        self.assertEqual(build_buckets(2048, 128), [128, 256, 512, 1024, 2048])
        self.assertEqual(build_buckets(8, 1), [1, 2, 4, 8])

    def test_non_power_of_two_max_still_ends_at_max(self):
        # The last bucket must equal the max exactly, even when the max is not
        # itself a power of two.
        buckets = build_buckets(3000, 128)
        self.assertEqual(buckets[-1], 3000)
        self.assertEqual(buckets, sorted(set(buckets)))

    def test_single_bucket_when_max_is_minimum(self):
        self.assertEqual(build_buckets(128, 128), [128])
        self.assertEqual(build_buckets(1, 1), [1])

    @requires_plugin
    def test_buckets_match_plugin_defaults(self):
        for max_tokens in (512, 2048, 16384):
            with self.subTest(max_tokens=max_tokens):
                self.assertEqual(
                    build_buckets(max_tokens, 128),
                    PLUGIN.get_default_num_batched_tokens_buckets(max_tokens),
                )
        for max_seqs in (1, 8, 64):
            with self.subTest(max_seqs=max_seqs):
                self.assertEqual(
                    build_buckets(max_seqs, 1),
                    PLUGIN.get_default_num_seqs_buckets(max_seqs),
                )


class TestAdditionalConfigModel(unittest.TestCase):
    """The pydantic model has to accept good payloads and reject bad ones."""

    def test_round_trips_through_json(self):
        raw = build_additional_config(2048, 8)
        model = NeuronAdditionalConfig.from_json(raw)

        self.assertEqual(json.loads(model.to_json()), json.loads(raw))
        self.assertTrue(model.neuron_config.on_device_sampling_config.all_greedy)

    def test_greedy_sampling_can_be_omitted(self):
        model = NeuronAdditionalConfig.from_json(
            build_additional_config(2048, 8, greedy_sampling=False)
        )
        self.assertIsNone(model.neuron_config.on_device_sampling_config)

    def test_rejects_buckets_not_ending_at_max(self):
        model = NeuronAdditionalConfig.from_json(build_additional_config(2048, 8))
        with self.assertRaises(ValueError):
            model.neuron_config.validate_against(4096, 8)
        with self.assertRaises(ValueError):
            model.neuron_config.validate_against(2048, 16)

    def test_rejects_descending_buckets(self):
        with self.assertRaises(ValueError):
            NeuronAdditionalConfig.from_json(
                json.dumps(
                    {
                        "neuron_config": {
                            "num_batched_tokens_buckets": [512, 256, 1024],
                            "num_seqs_buckets": [1, 2, 4],
                        }
                    }
                )
            )

    def test_rejects_empty_buckets(self):
        with self.assertRaises(ValueError):
            NeuronAdditionalConfig.from_json(
                json.dumps(
                    {
                        "neuron_config": {
                            "num_batched_tokens_buckets": [],
                            "num_seqs_buckets": [1],
                        }
                    }
                )
            )


class TestGeneratedConfigsAreNeuronValid(unittest.TestCase):
    """End-to-end: what the generator emits must be servable on Neuron."""

    def test_tensor_parallel_is_always_a_power_of_two(self):
        for gpu_name, num_gpus in NEURON_SKUS:
            for target_throughput in (True, False):
                configs = generate_for(gpu_name, num_gpus, target_throughput)
                for config in configs:
                    args = server_args_of(config)
                    tp = args.get("tensor_parallel_size")
                    if tp is None:
                        continue
                    with self.subTest(gpu=gpu_name, tp=tp):
                        self.assertGreater(tp, 0)
                        self.assertEqual(
                            tp & (tp - 1), 0, f"TP={tp} is not a power of 2"
                        )

    def test_parallelism_fits_the_device_count(self):
        for gpu_name, num_gpus in NEURON_SKUS:
            for config in generate_for(gpu_name, num_gpus):
                args = server_args_of(config)
                tp = args.get("tensor_parallel_size")
                dp = args.get("data_parallel_size")
                if tp is None or dp is None:
                    continue
                with self.subTest(gpu=gpu_name, tp=tp, dp=dp):
                    self.assertLessEqual(tp * dp, num_gpus)

    def test_block_size_is_supported(self):
        for gpu_name, num_gpus in NEURON_SKUS:
            for config in generate_for(gpu_name, num_gpus):
                block_size = server_args_of(config).get("block_size")
                with self.subTest(gpu=gpu_name):
                    self.assertIn(block_size, SUPPORTED_BLOCK_SIZES)

    def test_no_unsupported_arguments_are_emitted(self):
        for gpu_name, num_gpus in NEURON_SKUS:
            for config in generate_for(gpu_name, num_gpus):
                emitted = set(server_args_of(config))
                with self.subTest(gpu=gpu_name):
                    self.assertEqual(emitted & UNSUPPORTED_ARGS, set())

    def test_buckets_top_out_at_the_serving_maxima(self):
        """The check the plugin makes at startup, made here instead."""
        for gpu_name, num_gpus in NEURON_SKUS:
            for config in generate_for(gpu_name, num_gpus):
                args = server_args_of(config)
                raw = args.get("additional_config")
                self.assertIsNotNone(raw, f"{gpu_name}: no additional_config emitted")

                model = NeuronAdditionalConfig.from_json(raw)
                with self.subTest(gpu=gpu_name, description=config.description):
                    model.neuron_config.validate_against(
                        args["max_num_batched_tokens"], args["max_num_seqs"]
                    )

    def test_nvidia_configs_carry_no_neuron_arguments(self):
        for config in generate_tuning_configs(
            framework="vllm",
            num_gpus=8,
            gpu_name="H100",
            model_config=LLAMA_8B,
            optimal_concurrency=32,
            precision="bf16",
            sequence_length=4096,
        ):
            self.assertNotIn("additional_config", server_args_of(config))


class TestCommandRoundTrip(unittest.TestCase):
    """The JSON has to survive the generated shell command intact."""

    def test_additional_config_survives_shell_quoting(self):
        configs = generate_for("trn2.48xlarge", 16)
        commands = generate_llm_optimizer_commands(
            configs,
            model_id="meta-llama/Llama-3.1-8B",
            input_length=1024,
            output_length=256,
            num_gpus=16,
        )

        for command, config in zip(commands, configs):
            # shlex.split is what a shell does to the command line, so the
            # value the second llm-optimizer process receives is this one.
            tokens = shlex.split(command)
            server_args = tokens[tokens.index("--server-args") + 1]

            self.assertEqual(server_args, config.server_args_str)

            arg_sets = parse_args_str(
                server_args,
                scope=ArgScope.SERVER,
                configs=SERVER_CONFIGS["vllm-neuron"],
                strict=False,
            )
            raw = next(
                a.values[0] for a in arg_sets if a.name == "additional_config"
            )
            # Parses as JSON on the far side, not as a Python dict repr.
            self.assertEqual(raw[0], "{")
            NeuronAdditionalConfig.from_json(raw)


class TestAgainstPluginValidators(unittest.TestCase):
    """Validate against the plugin's own code when it is installed."""

    @requires_plugin
    def test_plugin_accepts_every_generated_config(self):
        for gpu_name, num_gpus in NEURON_SKUS:
            for config in generate_for(gpu_name, num_gpus):
                args = server_args_of(config)
                neuron_config = json.loads(args["additional_config"])["neuron_config"]

                with self.subTest(gpu=gpu_name, description=config.description):
                    PLUGIN.validate_num_batched_tokens_buckets(
                        neuron_config["num_batched_tokens_buckets"],
                        args["max_num_batched_tokens"],
                    )
                    PLUGIN.validate_num_seqs_buckets(
                        neuron_config["num_seqs_buckets"], args["max_num_seqs"]
                    )


if __name__ == "__main__":
    unittest.main()

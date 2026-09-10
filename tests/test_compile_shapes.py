"""Tests for compile-shape grouping and server readiness timeouts.

Both exist because of AWS Neuron: the model is compiled to static-shape NEFFs
on first start, which takes tens of minutes. A sweep that restarts the server
for every run recompiles every time, and a five-minute readiness timeout kills
the run before the compile finishes.

These drive the real argument machinery and the real polling loop. Only the
network and the clock are stubbed.
"""

import unittest
from unittest.mock import patch

import requests

from llm_optimizer.args import ArgScope, ArgSet, get_all_arg_combinations
from llm_optimizer.common import construct_benchmark_settings, group_by_compile_shape
from llm_optimizer.server_utils import (
    DEFAULT_READY_TIMEOUT,
    NEURON_READY_TIMEOUT,
    wait_for_server,
)


def server_set(name, values, arg_type=int):
    return ArgSet(scope=ArgScope.SERVER, name=name, arg_type=arg_type, values=values)


def client_set(name, values, arg_type=int):
    return ArgSet(scope=ArgScope.CLIENT, name=name, arg_type=arg_type, values=values)


class TestGroupByCompileShape(unittest.TestCase):
    """Runs sharing a server configuration must share a server."""

    def test_client_only_sweep_is_a_single_shape(self):
        """Varying concurrency alone hits already-compiled buckets."""
        combinations = get_all_arg_combinations(
            server_args_sets=[
                server_set("tensor_parallel_size", [8]),
                server_set("max_num_seqs", [64]),
            ],
            client_args_sets=[client_set("max_concurrency", [1, 2, 4, 8, 16])],
        )

        groups = group_by_compile_shape(combinations)

        self.assertEqual(len(combinations), 5)
        self.assertEqual(len(groups), 1, "one compiled shape should serve all five")
        self.assertEqual(len(next(iter(groups.values()))), 5)

    def test_each_server_value_is_its_own_shape(self):
        """Changing max_num_seqs changes the decode buckets, so it recompiles."""
        combinations = get_all_arg_combinations(
            server_args_sets=[server_set("max_num_seqs", [8, 16, 32])],
            client_args_sets=[client_set("max_concurrency", [4])],
        )

        groups = group_by_compile_shape(combinations)

        self.assertEqual(len(groups), 3)
        for group in groups.values():
            self.assertEqual(len(group), 1)

    def test_server_and_client_sweep_groups_by_server_only(self):
        """The point of the exercise: 12 runs, 3 compiles."""
        combinations = get_all_arg_combinations(
            server_args_sets=[
                server_set("max_num_batched_tokens", [4096, 8192, 16384]),
            ],
            client_args_sets=[client_set("max_concurrency", [1, 8, 32, 64])],
        )

        groups = group_by_compile_shape(combinations)

        self.assertEqual(len(combinations), 12)
        self.assertEqual(len(groups), 3, "one compile per max_num_batched_tokens")
        for group in groups.values():
            self.assertEqual(len(group), 4)
            # Every run in a group must agree on the server args, or they
            # could not share a server.
            server_args = {tuple(sorted(s["server_args"].items())) for s in group}
            self.assertEqual(len(server_args), 1)
            # ...and differ on the client side, or they are duplicates.
            concurrencies = {s["client_args"]["max_concurrency"] for s in group}
            self.assertEqual(concurrencies, {1, 8, 32, 64})

    def test_group_key_is_the_server_argv(self):
        """The key is what actually gets appended to the server command."""
        combinations = get_all_arg_combinations(
            server_args_sets=[server_set("max_num_seqs", [16])],
            client_args_sets=[client_set("max_concurrency", [4])],
        )

        (key,) = group_by_compile_shape(combinations).keys()

        self.assertEqual(key, ("--max-num-seqs=16",))

    def test_grouping_preserves_every_run(self):
        """Nothing may be dropped or duplicated by grouping."""
        combinations = get_all_arg_combinations(
            server_args_sets=[
                server_set("tensor_parallel_size", [1, 2, 4]),
                server_set("max_num_seqs", [8, 64]),
            ],
            client_args_sets=[client_set("max_concurrency", [1, 16])],
        )

        groups = group_by_compile_shape(combinations)
        regrouped = [s for group in groups.values() for s in group]

        self.assertEqual(len(regrouped), len(combinations))
        expected = [construct_benchmark_settings(c) for c in combinations]
        self.assertCountEqual(
            [str(sorted(s["server_args"].items())) for s in regrouped],
            [str(sorted(s["server_args"].items())) for s in expected],
        )

    def test_neuron_bucket_config_separates_shapes(self):
        """Two configs whose only difference is the Neuron bucket JSON."""
        from llm_optimizer.predefined.vllm_neuron import build_additional_config

        combinations = get_all_arg_combinations(
            server_args_sets=[
                server_set("max_num_seqs", [8]),
                ArgSet(
                    scope=ArgScope.SERVER,
                    name="additional_config",
                    arg_type=str,
                    values=[
                        build_additional_config(2048, 8),
                        build_additional_config(4096, 8),
                    ],
                ),
            ],
            client_args_sets=[client_set("max_concurrency", [4])],
        )

        # Different bucket lists are different NEFFs, so they cannot share.
        self.assertEqual(len(group_by_compile_shape(combinations)), 2)

    def test_empty_input(self):
        self.assertEqual(group_by_compile_shape([]), {})


class TestWaitForServer(unittest.TestCase):
    """The readiness poll has to survive a Neuron cold compile."""

    def test_returns_true_once_the_endpoint_answers(self):
        responses = [
            requests.RequestException("refused"),
            requests.RequestException("refused"),
            type("R", (), {"status_code": 200})(),
        ]

        def fake_get(*args, **kwargs):
            item = responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        with patch("llm_optimizer.server_utils.requests.get", side_effect=fake_get):
            with patch("llm_optimizer.server_utils.time.sleep"):
                self.assertTrue(wait_for_server("http://x/health", timeout=60))

        self.assertEqual(responses, [], "should stop polling once ready")

    def test_gives_up_at_the_deadline(self):
        """A short timeout must not poll forever."""
        clock = {"now": 0.0}

        def fake_sleep(seconds):
            clock["now"] += seconds

        with patch(
            "llm_optimizer.server_utils.requests.get",
            side_effect=requests.RequestException("refused"),
        ):
            with patch("llm_optimizer.server_utils.time.sleep", side_effect=fake_sleep):
                with patch(
                    "llm_optimizer.server_utils.time.monotonic",
                    side_effect=lambda: clock["now"],
                ):
                    self.assertFalse(wait_for_server("http://x/health", timeout=30))

        # Deadline honoured rather than overshot: 30s at 5s per attempt.
        self.assertLessEqual(clock["now"], 30)

    def test_long_timeout_keeps_polling_past_the_old_limit(self):
        """The old code stopped at 60 attempts; a Neuron compile outlasts that."""
        attempts = {"n": 0}
        clock = {"now": 0.0}

        def fake_get(*args, **kwargs):
            attempts["n"] += 1
            # Come up well past where the old 60-attempt cap would have failed.
            if attempts["n"] < 200:
                raise requests.RequestException("still compiling")
            return type("R", (), {"status_code": 200})()

        with patch("llm_optimizer.server_utils.requests.get", side_effect=fake_get):
            with patch(
                "llm_optimizer.server_utils.time.sleep",
                side_effect=lambda s: clock.__setitem__("now", clock["now"] + s),
            ):
                with patch(
                    "llm_optimizer.server_utils.time.monotonic",
                    side_effect=lambda: clock["now"],
                ):
                    ready = wait_for_server(
                        "http://x/health", timeout=NEURON_READY_TIMEOUT
                    )

        self.assertTrue(ready)
        self.assertEqual(attempts["n"], 200)
        self.assertGreater(
            clock["now"],
            DEFAULT_READY_TIMEOUT,
            "this compile outlasted the default timeout, which is the point",
        )

    def test_neuron_default_is_much_larger_than_the_general_one(self):
        self.assertEqual(DEFAULT_READY_TIMEOUT, 300)
        self.assertGreaterEqual(NEURON_READY_TIMEOUT, 3600)


if __name__ == "__main__":
    unittest.main()

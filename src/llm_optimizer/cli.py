import json
import pathlib
import time
import typing as t

import click
import pynvml

import llm_optimizer.args as lo_args
import llm_optimizer.bench_client as bench_client
import llm_optimizer.predefined as predefined
from llm_optimizer.logging import get_logger, setup_logging
from llm_optimizer.server_utils import (
    start_server,
    terminate_process_top_down,
)

setup_logging()
logger = get_logger("main")

PREDEFINED_FRAMEWORKS = list(predefined.SERVER_CONFIGS.keys())

def get_gpu_count():
    """Returns the number of available GPUs."""
    try:
        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetCount()
    except pynvml.NVMLError:
        return 0
    finally:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass


def construct_benchmark_settings(combo: list[lo_args.BaseArg]) -> dict[str, t.Any]:
    client_args = [arg for arg in combo if arg.scope == lo_args.ArgScope.CLIENT]
    server_args = [arg for arg in combo if arg.scope == lo_args.ArgScope.SERVER]
    client_kv_pairs = lo_args.get_all_kv_pairs(client_args)
    server_cmd_args = lo_args.get_all_cmd_args(server_args)
    server_kv_pairs = lo_args.get_all_kv_pairs(server_args)
    return {
        "client": dict(client_kv_pairs),
        "server": dict(server_kv_pairs),
        "server_args": server_cmd_args,
    }


@click.command()
@click.option("--server-cmd", type=str, help="The command to start the server.")
@click.option("--model", type=str, help="The model to use.")
@click.option(
    "--framework",
    type=click.Choice(PREDEFINED_FRAMEWORKS),
    help="The framework to use.",
)
@click.option("--server-args", type=str, help="Arguments for the server.", multiple=True)
@click.option("--client-args", type=str, help="Arguments for the client.", multiple=True)
@click.option("--gpus", type=int, help="The number of GPUs to use.")
@click.option("--dry-run", is_flag=True, help="A dry run will not run the command.")
@click.option("--output-dir", default="results", help="Directory to store output files.")
@click.option("--continue", "-c", "continue_flag", is_flag=True, help="Skip configs that already have output files.")
@click.option("--rest", type=int, default=10, help="Rest time in seconds between benchmark runs.")
@click.option("--mute-server", is_flag=True, help="Suppress server process stdout.")
@click.option("--ready-endpoint", default="/health", help="Endpoint to check if server is ready (e.g., /health, /readyz).")
@click.option("--host", type=str, default="127.0.0.1", help="Server host to connect to.")
@click.option("--port", type=int, default=None, help="Server port to connect to.")
def main(server_cmd, model, framework, server_args, client_args, gpus, dry_run, output_dir, continue_flag, rest, mute_server, ready_endpoint, host, port):
    """A CLI tool to optimize LLM performance."""
    if not server_cmd:
        if (not model or not framework):
            raise click.UsageError(
                "If --server-cmd is not provided, both --model and --framework are required."
            )

        if port is None:
            port = {
                "sglang": 30000,
                "vllm": 8000,
            }.get(framework, 30000)

        tmpl = predefined.SEVER_CMD_TMPL[framework]
        server_cmd = tmpl.format(
            model=model,
            host=host,
            port=port,
        )

    if gpus is None:
        gpus = get_gpu_count()

    if dry_run:
        click.echo("Dry run mode enabled.")

    if server_args:
        server_args = ";".join(server_args)
    if client_args:
        client_args = ";".join(client_args)

    server_configs = None
    if framework:
        server_configs = predefined.SERVER_CONFIGS[framework]

    server_args_sets = lo_args.parse_args_str(
        server_args,
        scope=lo_args.ArgScope.SERVER,
        configs=server_configs,
        strict=False,
    )

    client_args_sets = lo_args.parse_args_str(
        client_args,
        scope=lo_args.ArgScope.CLIENT,
        configs=predefined.CLIENT_CONFIGS,
        strict=True,
    )

    all_combinations = lo_args.get_all_arg_combinations(
        client_args_sets=client_args_sets,
        server_args_sets=server_args_sets,
    )

    total_configs = len(all_combinations)
    logger.info(f"Generated {total_configs} configuration(s) to run.")

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ready_url = f"http://{host}:{port}{ready_endpoint}"

    for idx, combo in enumerate(all_combinations):
        benchmark_settings = construct_benchmark_settings(combo)

        client_params = benchmark_settings["client"]
        server_params = benchmark_settings["server"]
        server_args = benchmark_settings["server_args"]

        # Create a descriptive filename for the output
        client_param_strs = [f"{k}-{v}" for k, v in sorted(client_params.items())]
        server_param_strs = [f"{k}-{v}" for k, v in sorted(server_params.items())]

        config_id_parts = []
        if client_param_strs:
            config_id_parts.append("client_" + "-".join(client_param_strs))
        if server_param_strs:
            config_id_parts.append("server_" + "-".join(server_param_strs))

        config_id = "_".join(config_id_parts) or "default"
        output_file_path = output_dir / f"{config_id}.json"

        logger.info("-" * 80)
        logger.info(f"Starting run {idx+1}/{total_configs}: {config_id}")

        if continue_flag and output_file_path.exists():
            logger.info(f"Skipping as output file already exists: {output_file_path}")
            continue

        if dry_run:
            print(benchmark_settings)
            continue

        # Build Server Command & Start Server
        server_process = None
        full_server_cmd = f"{server_cmd} {' '.join(server_args)}"

        try:
            server_process = start_server(full_server_cmd, {}, ready_url, mute_server)

            # Run Benchmark
            benchmark_args = {
                "backend": framework,
                "model": model,
                "host": host,
                "port": port,
                "dataset_name": "sharegpt", # default value
                "num_prompts": 1000, # default value
                "request_rate": float("inf"), # default value
                "seed": 1, # default value
            }
            benchmark_args.update(client_params)
            benchmark_result = bench_client.run_benchmark(benchmark_args)

            with open(output_file_path, "w") as f:
                json.dump(benchmark_result, f, indent=2)
            logger.info(f"Benchmark results saved to {output_file_path}")

        except Exception as e:
            logger.error(f"Error during run for config {config_id}: {e}")

        finally:
            # Clean up
            if server_process:
                terminate_process_top_down(server_process)

            if idx < total_configs - 1:
                logger.info(f"Resting for {rest} seconds before the next run.")
                time.sleep(rest)

    logger.info("-" * 80)
    logger.info("All benchmark runs completed.")


if __name__ == "__main__":
    main()

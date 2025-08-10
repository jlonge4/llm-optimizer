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
from llm_optimizer.performance import (
    calculate_concurrency_limits,
    estimate_performance_under_constraints,
    find_best_performance,
    find_optimal_concurrency_threshold,
    get_model_config_from_hf,
    parse_slo_constraints,
)
from llm_optimizer.predefined.gpus import list_available_gpus
from llm_optimizer.server_utils import (
    start_server,
    terminate_process_top_down,
)
from llm_optimizer.tuning import (
    generate_llm_optimizer_commands,
    get_framework_tuning_configs,
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


def detect_gpu_type():
    """Detect the GPU type from the system."""
    try:
        pynvml.nvmlInit()
        if pynvml.nvmlDeviceGetCount() == 0:
            return None

        # Get the first GPU
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")

        # Map GPU names to our standardized names
        gpu_mapping = {
            "NVIDIA H100": "H100",
            "NVIDIA H200": "H200",
            "NVIDIA A100": "A100",
            "NVIDIA L20": "L20",
            "NVIDIA L40": "L40",
        }

        for full_name, short_name in gpu_mapping.items():
            if full_name in gpu_name:
                return short_name

        # Try to extract model from name
        if "H100" in gpu_name:
            return "H100"
        elif "H200" in gpu_name:
            return "H200"
        elif "A100" in gpu_name:
            return "A100"
        elif "L20" in gpu_name:
            return "L20"
        elif "L40" in gpu_name:
            return "L40"
        elif "B100" in gpu_name:
            return "B100"
        elif "B200" in gpu_name:
            return "B200"

        return None

    except Exception:
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except:
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


def get_config_id(client_params: dict, server_params: dict) -> str:
    """Create a descriptive ID for a configuration."""
    client_param_strs = [f"{k}-{v}" for k, v in sorted(client_params.items())]
    server_param_strs = [f"{k}-{v}" for k, v in sorted(server_params.items())]

    config_id_parts = []
    if client_param_strs:
        config_id_parts.append("client_" + "-".join(client_param_strs))
    if server_param_strs:
        config_id_parts.append("server_" + "-".join(server_param_strs))

    return "_".join(config_id_parts) or "default"


@click.group(invoke_without_command=True)
@click.option("--server-cmd", type=str, help="The command to start the server.")
@click.option("--model", type=str, help="The model to use.")
@click.option(
    "--framework",
    type=click.Choice(PREDEFINED_FRAMEWORKS),
    help="The framework to use.",
)
@click.option(
    "--server-args", type=str, help="Arguments for the server.", multiple=True
)
@click.option(
    "--client-args", type=str, help="Arguments for the client.", multiple=True
)
@click.option("--gpus", type=int, help="The number of GPUs to use.")
@click.option("--dry-run", is_flag=True, help="A dry run will not run the command.")
@click.option(
    "--output-dir", default="results", help="Directory to store output files."
)
@click.option(
    "--output-json",
    type=str,
    default=None,
    help="Path to output a single JSON file with all results.",
)
@click.option(
    "--continue",
    "-c",
    "continue_flag",
    is_flag=True,
    help="Skip configs that already have output files.",
)
@click.option(
    "--rest", type=int, default=10, help="Rest time in seconds between benchmark runs."
)
@click.option("--mute-server", is_flag=True, help="Suppress server process stdout.")
@click.option(
    "--ready-endpoint",
    default="/health",
    help="Endpoint to check if server is ready (e.g., /health, /readyz).",
)
@click.option(
    "--host", type=str, default="127.0.0.1", help="Server host to connect to."
)
@click.option("--port", type=int, default=None, help="Server port to connect to.")
@click.option(
    "--dashboard-port", type=int, default=8080, help="Port to run the dashboard."
)
@click.pass_context
def cli(
    ctx,
    server_cmd,
    model,
    framework,
    server_args,
    client_args,
    gpus,
    dry_run,
    output_dir,
    output_json,
    continue_flag,
    rest,
    mute_server,
    ready_endpoint,
    host,
    port,
    dashboard_port,
):
    """A CLI tool to optimize LLM performance."""
    if ctx.invoked_subcommand is None:
        # If no subcommand is provided, run the main benchmark command
        benchmark(
            server_cmd,
            model,
            framework,
            server_args,
            client_args,
            gpus,
            dry_run,
            output_dir,
            output_json,
            continue_flag,
            rest,
            mute_server,
            ready_endpoint,
            host,
            port,
            dashboard_port,
        )


def benchmark(
    server_cmd,
    model,
    framework,
    server_args,
    client_args,
    gpus,
    dry_run,
    output_dir,
    output_json,
    continue_flag,
    rest,
    mute_server,
    ready_endpoint,
    host,
    port,
    dashboard_port,
):
    """A CLI tool to optimize LLM performance."""
    if not server_cmd:
        if not model or not framework:
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

    server_args = ";".join(server_args)
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

    output_jsonl_path = None
    if output_json:
        output_jsonl_path = pathlib.Path(output_json).with_suffix(".jsonl")

    completed_config_ids = set()
    if continue_flag and output_jsonl_path and output_jsonl_path.exists():
        logger.info(
            f"Found existing JSONL file, loading completed runs: {output_jsonl_path}"
        )
        with open(output_jsonl_path) as f:
            for line in f:
                try:
                    result = json.loads(line)
                    client_params = result.get("config", {}).get("client", {})
                    server_params = result.get("config", {}).get("server", {})
                    config_id = get_config_id(client_params, server_params)
                    completed_config_ids.add(config_id)
                except json.JSONDecodeError:
                    logger.warning(
                        f"Could not parse line in {output_jsonl_path}: {line.strip()}"
                    )
        logger.info(f"Loaded {len(completed_config_ids)} completed runs.")

    for idx, combo in enumerate(all_combinations):
        benchmark_settings = construct_benchmark_settings(combo)

        client_params = benchmark_settings["client"]
        server_params = benchmark_settings["server"]
        server_args = benchmark_settings["server_args"]

        config_id = get_config_id(client_params, server_params)
        output_file_path = output_dir / f"{config_id}.json"

        logger.info("-" * 80)
        logger.info(f"Starting run {idx + 1}/{total_configs}: {config_id}")

        if continue_flag:
            if output_jsonl_path:
                if config_id in completed_config_ids:
                    logger.info(
                        f"Skipping as config_id '{config_id}' found in {output_jsonl_path}"
                    )
                    continue
            elif output_file_path.exists():
                logger.info(
                    f"Skipping as output file already exists: {output_file_path}"
                )
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
                "dataset_name": "sharegpt",  # default value
                "num_prompts": 1000,  # default value
                "request_rate": float("inf"),  # default value
                "seed": 1,  # default value
            }
            benchmark_args.update(client_params)
            benchmark_result = bench_client.run_benchmark(benchmark_args)

            result_with_config = {
                "config": benchmark_settings,
                "results": benchmark_result,
                "cmd": full_server_cmd,
            }

            if output_jsonl_path:
                with open(output_jsonl_path, "a") as f:
                    f.write(json.dumps(result_with_config) + "\n")
                logger.info(f"Appended result to {output_jsonl_path}")
            else:
                with open(output_file_path, "w") as f:
                    json.dump(result_with_config, f, indent=2)
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

    if output_jsonl_path and output_jsonl_path.exists():
        all_results = []
        with open(output_jsonl_path) as f:
            for line in f:
                try:
                    all_results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # Already warned about this

        with open(output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"All benchmark results saved to {output_json}")

    logger.info("-" * 80)
    logger.info("All benchmark runs completed.")

    # Auto-visualize if requested and output JSON file exists
    if output_json and pathlib.Path(output_json).exists():
        try:
            logger.info("Opening visualization dashboard...")
            from llm_optimizer.visualization.visualize import ParetoLLMOptimizer

            # Create optimizer instance with default config
            config_path = (
                pathlib.Path(__file__).parent
                / "visualization"
                / "visualization_config.json"
            )
            optimizer = ParetoLLMOptimizer(str(config_path))

            # Generate dashboard and start server
            optimizer.generate_dashboard(output_json)
            optimizer.start_server(port=dashboard_port)

        except Exception as e:
            logger.error(f"Failed to open visualization dashboard: {e}")


@cli.command()
@click.option(
    "--data-file",
    type=str,
    required=True,
    help="Path to the JSON data file to visualize",
)
@click.option(
    "--config", type=str, default=None, help="Path to visualization config file"
)
@click.option("--port", type=int, default=8080, help="Port to run the dashboard server")
def visualize(data_file, config, port):
    """Generate and open visualization dashboard from benchmark results."""
    try:
        from llm_optimizer.visualization.visualize import ParetoLLMOptimizer

        # Determine config file path
        if config is None:
            config_path = (
                pathlib.Path(__file__).parent
                / "visualization"
                / "visualization_config.json"
            )
        else:
            config_path = pathlib.Path(config)

        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            return

        # Create optimizer instance
        optimizer = ParetoLLMOptimizer(str(config_path))

        # Check if data file exists
        if not pathlib.Path(data_file).exists():
            logger.error(f"Data file not found: {data_file}")
            return

        # Generate dashboard
        logger.info(f"Generating dashboard from {data_file}...")
        html_file = optimizer.generate_dashboard(data_file)
        logger.info(f"Dashboard generated: {html_file}")

        # Start server and open browser
        logger.info(f"Starting server on port {port}...")
        optimizer.start_server(port=port)

    except Exception as e:
        logger.error(f"Failed to generate visualization: {e}")
        import traceback

        traceback.print_exc()


@cli.command()
@click.option(
    "--model",
    type=str,
    required=True,
    help="HuggingFace model ID (e.g., 'meta-llama/Meta-Llama-3-8B')",
)
@click.option(
    "--input-len", type=int, required=True, help="Input sequence length in tokens"
)
@click.option(
    "--output-len", type=int, required=True, help="Output sequence length to generate"
)
@click.option(
    "--gpu",
    type=click.Choice(list_available_gpus()),
    help="GPU model (auto-detected if not specified)",
)
@click.option(
    "--num-gpus", type=int, help="Number of GPUs (auto-detected if not specified)"
)
@click.option(
    "--precision",
    type=click.Choice(["fp16", "fp8"]),
    default="fp16",
    help="Model precision",
)
@click.option(
    "--framework",
    type=click.Choice(["sglang", "vllm", "both"]),
    default="both",
    help="Framework to optimize for",
)
@click.option(
    "--constraints", type=str, help="SLO constraints (e.g., 'ttft<300ms;itl<8.5ms')"
)
@click.option(
    "--target",
    type=click.Choice(["throughput", "latency"]),
    default="throughput",
    help="Optimization target",
)
@click.option("--interactive", is_flag=True, help="Run in interactive mode")
@click.option(
    "--generate-commands", is_flag=True, help="Generate llm-optimizer tuning commands"
)
def estimate_performance(
    model,
    input_len,
    output_len,
    gpu,
    num_gpus,
    precision,
    framework,
    constraints,
    target,
    interactive,
    generate_commands,
):
    """Estimate LLM performance and suggest optimal configurations."""

    if interactive:
        click.echo("=== LLM Performance Estimation (Interactive Mode) ===")

        # Get model if not provided
        if not model:
            model = click.prompt("HuggingFace model ID")

        # Get input/output lengths if not provided
        if not input_len:
            input_len = click.prompt("Input sequence length", type=int, default=1024)
        if not output_len:
            output_len = click.prompt("Output sequence length", type=int, default=1024)

        # Get optimization target
        if not target:
            target = click.prompt(
                "Optimization target",
                type=click.Choice(["throughput", "latency"]),
                default="throughput",
            )

        # Get constraints
        if not constraints:
            constraints = click.prompt(
                "SLO constraints (optional, e.g., 'ttft<300ms;itl<8.5ms')",
                default="",
                show_default=False,
            )
            constraints = constraints if constraints.strip() else None

        # Get framework
        if framework == "both":
            framework = click.prompt(
                "Framework",
                type=click.Choice(["sglang", "vllm", "both"]),
                default="both",
            )

    try:
        # Auto-detect GPU if not specified
        if not gpu:
            gpu = detect_gpu_type()
            if not gpu:
                available_gpus = ", ".join(list_available_gpus())
                click.echo(
                    f"Could not auto-detect GPU. Available GPUs: {available_gpus}"
                )
                gpu = click.prompt(
                    "GPU model", type=click.Choice(list_available_gpus())
                )
            else:
                click.echo(f"Auto-detected GPU: {gpu}")

        # Auto-detect GPU count if not specified
        if not num_gpus:
            num_gpus = get_gpu_count()
            if num_gpus == 0:
                num_gpus = click.prompt("Number of GPUs", type=int, default=1)
            else:
                click.echo(f"Auto-detected {num_gpus} GPU(s)")

        click.echo("\n=== Configuration ===")
        click.echo(f"Model: {model}")
        click.echo(f"GPU: {num_gpus}x {gpu}")
        click.echo(f"Precision: {precision}")
        click.echo(f"Input/Output: {input_len}/{output_len} tokens")
        click.echo(f"Target: {target}")
        if constraints:
            click.echo(f"Constraints: {constraints}")

        # Load model configuration
        click.echo("\nFetching model configuration...")
        model_config = get_model_config_from_hf(model)
        click.echo(
            f"Model: {model_config.num_params:.1f}B parameters, {model_config.num_layers} layers"
        )

        # Parse constraints if provided
        parsed_constraints = []
        if constraints:
            try:
                parsed_constraints = parse_slo_constraints(constraints)
                click.echo(f"Parsed {len(parsed_constraints)} constraint(s)")
            except ValueError as e:
                click.echo(f"Error parsing constraints: {e}")
                return

        # Find best performance configurations
        click.echo("\n=== Performance Analysis ===")
        best_configs = find_best_performance(
            num_gpus=num_gpus,
            gpu_name=gpu,
            model_config=model_config,
            precision=precision,
            input_length=input_len,
            output_length=output_len,
        )

        if best_configs["best_latency"]:
            latency_config = best_configs["best_latency"]
            click.echo(f"Best Latency (concurrency={latency_config.concurrency}):")
            click.echo(f"  TTFT: {latency_config.ttft_ms:.1f} ms")
            click.echo(f"  ITL: {latency_config.itl_ms:.1f} ms")
            click.echo(f"  E2E: {latency_config.e2e_latency_s:.2f} s")

        if best_configs["best_output_throughput"]:
            throughput_config = best_configs["best_output_throughput"]
            click.echo(
                f"\nBest Throughput (concurrency={throughput_config.concurrency}):"
            )
            click.echo(
                f"  Output: {throughput_config.output_throughput_tps:.1f} tokens/s"
            )
            click.echo(
                f"  Input: {throughput_config.input_throughput_tps:.1f} tokens/s"
            )
            click.echo(f"  Requests: {throughput_config.requests_per_sec:.2f} req/s")
            click.echo(
                f"  Bottleneck: {'Memory' if throughput_config.bottleneck_is_memory else 'Compute'}"
            )

        # Calculate theoretical concurrency limits
        concurrency_limits = calculate_concurrency_limits(
            num_gpus=num_gpus,
            gpu_name=gpu,
            model_config=model_config,
            precision=precision,
            input_length=input_len,
            output_length=output_len,
        )

        click.echo("\n=== Roofline Analysis ===")
        if best_configs["best_output_throughput"]:
            result = best_configs["best_output_throughput"]
            click.echo(f"Hardware Ops/Byte Ratio: {result.hardware_ops_per_byte:.1f} ops/byte")
            click.echo(f"Prefill Arithmetic Intensity: {result.prefill_arithmetic_intensity:.1f} ops/byte")
            click.echo(f"Decode Arithmetic Intensity: {result.decode_arithmetic_intensity:.1f} ops/byte")
            click.echo(f"Prefill Phase: {'Memory Bound' if result.prefill_is_memory_bound else 'Compute Bound'}")
            click.echo(f"Decode Phase: {'Memory Bound' if result.decode_is_memory_bound else 'Compute Bound'}")

        click.echo("\n=== Concurrency Analysis ===")
        click.echo(f"KV Cache Memory Limit: {concurrency_limits['kv_cache_limit']} concurrent requests")
        click.echo(f"Prefill Compute Limit: {concurrency_limits['prefill_compute_limit']} concurrent requests")
        click.echo(f"Decode Capacity Limit: {concurrency_limits['decode_capacity_limit']} concurrent requests")
        click.echo(f"Theoretical Overall Limit: {concurrency_limits['overall_limit']} concurrent requests")

        # Find optimal concurrency
        optimal_concurrency = find_optimal_concurrency_threshold(
            num_gpus=num_gpus,
            gpu_name=gpu,
            model_config=model_config,
            precision=precision,
            input_length=input_len,
            output_length=output_len,
        )
        click.echo(f"Empirical Optimal Concurrency: {optimal_concurrency} concurrent requests")

        # Check constraints if provided
        constrained_result = None
        if parsed_constraints:
            constrained_result = estimate_performance_under_constraints(
                num_gpus=num_gpus,
                gpu_name=gpu,
                model_config=model_config,
                precision=precision,
                input_length=input_len,
                output_length=output_len,
                constraints=parsed_constraints,
            )

            if constrained_result:
                click.echo("\n=== Performance under Constraints ===")
                click.echo(f"Concurrency: {constrained_result.concurrency}")
                click.echo(f"TTFT: {constrained_result.ttft_ms:.1f} ms")
                click.echo(f"ITL: {constrained_result.itl_ms:.1f} ms")
                click.echo(
                    f"Output throughput: {constrained_result.output_throughput_tps:.1f} tokens/s"
                )
            else:
                click.echo(
                    "\n❌ Cannot satisfy the given constraints with this configuration"
                )
                return

        # Generate tuning configurations if requested
        if generate_commands:
            click.echo("\n=== Tuning Commands ===")

            # Use constrained result if available, otherwise best throughput
            reference_concurrency = optimal_concurrency
            if constrained_result:
                reference_concurrency = constrained_result.concurrency
            elif target == "latency" and best_configs["best_latency"]:
                reference_concurrency = best_configs["best_latency"].concurrency
            elif best_configs["best_output_throughput"]:
                reference_concurrency = best_configs[
                    "best_output_throughput"
                ].concurrency

            target_throughput = target == "throughput"
            frameworks_to_test = (
                ["sglang", "vllm"] if framework == "both" else [framework]
            )

            for fw in frameworks_to_test:
                click.echo(f"\n--- {fw.upper()} Configurations ---")

                # Use simplified configurations for constrained scenarios
                if parsed_constraints:
                    from llm_optimizer.tuning import (
                        generate_simplified_throughput_configs,
                    )
                    tuning_configs = generate_simplified_throughput_configs(
                        framework=fw,
                        num_gpus=num_gpus,
                        gpu_name=gpu,
                        model_config=model_config,
                        optimal_concurrency=reference_concurrency,
                        precision=precision,
                        sequence_length=input_len,
                        constraints=parsed_constraints,
                    )
                else:
                    tuning_configs = get_framework_tuning_configs(
                        framework=fw,
                        num_gpus=num_gpus,
                        gpu_name=gpu,
                        model_config=model_config,
                        optimal_concurrency=reference_concurrency,
                        target_throughput=target_throughput,
                        precision=precision,
                        sequence_length=input_len,
                    )

                commands = generate_llm_optimizer_commands(
                    configs=tuning_configs,
                    model_id=model,
                    input_length=input_len,
                    output_length=output_len,
                    num_gpus=num_gpus,
                    constraints=constraints,
                )

                for i, (config, cmd) in enumerate(zip(tuning_configs, commands), 1):
                    click.echo(f"\nConfig {i}: {config.description}")
                    click.echo(f"Command: {cmd}")

    except Exception as e:
        click.echo(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    cli()

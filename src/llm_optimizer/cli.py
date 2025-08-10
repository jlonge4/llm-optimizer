import json
import pathlib
import time
import typing as t

import click
import pynvml

# Readline support for better interactive experience
try:
    import readline
    HAS_READLINE = True
    
    # Configure readline for better UX
    readline.set_startup_hook(None)
    readline.parse_and_bind('tab: complete')
    readline.parse_and_bind('set editing-mode emacs')  # Enable emacs-style editing
    readline.parse_and_bind('set completion-ignore-case on')
    
    # History file for model completions (optional)
    import os
    history_file = os.path.expanduser('~/.llm_optimizer_history')
    try:
        readline.read_history_file(history_file)
    except FileNotFoundError:
        pass
    
    def save_history():
        try:
            readline.set_history_length(1000)
            readline.write_history_file(history_file)
        except:
            pass
    
    import atexit
    atexit.register(save_history)
    
except ImportError:
    HAS_READLINE = False

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
from llm_optimizer.predefined.gpus import list_available_gpus, list_available_gpus_with_lowercase
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


def friendly_prompt(message: str, default=None, choices=None, type_converter=None, completions=None, gpu_type_field=False):
    """
    User-friendly prompt with readline support for better interactive experience.
    
    Args:
        message: Prompt message to display
        default: Default value (shown in brackets)
        choices: List of valid choices (for validation)
        type_converter: Function to convert input (e.g., int)
        completions: List of completion options
    
    Returns:
        User input with appropriate type conversion
    """
    if not HAS_READLINE:
        # Fallback to click.prompt if readline not available
        if choices:
            return click.prompt(message, type=click.Choice(choices), default=default)
        elif type_converter == int:
            return click.prompt(message, type=int, default=default)
        else:
            return click.prompt(message, default=default if default else "")
    
    # Setup completions if provided
    if completions:
        def completer(text, state):
            matches = [item for item in completions if item.lower().startswith(text.lower())]
            try:
                return matches[state]
            except IndexError:
                return None
        readline.set_completer(completer)
        readline.set_completer_delims(' \t\n')
    else:
        readline.set_completer(None)
    
    # Format prompt with default
    if default is not None:
        prompt_text = f"{message} [{default}]: "
    else:
        prompt_text = f"{message}: "
    
    while True:
        try:
            user_input = input(prompt_text).strip()
            
            # Use default if empty input
            if not user_input and default is not None:
                user_input = str(default)
            
            # GPU type special handling (case-insensitive)
            if gpu_type_field:
                normalized_input = normalize_gpu_choice(user_input)
                if normalized_input != user_input.upper() and user_input.upper() not in list_available_gpus():
                    available_gpus = ", ".join(list_available_gpus())
                    available_lower = ", ".join([name.lower() for name in list_available_gpus()])
                    print(f"❌ Invalid GPU. Available: {available_gpus} (case-insensitive: {available_lower})")
                    continue
                user_input = normalized_input
            
            # Validate choices (skip for GPU types as they're handled above)
            elif choices and user_input not in choices:
                print(f"❌ Invalid choice. Please select from: {', '.join(choices)}")
                continue
            
            # Type conversion
            if type_converter:
                try:
                    return type_converter(user_input)
                except ValueError:
                    print(f"❌ Invalid format. Please enter a valid {type_converter.__name__}.")
                    continue
            
            return user_input
            
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")
            raise click.Abort()


def normalize_gpu_choice(user_input: str) -> str:
    """Normalize GPU input to uppercase for internal use."""
    if user_input.lower() in [name.lower() for name in list_available_gpus()]:
        return user_input.upper()
    return user_input  # Return as-is if not found (for error handling)


def friendly_confirm(message: str, default=True):
    """User-friendly yes/no confirmation with readline support."""
    if not HAS_READLINE:
        return click.confirm(message, default=default)
    
    default_text = "Y/n" if default else "y/N"
    prompt_text = f"{message} [{default_text}]: "
    
    while True:
        try:
            user_input = input(prompt_text).strip().lower()
            
            if not user_input:
                return default
            elif user_input in ['y', 'yes', 'true', '1']:
                return True
            elif user_input in ['n', 'no', 'false', '0']:
                return False
            else:
                print("❌ Please enter 'y' for yes or 'n' for no.")
                
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")
            raise click.Abort()


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
    required=False,  # Made optional for interactive mode
    help="HuggingFace model ID (e.g., 'meta-llama/Meta-Llama-3-8B')",
)
@click.option(
    "--input-len", type=int, required=False, help="Input sequence length in tokens"
)
@click.option(
    "--output-len", type=int, required=False, help="Output sequence length to generate"
)
@click.option(
    "--gpu",
    type=click.Choice(list_available_gpus_with_lowercase(), case_sensitive=False),
    help="GPU model (auto-detected if not specified, case-insensitive)",
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
    
    # Normalize GPU input if provided via CLI
    if gpu:
        gpu = normalize_gpu_choice(gpu)
    
    # Validate that required parameters are provided either via CLI or interactive mode
    if not interactive and (not model or input_len is None or output_len is None):
        click.echo("Error: --model, --input-len, and --output-len are required when not using --interactive mode")
        click.echo("Use --interactive for guided input or provide all required parameters")
        return

    if interactive:
        click.echo("=== LLM Performance Estimation (Interactive Mode) ===")
        click.echo()
        
        # Get model if not provided
        if not model:
            click.echo("🤖 Model Selection")
            click.echo("Popular options: meta-llama/Llama-3.2-1B, meta-llama/Meta-Llama-3-8B, meta-llama/Meta-Llama-3-70B")
            model_completions = [
                "meta-llama/Llama-3.2-1B",
                "meta-llama/Meta-Llama-3-8B", 
                "meta-llama/Meta-Llama-3-70B",
                "meta-llama/Llama-2-7b-chat-hf",
                "meta-llama/Llama-2-13b-chat-hf",
                "mistralai/Mistral-7B-v0.1",
                "microsoft/DialoGPT-medium"
            ]
            model = friendly_prompt("HuggingFace model ID", completions=model_completions)

        # Get input/output lengths if not provided
        if input_len is None:
            click.echo("\n📏 Sequence Length Configuration")
            click.echo("Typical values: 512 (short), 1024 (medium), 2048 (long), 4096 (very long)")
            input_len = friendly_prompt("Input sequence length (tokens)", default=1024, type_converter=int)
        if output_len is None:
            output_len = friendly_prompt("Output sequence length (tokens)", default=512, type_converter=int)

        # Get optimization target
        if not target:
            click.echo("\n🎯 Optimization Target")
            click.echo("• throughput: Maximize tokens/second (good for batch processing)")
            click.echo("• latency: Minimize response time (good for interactive use)")
            target = friendly_prompt(
                "Optimization target",
                default="throughput",
                choices=["throughput", "latency"]
            )

        # Get constraints
        if not constraints:
            click.echo("\n⚡ Performance Constraints (Optional)")
            click.echo("Examples:")
            click.echo("• 'ttft:median<300ms' - First token in under 300ms (median)")
            click.echo("• 'itl:p95<50ms' - Inter-token latency under 50ms (95th percentile)")
            click.echo("• 'ttft<200ms;itl:p99<10ms' - Multiple constraints")
            click.echo("Statistical types: mean, median, p95, p99")
            constraint_examples = [
                "ttft:median<300ms",
                "itl:p95<50ms",
                "ttft<200ms;itl:p99<10ms",
                "e2e_latency:p95<2s"
            ]
            constraints = friendly_prompt(
                "SLO constraints (press Enter to skip)",
                default="",
                completions=constraint_examples
            )
            constraints = constraints if constraints.strip() else None

        # Get precision
        if precision is None:
            click.echo("\n🔢 Model Precision")
            click.echo("• fp16: Standard precision (good balance)")
            click.echo("• fp8: Higher throughput but requires newer GPUs (H100+)")
            precision = friendly_prompt(
                "Model precision",
                default="fp16",
                choices=["fp16", "fp8"]
            )

        # Get framework
        if framework == "both" or framework is None:
            click.echo("\n🚀 Framework Selection")
            click.echo("• sglang: Fast inference engine optimized for throughput")
            click.echo("• vllm: Popular serving framework with good compatibility")
            click.echo("• both: Generate configs for both frameworks")
            framework = friendly_prompt(
                "Framework",
                default="both",
                choices=["sglang", "vllm", "both"]
            )
            
        # Ask about command generation
        if not generate_commands:
            click.echo("\n📋 Command Generation")
            generate_commands = friendly_confirm(
                "Generate llm-optimizer tuning commands?",
                default=True
            )

    try:
        # Auto-detect and prompt for GPU configuration
        if interactive or not gpu or not num_gpus:
            click.echo(f"\n💻 GPU Configuration")
            
            # Handle GPU type
            if not gpu:
                detected_gpu = detect_gpu_type()
                if detected_gpu:
                    click.echo(f"Auto-detected GPU: {detected_gpu}")
                    gpu = friendly_prompt(
                        "GPU model (press Enter to use auto-detected)",
                        default=detected_gpu,
                        completions=list_available_gpus_with_lowercase(),
                        gpu_type_field=True
                    )
                else:
                    available_gpus = ", ".join(list_available_gpus())
                    click.echo(f"Could not auto-detect GPU. Available GPUs: {available_gpus}")
                    gpu = friendly_prompt(
                        "GPU model",
                        completions=list_available_gpus_with_lowercase(),
                        gpu_type_field=True
                    )
            elif interactive:
                # GPU was provided via command line but we're in interactive mode - ask for confirmation
                click.echo(f"Command-line GPU: {gpu}")
                gpu = friendly_prompt(
                    "GPU model (press Enter to keep current)",
                    default=gpu,
                    completions=list_available_gpus_with_lowercase(),
                    gpu_type_field=True
                )

            # Handle GPU count
            if not num_gpus:
                detected_gpus = get_gpu_count()
                if detected_gpus > 0:
                    click.echo(f"Auto-detected: {detected_gpus} GPU(s)")
                    num_gpus = friendly_prompt(
                        "Number of GPUs (press Enter to use auto-detected)",
                        default=detected_gpus,
                        type_converter=int
                    )
                else:
                    num_gpus = friendly_prompt("Number of GPUs", default=1, type_converter=int)
            elif interactive:
                # GPU count was provided via command line but we're in interactive mode - ask for confirmation
                click.echo(f"Command-line GPU count: {num_gpus}")
                num_gpus = friendly_prompt(
                    "Number of GPUs (press Enter to keep current)",
                    default=num_gpus,
                    type_converter=int
                )
        else:
            # Non-interactive mode: auto-detect only if not specified
            if not gpu:
                gpu = detect_gpu_type()
                if not gpu:
                    available_gpus = ", ".join(list_available_gpus())
                    click.echo(f"Could not auto-detect GPU. Available GPUs: {available_gpus}")
                    gpu = friendly_prompt(
                        "GPU model",
                        completions=list_available_gpus_with_lowercase(),
                        gpu_type_field=True
                    )
                else:
                    click.echo(f"Auto-detected GPU: {gpu}")

            if not num_gpus:
                num_gpus = get_gpu_count()
                if num_gpus == 0:
                    num_gpus = friendly_prompt("Number of GPUs", default=1, type_converter=int)
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

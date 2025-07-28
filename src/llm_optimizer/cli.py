import click
import pynvml
import typing as t
import llm_optimizer.args as lo_args
import llm_optimizer.predefined as predefined

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


def construct_benchmark_settings(combo: t.List[lo_args.BaseArg]) -> t.Dict[str, t.Any]:
    client_args = [arg for arg in combo if arg.scope == lo_args.ArgScope.CLIENT]
    server_args = [arg for arg in combo if arg.scope == lo_args.ArgScope.SERVER]
    client_kv_pairs = lo_args.get_all_kv_pairs(client_args)
    server_cmd_args = lo_args.get_all_cmd_args(server_args)
    return {
        "client": client_kv_pairs,
        "server": server_cmd_args,
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
def main(server_cmd, model, framework, server_args, client_args, gpus, dry_run):
    """A CLI tool to optimize LLM performance."""
    if not server_cmd:
        if (not model or not framework):
            raise click.UsageError(
                "If --server-cmd is not provided, both --model and --framework are required."
            )

        tmpl = predefined.SEVER_CMD_TMPL[framework]
        server_cmd = tmpl.format(
            model=model,
            host="127.0.0.1",
            port=40000,
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
        server_configs = predefined.SERVER_CONFIGS["framework"]

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

    for idx, combo in enumerate(all_combinations):
        print(f"run {idx}")
        benchmark_settings = construct_benchmark_settings(combo)
        if dry_run:
            print(benchmark_settings)
        else:
            # do acutal benchmark
            pass


if __name__ == "__main__":
    main()

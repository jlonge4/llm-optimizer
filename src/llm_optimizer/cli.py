import click
import pynvml

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

@click.command()
@click.option("--server-cmd", type=str, help="The command to start the server.")
@click.option("--model", type=str, help="The model to use.")
@click.option("--framework", type=str, help="The framework to use.")
@click.option("--server-args", type=str, help="Arguments for the server.", multiple=True)
@click.option("--client-args", type=str, help="Arguments for the client.", multiple=True)
@click.option("--gpus", type=int, help="The number of GPUs to use.")
@click.option("--dry-run", is_flag=True, help="A dry run will not run the command.")
def main(server_cmd, model, framework, server_args, client_args, gpus, dry_run):
    """A CLI tool to optimize LLM performance."""
    if not server_cmd and (not model or not framework):
        raise click.UsageError(
            "If --server-cmd is not provided, both --model and --framework are required."
        )

    if gpus is None:
        gpus = get_gpu_count()

    if dry_run:
        click.echo("Dry run mode enabled.")

    click.echo("CLI tool for llm-optimizer")
    if server_cmd:
        click.echo(f"Server command: {server_cmd}")
    if model:
        click.echo(f"Model: {model}")
    if framework:
        click.echo(f"Framework: {framework}")
    if server_args:
        click.echo(f"Server arguments: {server_args}")
    if client_args:
        click.echo(f"Client arguments: {client_args}")
    click.echo(f"Number of GPUs: {gpus}")

if __name__ == "__main__":
    main()

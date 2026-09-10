import os
import shlex
import subprocess
import time

import psutil
import requests

from llm_optimizer.exceptions import ServerNotReadyError
from llm_optimizer.logging import get_logger

logger = get_logger("server_utils")


DEFAULT_READY_TIMEOUT = 300

# AWS Neuron compiles the model to static-shape NEFFs on first start, which
# routinely takes tens of minutes on a cold cache. Waiting five minutes would
# kill every run before the server ever came up.
NEURON_READY_TIMEOUT = 5400


def wait_for_server(
    url: str,
    timeout: int = DEFAULT_READY_TIMEOUT,
    delay: int = 5,
    request_timeout: int = 5,
) -> bool:
    """Repeatedly try to access the server until it's available.

    Args:
        url: Readiness endpoint to poll
        timeout: Seconds to keep trying before giving up
        delay: Seconds between attempts
        request_timeout: Per-request timeout

    Returns:
        True once the endpoint returns 200, False if the deadline passes
    """
    logger.info(f"Waiting up to {timeout}s for server to be ready at {url}...")
    deadline = time.monotonic() + timeout
    attempt = 0

    while True:
        attempt += 1
        try:
            response = requests.get(url, timeout=request_timeout)
            if response.status_code == 200:
                elapsed = timeout - max(0, deadline - time.monotonic())
                logger.info(
                    f"Server is up after {attempt} attempts ({elapsed:.0f}s)."
                )
                return True
        except requests.RequestException:
            pass

        if time.monotonic() + delay >= deadline:
            return False

        logger.debug(
            f"Server not ready yet (attempt {attempt}, "
            f"{deadline - time.monotonic():.0f}s left). Retrying in {delay}s."
        )
        time.sleep(delay)


def start_server(
    server_cmd: str,
    server_envs: dict[str, str],
    ready_url: str,
    mute: bool,
    ready_timeout: int = DEFAULT_READY_TIMEOUT,
) -> psutil.Process:
    """Starts the server and waits for it to become ready."""
    cmd = shlex.split(server_cmd)
    logger.info(f"Starting server with command: {server_cmd}")
    server_process = None
    envs = os.environ.copy()
    envs.update(server_envs)

    try:
        stdout_pipe = subprocess.DEVNULL if mute else None
        server_process = psutil.Popen(
            cmd, text=True, stdout=stdout_pipe, stderr=subprocess.STDOUT, env=envs
        )
        logger.debug(f"Started server with PID: {server_process.pid}")

        if not wait_for_server(ready_url, timeout=ready_timeout):
            raise ServerNotReadyError(
                f"Server did not become ready within {ready_timeout}s. "
                f"On AWS Neuron a cold compile can take far longer -- raise "
                f"--server-timeout, or pre-populate "
                f"NEURON_COMPILED_ARTIFACTS."
            )

        return server_process
    except Exception:
        if server_process:
            terminate_process_top_down(server_process)
        raise


def terminate_process_top_down(process: psutil.Process, timeout: int = 10):
    """Gracefully terminates a process and its children, escalating to KILL if necessary."""
    logger.info(f"Terminating server process {process.pid} and its children.")
    try:
        children = process.children(recursive=True)
        # Terminate children first
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass

        # Terminate parent
        process.terminate()

        # Wait for all to die
        _, alive = psutil.wait_procs([process] + children, timeout=timeout)

        # Force kill any remaining processes
        if alive:
            logger.warning(
                "Some processes did not terminate gracefully. Escalating to SIGKILL."
            )
            for p in alive:
                try:
                    p.kill()
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(alive, timeout=timeout)
    except psutil.NoSuchProcess:
        logger.debug(f"Process {process.pid} was already gone.")
    except Exception as e:
        logger.error(f"Error during process termination: {e}")

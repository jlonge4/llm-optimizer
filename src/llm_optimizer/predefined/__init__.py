import llm_optimizer.predefined.max as max
import llm_optimizer.predefined.sglang as sglang
import llm_optimizer.predefined.vllm as vllm
import llm_optimizer.predefined.vllm_neuron as vllm_neuron
from llm_optimizer.predefined.client import CLIENT_CONFIGS

SERVER_CONFIGS = {
    "sglang": sglang.SERVER_CONFIGS,
    "vllm": vllm.SERVER_CONFIGS,
    "vllm-neuron": vllm_neuron.SERVER_CONFIGS,
    "max": max.SERVER_CONFIGS,
}

PARAMETER_MAPPINGS = {
    "sglang": sglang.PARAMETER_MAPPING,
    "vllm": vllm.PARAMETER_MAPPING,
    "vllm-neuron": vllm_neuron.PARAMETER_MAPPING,
    "max": max.PARAMETER_MAPPING,
}


SEVER_CMD_TMPL = {
    "sglang": "python3 -m sglang.launch_server --model-path {model} --host {host} --port {port}",
    "vllm": "vllm serve {model} --host {host} --port {port}",
    # The Neuron plugin registers itself with vLLM, so the launch command is
    # identical -- it is the tuned arguments that differ.
    "vllm-neuron": "vllm serve {model} --host {host} --port {port}",
    "max": "max serve --model-path {model} --port {port}",
}


# Frameworks that can serve AWS Trainium/Inferentia devices.
NEURON_FRAMEWORKS = frozenset({"vllm-neuron"})


def framework_for_device(framework: str, gpu_name: str) -> str:
    """
    Resolve the framework actually used to serve a device.

    On AWS Neuron devices, vLLM serves through the vLLM Neuron plugin, so
    "vllm" resolves to "vllm-neuron". Everything else is returned unchanged.

    Args:
        framework: Framework requested by the user
        gpu_name: GPU or instance SKU being targeted

    Returns:
        The framework to generate configurations for

    Raises:
        ValueError: If the framework cannot serve the device
    """
    from llm_optimizer.predefined.gpus import is_neuron_device

    if not gpu_name or not is_neuron_device(gpu_name):
        return framework

    if framework == "vllm":
        return "vllm-neuron"
    if framework in NEURON_FRAMEWORKS:
        return framework

    raise ValueError(
        f"Framework '{framework}' does not support AWS Neuron devices "
        f"({gpu_name}). Use --framework vllm to serve through the vLLM Neuron "
        f"plugin: https://github.com/vllm-project/vllm-neuron"
    )

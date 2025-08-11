import llm_optimizer.predefined.sglang as sglang
import llm_optimizer.predefined.vllm as vllm
from llm_optimizer.predefined.client import CLIENT_CONFIGS

SERVER_CONFIGS = {
    "sglang": sglang.SERVER_CONFIGS,
    "vllm": vllm.SERVER_CONFIGS,
}

PARAMETER_MAPPINGS = {
    "sglang": sglang.PARAMETER_MAPPING,
    "vllm": vllm.PARAMETER_MAPPING,
}


SEVER_CMD_TMPL = {
    "sglang": "python3 -m sglang.launch_server --model-path {model} --host {host} --port {port}",
    "vllm": "vllm serve {model} --host {host} --port {port}",
}

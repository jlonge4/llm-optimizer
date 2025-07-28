from llm_optimizer.predefined.client import CLIENT_CONFIGS
from llm_optimizer.predefined.sglang import SGLANG_SERVER_CONFIGS
from llm_optimizer.predefined.vllm import VLLM_SERVER_CONFIGS

SERVER_CONFIGS = {
    "sglang": SGLANG_SERVER_CONFIGS,
    "vllm": VLLM_SERVER_CONFIGS,
}


SEVER_CMD_TMPL = {
    "sglang": "python3 -m sglang.launch_server --model-path {model} --host {host} --port {port}",
    "vllm": "vllm serve {model} --host {host} --port {port}",
}

# LLM Optimizer

A Python tool for optimizing Large Language Model (LLM) serving performance. Supports SGLang and vLLM frameworks with automated benchmarking and performance estimation.

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e .[dev]
```

## Quick Start

### Raw Benchmarking with Parameter Combinations

Run comprehensive benchmarks across multiple parameter configurations:

```bash
# SGLang with multiple TP/DP combinations
llm-optimizer \
  --framework sglang \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --server-args "tp_size*dp_size=[(1,4),(2,2),(4,1)];chunked_prefill_size=[2048,4096,8192]" \
  --client-args "max_concurrency=[50,100,200];num_prompts=1000" \
  --output-json sglang_results.json

# vLLM with batch size tuning
llm-optimizer \
  --framework vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --server-args "tensor_parallel_size=[1,2,4];max_num_batched_tokens=[4096,8192,16384]" \
  --client-args "max_concurrency=[32,64,128];num_prompts=1000;dataset_name=sharegpt" \
  --output-json vllm_results.json

# Complex parameter grid for throughput optimization
llm-optimizer \
  --framework sglang \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --server-args "tp_size*dp_size=[(1,8),(2,4),(4,2)];schedule_conservativeness=[0.3,0.6,1.0];chunked_prefill_size=range(2048,8193,2048)" \
  --client-args "max_concurrency=range(50,201,50);request_rate=[10,20,50]" \
  --gpus 8 \
  --output-json complex_benchmark.json

# Latency-optimized configuration
llm-optimizer \
  --framework vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --server-args "tensor_parallel_size=[2,4];max_num_seqs=[16,32,64]" \
  --client-args "max_concurrency=[8,16,32];num_prompts=500" \
  --constraints "ttft<200ms;itl:p99<10ms" \
  --output-json latency_optimized.json
```

### Performance Estimation

Get optimal serving configurations without running actual benchmarks:

```bash
# Basic performance estimation
llm-optimizer estimate \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --input-len 1024 \
  --output-len 512

# With GPU specification
llm-optimizer estimate \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --input-len 2048 \
  --output-len 1024 \
  --gpu H100 \
  --num-gpus 8

# With performance constraints and command generation
llm-optimizer estimate \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --input-len 1024 \
  --output-len 512 \
  --gpu H100 \
  --num-gpus 4 \
  --constraints "ttft:mean<300ms;itl:p95<50ms" \
  --generate-commands

# Interactive mode

llm-optimizer estimate --interactive
```

### Using Custom Server Commands

```bash
# Custom SGLang server
llm-optimizer \
  --server-cmd "python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --port 30000" \
  --client-args "max_concurrency=[25,50,100];num_prompts=1000" \
  --host 0.0.0.0 \
  --port 30000

# Custom vLLM server with specific GPU allocation
llm-optimizer \
  --server-cmd "vllm serve meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 4" \
  --client-args "max_concurrency=[64,128];num_prompts=2000" \
  --port 8000
```

## Visualization

Generate interactive dashboard from benchmark results:

```bash
# Visualize results with Pareto frontier analysis
llm-optimizer visualize --data-file results.json --port 8080

# Combine multiple result files
llm-optimizer visualize --data-file "sglang_results.json,vllm_results.json" --port 8080
```

## Configuration Examples

### SGLang Parameter Tuning
- `tp_size*dp_size`: Tensor/Data parallelism combinations
- `chunked_prefill_size`: Prefill chunk size for throughput
- `schedule_conservativeness`: Request scheduling aggressiveness
- `schedule_policy`: Scheduling policy (fcfs, priority)

### vLLM Parameter Tuning  
- `tensor_parallel_size`: Tensor parallelism degree
- `max_num_batched_tokens`: Maximum batch size in tokens
- `max_num_seqs`: Maximum concurrent sequences

### Client Parameters
- `max_concurrency`: Maximum concurrent requests
- `num_prompts`: Total number of requests to send
- `dataset_name`: Dataset for request generation (sharegpt, alpaca)
- `random_input/random_output`: Random sequence lengths

## Performance Constraints

Specify SLO constraints for optimization:

```bash
# Time to first token constraints
--constraints "ttft<300ms"                    # Mean TTFT under 300ms
--constraints "ttft:median<200ms"             # Median TTFT under 200ms  
--constraints "ttft:p95<500ms"                # 95th percentile under 500ms

# Inter-token latency constraints
--constraints "itl:mean<20ms"                 # Mean ITL under 20ms
--constraints "itl:p99<50ms"                  # 99th percentile under 50ms

# Combined constraints
--constraints "ttft:median<300ms;itl:p95<10ms;e2e_latency:p95<2s"
```

## Supported GPUs

H100, H200, A100, L20, L40, B100, B200 with accurate TFLOPS specifications.

## Output Formats

- JSON: Structured benchmark results
- JSONL: Line-delimited for streaming
- Interactive Dashboard: Pareto optimization analysis with SciPy curve fitting

## Development

```bash
# Code formatting and linting
ruff format
ruff check

# Type checking
mypy src/
```

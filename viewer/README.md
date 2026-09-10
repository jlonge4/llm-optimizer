# vLLM Benchmark Viewer

Loads benchmark results produced by
[llm-optimizer](https://github.com/bentoml/llm-optimizer) and lets you pick a
configuration out of a sweep against a budget you set.

![Three Trainium2 runs plotted as output throughput against p99 latency, with the budget shaded green and the selected run detailed in cards below](docs/tradeoff-view.png)

Above: three Trainium2 configurations against a `p99 e2e latency < 4300 ms`
budget. The run inside it is green, the two outside are grey, and the starred
chip marks the winner. The cards describe whichever point is selected — here
`TP1/DP16/seqs64`, the highest-throughput run, which misses the budget.

```bash
pnpm install
pnpm dev
```

## Accepted input

Paste into the textarea, or drag a file onto the drop zone. All four shapes
below are handled:

| Input | Where it comes from |
|---|---|
| `results.jsonl` | Appended by llm-optimizer as each run finishes — one JSON record per line |
| `results.json` | The aggregated file written at the end: `{ metadata, best_configurations, test_results }` |
| A JSON array of records | Hand-assembled, or an older export |
| A single JSON record | One run |

A JSONL file whose last line is truncated still loads — that happens when a
sweep is interrupted mid-run — and the incomplete record is skipped.

`public/sample-trn2.jsonl` is a real three-run Trainium2 sweep you can drop in
to see the viewer populated.

## Record shape

llm-optimizer writes each record as:

```json
{
  "config": {
    "client_args":     { "max_concurrency": 16, "num_prompts": 1000 },
    "server_args":     { "tensor_parallel_size": 8, "max_num_seqs": 64 },
    "server_cmd_args": ["--tensor-parallel-size=8", "--max-num-seqs=64"]
  },
  "results":     { "output_throughput": 1234.0, "mean_ttft_ms": 274.8 },
  "cmd":         "vllm serve ...",
  "constraints": [{ "metric": "mean_ttft_ms", "operator": "<", "value": 300.0 }],
  "metadata":    { "gpu_type": "trn2.48xlarge", "gpu_count": 16, "framework": "vllm-neuron" }
}
```

`lib/benchmark-data.ts` normalizes that into the shape the components read.
Two details are worth knowing if you touch it:

- **`server_args` means two different things.** llm-optimizer uses it for the
  resolved name → value map and puts the CLI strings in `server_cmd_args`. The
  viewer expects the map under `config.server` and the CLI strings under
  `config.server_args`. The normalizer swaps them.
- **`metadata` and `constraints` are per-record** in JSONL but live in a header
  in the aggregated `.json`. Both are hoisted to the run header above the
  metric cards, and dropped from the table columns since they repeat on every
  row.

## AWS Neuron runs

Trainium and Inferentia are served through the
[vLLM Neuron plugin](https://github.com/vllm-project/vllm-neuron), which takes
its settings as a JSON blob in `--additional-config`. That blob would be one
unreadable table column, so the normalizer unpacks it into `config.neuron`:

- `num_batched_tokens_buckets` / `num_seqs_buckets` — the shapes the Neuron
  compiler builds graphs for
- `prefill_buckets` / `decode_buckets` — how many of each, which is what drives
  compile time, so these are sortable columns
- `on_device_sampling` — `greedy`, or the configured sampling parameters

The bucket lists themselves stay off the table (they are long, and the full
blob is in `cmd`). Neuron runs are labelled in the run header; NVIDIA results
are unaffected, since these fields simply never appear.

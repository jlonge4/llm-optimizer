# Tuning vLLM on AWS Neuron for an SLA

Guidance for choosing serving parameters on Trainium/Inferentia, and for
deciding what `llm-optimizer` should sweep.

Most vLLM tuning advice assumes the upstream scheduler. The Neuron scheduler
differs in ways that invert some of it, so start here rather than with a
generic vLLM guide.

## The one mechanic that drives everything

Three properties of `NeuronScheduler`, all from the plugin's own design docs:

1. **One prefill per iteration.** `max_prefills_per_batch = 1`, hardcoded and
   not configurable.
2. **Prefill and decode never share a batch.** Unlike upstream chunked prefill,
   the two phases always run in separate iterations.
3. **Prefill has priority.** When decode frees capacity and prefills are
   waiting, the next iteration is a prefill.

Put together: **every prefill freezes every in-flight decode.** The design doc
states it directly — "decode requests are hidden during each prefill
iteration."

The consequence, which is the part worth internalizing:

> Your inter-token latency is not set by how fast decode runs. It is set by how
> often a prefill lands and how long that prefill takes.

A single decode step might be 15 ms while a 8K-token prefill takes 400 ms. Every
request in the batch sees that 400 ms as one enormous inter-token gap. Mean ITL
hides this completely; it shows up in p95/p99 ITL and in `max_itl_ms`.

So on Neuron, **tail ITL is a prefill-scheduling problem**, and the levers are
different from the ones you would reach for on a GPU.

## Tuning by SLA

### "Inter-token latency must stay under X" (streaming UX)

The target is to shrink the stall a prefill imposes on everyone else.

| Lever | Direction | Why |
|---|---|---|
| `max_num_batched_tokens` | **Lower** | Caps how long any single prefill iteration can run. A long prompt is segmented into several shorter iterations, so decodes resume between segments instead of waiting for the whole prompt. This is the primary ITL lever and it is Neuron-specific. |
| `max_num_seqs` | Lower | Fewer decodes are stalled per prefill, and the decode batch is smaller. Costs throughput. |
| Disaggregated inference (1P1D / xPyD) | Enable | The real fix — see [Disaggregated prefill/decode](#disaggregated-prefilldecode). Prefill runs on separate instances, so decode instances never stall. Worth the operational cost when tail ITL is a hard SLA. |
| Speculative decoding (EAGLE3) | Enable | Attacks decode-step time directly. Best where draft acceptance is high — code, structured output. |
| Async scheduling | Keep on | On by default; requires `on_device_sampling_config`. |

Note the direction on `max_num_batched_tokens` is the **opposite** of what
helps TTFT. That tension is the central trade-off on Neuron, and it is the one
worth sweeping.

One subtlety on async scheduling: the async flow breaks whenever batch
composition changes — a request finishing or joining. Workloads with uniform
output lengths hold the async path far more of the time than workloads mixing
20-token and 2000-token generations. If your ITL is worse than the model math
suggests, look at output-length variance before touching parameters.

### "Time to first token must stay under X"

Under any real concurrency, TTFT is dominated by **queueing**, not by prefill
compute — because only one prefill runs at a time. With a 300 ms prefill and
four requests queued ahead of you, your TTFT floor is 1.2 s no matter how the
model is sharded.

| Lever | Direction | Why |
|---|---|---|
| Prefix caching | Enable + size it | On by default, but useless without blocks to retain prefixes. Set `--block-size` and raise `--num-gpu-blocks-override`. The largest TTFT win available for shared system prompts, RAG context, or chat history. |
| `max_num_batched_tokens` | **Raise** | So a typical prompt completes in one iteration rather than several. Directly opposes the ITL lever. |
| Prefill bucket alignment | Match traffic | Prompts pad up to the next bucket, and you pay compute for the padding. Put buckets at your p50/p90/p99 prompt lengths, not at round numbers. |
| Tensor parallelism | Raise | Prefill is compute-bound, so it benefits from more shards — up to the point collectives dominate. |
| Data parallelism | Raise | Each replica has its own scheduler and therefore its own prefill slot. See below. |

### "Maximize throughput"

**Prefer data parallelism over a single wide tensor-parallel replica.** This is
the highest-leverage choice on Neuron and follows directly from the one-prefill
rule: `TP8 × DP4` gives you four independent schedulers and therefore four
concurrent prefills, where `TP32 × DP1` gives you one. Use the smallest TP that
fits the model and meets latency, then spend the remaining chips on replicas.

Then:

- Raise `max_num_seqs` as far as KV cache allows — larger decode batches
  amortize weight loading.
- Raise `max_num_batched_tokens` so prefills finish in one iteration.
- For MoE models, add expert parallelism when high TP shards the intermediate
  dimension too thin. Note `ep_degree` defaults to TP, and DP + EP requires
  `ep_degree = TP × DP`.

### "Throughput inside a latency budget"

The realistic goal, and what the viewer is built around. The shape of the
answer is usually:

1. Fix TP at the smallest value that fits the model.
2. Sweep `max_num_seqs` and `max_num_batched_tokens` together — they trade
   against each other through the prefill-stall mechanic.
3. Spend leftover chips on DP.
4. Rank by throughput, constrained on p99 ITL or p99 TTFT rather than the mean.

Rank on p99, not mean. The prefill stall is invisible in the mean and is
exactly what the budget is meant to catch.

## Bucket configuration

Buckets are compile-time shapes, and getting them wrong either fails at startup
or silently wastes compute.

Hard rules enforced by the plugin:

- The last prefill bucket **must equal** `max_num_batched_tokens`.
- The last decode bucket **must equal** `max_num_seqs`.
- Buckets must be positive integers in strictly ascending order.

Soft guidance:

- More buckets means slower startup and more device memory, but less padding
  waste. Each bucket is a separate NEFF compilation.
- Start minimal during development, profile the real prompt-length and
  concurrency distribution, then add buckets at those percentiles.
- Padding waste is real compute: a 600-token prompt in a `[128, 256, 512, 1024]`
  bucket set is processed as 1024 tokens.

`llm-optimizer` generates power-of-two buckets from 128 (prefill) and 1
(decode), which matches the plugin's own defaults. Override them via
`--additional-config` once you have profiled your traffic.

## Disaggregated prefill/decode

Everything above is mitigation. Disaggregation removes the mechanic.

Split prefill and decode onto separate instances and the decode workers'
scheduler never enters `ACTIVE_PREFILL` — no prefills arrive there. The stall
does not get shorter; it stops existing. Every compromise in the ITL section is
a workaround for something this deletes.

### What changes

**The central trade-off becomes two independent knobs.** `max_num_batched_tokens`
pulls TTFT and tail ITL in opposite directions only because one instance serves
both phases. Split them and each pool gets its own config: raise it on the
prefill workers so prompts finish in a single iteration, and on the decode
workers it barely matters. You stop trading and set both.

**xPyD is cheaper than TP x DP for the same concurrency.** The throughput advice
above is "prefer replicas, because each has its own scheduler and therefore its
own prefill slot." xPyD gets you *x* concurrent prefills without paying for a
full decode-capable replica per slot, and lets you size the two pools to your
actual prefill:decode ratio rather than buying them in fixed pairs.

**KV-aware routing amplifies prefix caching.** Prefix caching is the largest TTFT
lever, but a single instance can only reuse what is in its own blocks. A router
that sends a request to the worker already holding its prefix makes hit rate
scale with the pool, which raises the payoff on a generous
`--num-gpu-blocks-override`.

### Features gated behind it

Two decode-side levers are rejected at startup without `--kv-transfer-config`:

- **Component data parallelism** — per-layer DP for attention, MLP, embedding or
  LM head. Decode-only: `kv_role` must be `kv_consumer` or `kv_both`, and
  `data_parallel_size >= 2`.
- **Decode context parallelism (DCP)** — shards the KV sequence across ranks for
  long context. It can run single-node, but once a connector is set for a P/D
  split the plugin requires `kv_connector == "NeuronNixlConnector"`. Constraints:
  `tp >= dcp * num_kv_heads`, `tp % dcp == 0`, `gqa_ratio % dcp == 0`, and
  `long_prefill_token_threshold` divisible by `dcp * block_size`.

### Running it under Dynamo

[ai-dynamo/dynamo#13771](https://github.com/ai-dynamo/dynamo/pull/13771)
registers `NeuronNixlConnector` in Dynamo's KV connector protocol registry,
which is what lets a Dynamo frontend validate and route through Neuron
disaggregated workers. It is the same pull-based wire format as the upstream
`NixlConnector`, extended with DCP-aware topology handling. Validated on
trn2.48xlarge across multi-node 1P1D over EFA, single-node P/D on separate
NeuronCores, and KV-aware routing with prefix caching.

KV transfer runs over NIXL/LIBFABRIC (EFA) and needs `libcuda.so.1` and
`libfabric.so.1` on the loader path.

### The gap in llm-optimizer

llm-optimizer benchmarks a single server that it launches itself, so it cannot
express a P/D topology and cannot sweep any of this directly. The workable path
today is `--server-cmd` pointed at an already-running Dynamo frontend, treating
the topology as opaque and sweeping client-side parameters. The P/D shape can
ride along in the results `metadata`, which the viewer already carries and can
filter on. Making disaggregated topologies first-class is real work, not a flag.

## Compile cost in a sweep

A Neuron cold start compiles the model to NEFFs before it serves anything, and
that takes tens of minutes. Two things follow for benchmarking.

**Server args decide the compile; client args are free.** `max_num_seqs`,
`max_num_batched_tokens`, TP/DP, `block_size` and the bucket lists change the
NEFF set. Concurrency, prompt count and sequence lengths only select among
already-compiled buckets. So `llm-optimizer` groups runs by their server
arguments and starts one server per group, looping the client inside — a sweep
of 3 shapes x 4 concurrencies is 3 compiles, not 12. This also skips the model
load and warmup between runs.

The practical consequence for designing a sweep: **vary concurrency freely, and
be deliberate about server arguments.** Each distinct combination of server
arguments is a fresh compile.

Because a reused server carries its prefix cache from one run into the next —
which flatters the later runs' TTFT and makes them incomparable — prefix
caching is disabled automatically when a server is shared across runs. Set
`enable_prefix_caching` explicitly in `--server-args` to override. Each result
records `server_shared`, `runs_in_shape` and `prefix_caching_disabled` in its
metadata.

**Readiness timeout.** The default is 300 s, raised to 5400 s automatically when
`--gpu` names a Neuron SKU. Override with `--server-timeout`. Without this a
cold compile fails the run before the server ever comes up.

**Reuse the compile cache across sweeps.** Set `NEURON_COMPILED_ARTIFACTS` to a
persistent path; it is inherited by the server process. Related knobs:

| Variable | Use |
|---|---|
| `NEURON_COMPILED_ARTIFACTS` | Cache path. The single most useful setting for repeated sweeps. |
| `NEURON_LIBTORCH_PARALLEL_COMPILE_WORKERS` | Parallelise compilation. |
| `VLLM_NEURON_DISABLE_WARMUP_COMPILE=1` | Treat a cache miss as fatal. Use it to verify a warmed cache actually covers every shape in the sweep, so a gap fails in seconds instead of mid-run. |
| `VLLM_NEURON_CPU_COMPILE=1` + `NEURON_PLATFORM_TARGET_OVERRIDE=trn2` | Compile without Neuron hardware, so the cache can be built off-instance. |

## Other Neuron-specific constraints

- **Tensor parallelism must be a power of two.**
- **`block_size` is 16 or 32**, defaulting to 32 — unlike vLLM's 16.
- **Not implemented by the plugin:** pipeline parallelism, LoRA adapters, sleep
  mode, KV cache offloading. Chunked prefill exists only at batch size 1 and
  never mixes phases.
- **On-device sampling is required** for async scheduling, speculative
  decoding, and structured output enforcement. Set
  `on_device_sampling_config`; `{"all_greedy": true}` for benchmarking.
- **Compilation is not free.** Every distinct bucket set is a separate compile.
  Set `NEURON_COMPILED_ARTIFACTS` to cache across runs, or a sweep spends most
  of its wall time in the compiler rather than benchmarking.

## Sweeping this with llm-optimizer

`--framework vllm` on a Neuron SKU routes to the `vllm-neuron` framework, which
already restricts TP to powers of two, pins bucket lists to each config's
maxima, and drops unsupported arguments.

```bash
llm-optimizer \
  --framework vllm \
  --gpu trn2.48xlarge \
  --gpus 16 \
  --model meta-llama/Llama-3.1-8B \
  --server-args "max_num_seqs=[8,16,32,64];max_num_batched_tokens=[4096,8192,16384]" \
  --client-args "dataset_name=random;random_input_len=1024;random_output_len=256" \
  --output-json results.json
```

Pass `--gpu` explicitly: NVML cannot identify Neuron devices, so without it the
results are recorded as `unknown`.

Then open `viewer/` and rank by `output_throughput` with a budget on
`p99_itl_ms` — the metric the prefill stall actually shows up in.

## Sources and caveats

The scheduler behavior, hard constraints, and feature support are from the
vLLM Neuron plugin's own documentation and source:
`docs/design/vllm/neuron-scheduler.md`, `docs/guides/features-guide.md`, and
`docs/guides/reference-configuration.md`.

The Dynamo integration details are from the merged PR linked above.

The causal argument connecting the one-prefill rule to tail ITL, and the
resulting parameter directions, is synthesis rather than something AWS states
in those terms. It follows from the documented mechanics, but the specific
numbers depend on your model and traffic — measure rather than assume. No
figures in this document come from a benchmark run on real hardware.

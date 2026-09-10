"use client"

import { useState } from "react"
import { Upload, FileJson, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { parseBenchmarkInput, type ParsedBenchmarkData } from "@/lib/benchmark-data"

// One record per line is what `llm-optimizer --output-json results.json` appends
// to results.jsonl. This sample is a Trainium2 run, so it carries the Neuron
// plugin's --additional-config as well.
const EXAMPLE_JSONL = [
  {
    config: {
      client_args: {
        num_prompts: 1000,
        max_concurrency: 16,
        dataset_name: "random",
        random_input_len: 1024,
        random_output_len: 256,
      },
      server_args: {
        max_num_seqs: 16,
        tensor_parallel_size: 1,
        data_parallel_size: 16,
        max_num_batched_tokens: 16384,
        block_size: 32,
        additional_config:
          '{"neuron_config":{"num_batched_tokens_buckets":[128,256,512,1024,2048,4096,8192,16384],"num_seqs_buckets":[1,2,4,8,16],"on_device_sampling_config":{"all_greedy":true}}}',
      },
      server_cmd_args: [
        "--max-num-seqs=16",
        "--tensor-parallel-size=1",
        "--data-parallel-size=16",
        "--max-num-batched-tokens=16384",
        "--block-size=32",
      ],
    },
    results: {
      backend: "vllm",
      dataset_name: "random",
      max_concurrency: 16,
      total_input_tokens: 1024000,
      total_output_tokens: 256000,
      request_throughput: 2.1729,
      input_throughput: 2225.2,
      output_throughput: 556.3,
      mean_e2e_latency_ms: 3441.46,
      median_e2e_latency_ms: 3413.32,
      p99_e2e_latency_ms: 4004.0,
      mean_ttft_ms: 185.23,
      median_ttft_ms: 178.45,
      p99_ttft_ms: 395.67,
      mean_tpot_ms: 13.12,
      mean_itl_ms: 13.05,
    },
    cmd: "vllm serve meta-llama/Llama-3.1-8B --host 127.0.0.1 --port 8000 --max-num-seqs=16 --tensor-parallel-size=1 --data-parallel-size=16 --max-num-batched-tokens=16384 --block-size=32",
    constraints: [{ metric: "mean_ttft_ms", operator: "<", value: 300.0 }],
    metadata: {
      gpu_type: "trn2.48xlarge",
      gpu_count: 16,
      framework: "vllm-neuron",
      model_tag: "meta-llama/Llama-3.1-8B",
      input_tokens: 1024,
      output_tokens: 256,
    },
  },
  {
    config: {
      client_args: {
        num_prompts: 1000,
        max_concurrency: 64,
        dataset_name: "random",
        random_input_len: 1024,
        random_output_len: 256,
      },
      server_args: {
        max_num_seqs: 64,
        tensor_parallel_size: 8,
        data_parallel_size: 2,
        max_num_batched_tokens: 32768,
        block_size: 32,
        additional_config:
          '{"neuron_config":{"num_batched_tokens_buckets":[128,256,512,1024,2048,4096,8192,16384,32768],"num_seqs_buckets":[1,2,4,8,16,32,64],"on_device_sampling_config":{"all_greedy":true}}}',
      },
      server_cmd_args: [
        "--max-num-seqs=64",
        "--tensor-parallel-size=8",
        "--data-parallel-size=2",
        "--max-num-batched-tokens=32768",
        "--block-size=32",
      ],
    },
    results: {
      backend: "vllm",
      dataset_name: "random",
      max_concurrency: 64,
      total_input_tokens: 1024000,
      total_output_tokens: 256000,
      request_throughput: 4.8203,
      input_throughput: 4936.0,
      output_throughput: 1234.0,
      mean_e2e_latency_ms: 4871.2,
      median_e2e_latency_ms: 4790.5,
      p99_e2e_latency_ms: 6902.3,
      mean_ttft_ms: 274.8,
      median_ttft_ms: 261.4,
      p99_ttft_ms: 612.9,
      mean_tpot_ms: 17.9,
      mean_itl_ms: 17.6,
    },
    cmd: "vllm serve meta-llama/Llama-3.1-8B --host 127.0.0.1 --port 8000 --max-num-seqs=64 --tensor-parallel-size=8 --data-parallel-size=2 --max-num-batched-tokens=32768 --block-size=32",
    constraints: [{ metric: "mean_ttft_ms", operator: "<", value: 300.0 }],
    metadata: {
      gpu_type: "trn2.48xlarge",
      gpu_count: 16,
      framework: "vllm-neuron",
      model_tag: "meta-llama/Llama-3.1-8B",
      input_tokens: 1024,
      output_tokens: 256,
    },
  },
]

const EXAMPLE_TEXT = EXAMPLE_JSONL.map((r) => JSON.stringify(r)).join("\n")

interface JsonInputProps {
  onDataLoaded: (data: ParsedBenchmarkData) => void
  hasData: boolean
  onClear: () => void
}

export function JsonInput({ onDataLoaded, hasData, onClear }: JsonInputProps) {
  const [rawInput, setRawInput] = useState("")
  const [error, setError] = useState<string | null>(null)
  const [isDragOver, setIsDragOver] = useState(false)

  function load(text: string) {
    try {
      const parsed = parseBenchmarkInput(text)
      setError(null)
      setRawInput("")
      onDataLoaded(parsed)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not read this input.")
    }
  }

  function handleParse() {
    load(rawInput)
  }

  function handleLoadExample() {
    load(EXAMPLE_TEXT)
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault()
    setIsDragOver(false)
    const file = e.dataTransfer.files[0]
    if (!file) return

    // .jsonl files usually arrive with an empty MIME type, so accept the drop
    // and let the parser decide rather than filtering on file.type.
    const reader = new FileReader()
    reader.onload = (evt) => load(evt.target?.result as string)
    reader.onerror = () => setError("Could not read that file.")
    reader.readAsText(file)
  }

  if (hasData) {
    return (
      <div className="flex items-center gap-3">
        <Button
          variant="outline"
          size="sm"
          onClick={onClear}
          className="gap-2 border-border bg-secondary text-secondary-foreground hover:bg-muted"
        >
          <X className="h-4 w-4" />
          Clear Data
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={handleLoadExample}
          className="gap-2 border-border bg-secondary text-secondary-foreground hover:bg-muted"
        >
          <FileJson className="h-4 w-4" />
          Load Example
        </Button>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-6">
      <div
        className={`relative rounded-lg border-2 border-dashed transition-colors ${
          isDragOver
            ? "border-primary bg-primary/5"
            : "border-border bg-card"
        }`}
        onDragOver={(e) => {
          e.preventDefault()
          setIsDragOver(true)
        }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={handleDrop}
      >
        <div className="p-6">
          <div className="mb-4 flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
              <Upload className="h-5 w-5 text-primary" />
            </div>
            <div>
              <p className="text-sm font-medium text-foreground">
                Paste results or drop a .json / .jsonl file
              </p>
              <p className="text-xs text-muted-foreground">
                llm-optimizer output: results.jsonl, or the aggregated results.json
              </p>
            </div>
          </div>
          <textarea
            className="min-h-[180px] w-full resize-y rounded-md border border-border bg-secondary p-4 font-mono text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
            placeholder={`{"config": {...}, "results": {...}, "cmd": "vllm serve ...", "metadata": {...}}\n{"config": {...}, "results": {...}, "cmd": "vllm serve ...", "metadata": {...}}`}
            value={rawInput}
            onChange={(e) => {
              setRawInput(e.target.value)
              setError(null)
            }}
          />
          {error && (
            <p className="mt-2 text-sm text-destructive">{error}</p>
          )}
          <div className="mt-4 flex flex-col gap-3 sm:flex-row">
            <Button
              onClick={handleParse}
              disabled={!rawInput.trim()}
              className="gap-2 bg-primary text-primary-foreground hover:bg-primary/90"
            >
              <FileJson className="h-4 w-4" />
              Parse & Visualize
            </Button>
            <Button
              variant="outline"
              onClick={handleLoadExample}
              className="gap-2 border-border bg-secondary text-secondary-foreground hover:bg-muted"
            >
              Load Example Data
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}

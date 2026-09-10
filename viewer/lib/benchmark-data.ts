/**
 * Parsing and normalization for llm-optimizer benchmark output.
 *
 * llm-optimizer writes results in two shapes, and the viewer accepts both:
 *
 *   1. `<name>.jsonl` — one record per line, appended as each run finishes.
 *   2. `<name>.json`  — the aggregated file written at the end, wrapping the
 *      same records in `{ metadata, best_configurations, test_results }`.
 *
 * The record keys also differ from what the viewer's components read. Most
 * importantly `server_args` means different things on each side: llm-optimizer
 * uses it for the resolved name -> value map and puts the CLI strings in
 * `server_cmd_args`, while the viewer expects the CLI strings under
 * `server_args` and the map under `server`. Normalizing here keeps that
 * translation in one place instead of spreading it across every component.
 */

export interface BenchmarkMetadata {
  gpu_type?: string
  gpu_count?: number
  framework?: string
  model_tag?: string
  input_tokens?: number
  output_tokens?: number
  total_tests?: number
}

export interface Constraint {
  metric: string
  operator: string
  value: number
}

/**
 * AWS Neuron settings, unpacked from the `--additional-config` JSON blob that
 * the vLLM Neuron plugin takes. Kept separate so the table can show the
 * compiled bucket shapes as their own columns rather than one unreadable
 * string.
 */
export interface NeuronConfig {
  num_batched_tokens_buckets?: number[]
  num_seqs_buckets?: number[]
  on_device_sampling?: string
  prefill_buckets?: number
  decode_buckets?: number
}

export interface BenchmarkRecord {
  config: {
    client: Record<string, unknown>
    server: Record<string, unknown>
    server_args: string[]
    neuron?: NeuronConfig
  }
  results: Record<string, unknown>
  cmd?: string
  metadata?: BenchmarkMetadata
  [key: string]: unknown
}

export interface ParsedBenchmarkData {
  records: BenchmarkRecord[]
  metadata: BenchmarkMetadata
  constraints: Constraint[]
}

/** True when a value is a plain object we can treat as a key/value map. */
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

function asStringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.map(String) : []
}

/**
 * Unpack the vLLM Neuron `--additional-config` payload.
 *
 * The value arrives as a JSON string because it travels through the CLI, so a
 * malformed or absent blob simply yields no Neuron section rather than an
 * error — a benchmark run is still viewable without it.
 */
export function parseNeuronConfig(raw: unknown): NeuronConfig | undefined {
  let payload: unknown = raw

  if (typeof raw === "string") {
    try {
      payload = JSON.parse(raw)
    } catch {
      return undefined
    }
  }

  if (!isRecord(payload)) return undefined

  const neuron = payload.neuron_config
  if (!isRecord(neuron)) return undefined

  const prefill = Array.isArray(neuron.num_batched_tokens_buckets)
    ? (neuron.num_batched_tokens_buckets as number[])
    : undefined
  const decode = Array.isArray(neuron.num_seqs_buckets)
    ? (neuron.num_seqs_buckets as number[])
    : undefined

  const sampling = neuron.on_device_sampling_config
  let samplingLabel: string | undefined
  if (isRecord(sampling)) {
    samplingLabel = sampling.all_greedy
      ? "greedy"
      : Object.entries(sampling)
          .map(([k, v]) => `${k}=${v}`)
          .join(", ") || "on-device"
  }

  return {
    num_batched_tokens_buckets: prefill,
    num_seqs_buckets: decode,
    on_device_sampling: samplingLabel,
    // Bucket counts drive compile time, so they are worth sorting on directly.
    prefill_buckets: prefill?.length,
    decode_buckets: decode?.length,
  }
}

/**
 * Convert one raw llm-optimizer record into the shape the viewer reads.
 *
 * Records that already use the viewer's own key names pass through unchanged,
 * so previously exported data and the bundled example keep working.
 */
export function normalizeRecord(raw: Record<string, unknown>): BenchmarkRecord {
  const config = isRecord(raw.config) ? raw.config : {}

  const client = isRecord(config.client_args)
    ? config.client_args
    : isRecord(config.client)
      ? config.client
      : {}

  // llm-optimizer's `server_args` is the resolved map; the viewer's is the CLI
  // string list. Whichever this file uses, put each in the right place.
  const server = isRecord(config.server_args)
    ? config.server_args
    : isRecord(config.server)
      ? config.server
      : {}

  const serverArgs = asStringArray(
    config.server_cmd_args ?? (Array.isArray(config.server_args) ? config.server_args : [])
  )

  const neuron = parseNeuronConfig(server.additional_config)

  // The raw JSON blob would become a single very wide column, and everything
  // in it is already surfaced under `neuron` (and in full in `cmd`).
  const { additional_config: _dropped, ...serverRest } = server

  const record: BenchmarkRecord = {
    config: {
      client,
      server: serverRest,
      server_args: serverArgs,
      ...(neuron ? { neuron } : {}),
    },
    results: isRecord(raw.results) ? raw.results : {},
  }

  if (typeof raw.cmd === "string") record.cmd = raw.cmd
  if (isRecord(raw.metadata)) record.metadata = raw.metadata as BenchmarkMetadata

  return record
}

/** Pull the constraint list off a record or an aggregated file. */
function readConstraints(value: unknown): Constraint[] {
  if (!Array.isArray(value)) return []
  return value.filter(
    (c): c is Constraint =>
      isRecord(c) && typeof c.metric === "string" && typeof c.operator === "string"
  )
}

/**
 * Parse benchmark text in any shape llm-optimizer produces.
 *
 * Accepts a JSON array, a single JSON object, the aggregated
 * `{ metadata, test_results }` wrapper, or JSONL. JSONL is tried last because
 * a pretty-printed JSON file also spans many lines but is not line-delimited.
 *
 * @param text Raw file or pasted contents
 * @returns Normalized records plus any run-level metadata and constraints
 * @throws If nothing in the input parses as benchmark records
 */
export function parseBenchmarkInput(text: string): ParsedBenchmarkData {
  const trimmed = text.trim()
  if (!trimmed) throw new Error("No data to parse.")

  let rawRecords: Record<string, unknown>[] | null = null
  let metadata: BenchmarkMetadata = {}
  let constraints: Constraint[] = []

  try {
    const parsed = JSON.parse(trimmed)

    if (Array.isArray(parsed)) {
      rawRecords = parsed.filter(isRecord)
    } else if (isRecord(parsed) && Array.isArray(parsed.test_results)) {
      // The aggregated file: run metadata sits alongside the records.
      rawRecords = parsed.test_results.filter(isRecord)
      if (isRecord(parsed.metadata)) {
        metadata = parsed.metadata as BenchmarkMetadata
        constraints = readConstraints(parsed.metadata.constraints)
      }
    } else if (isRecord(parsed)) {
      rawRecords = [parsed]
    }
  } catch {
    // Not a single JSON document — fall through to JSONL.
  }

  if (rawRecords === null) {
    const lines = trimmed.split("\n").filter((line) => line.trim())
    const parsed: Record<string, unknown>[] = []

    for (const line of lines) {
      try {
        const value = JSON.parse(line)
        if (isRecord(value)) parsed.push(value)
      } catch {
        // Skip unparseable lines: llm-optimizer appends to the JSONL as each
        // run finishes, so an interrupted run can leave a partial last line.
      }
    }

    if (parsed.length === 0) {
      throw new Error(
        "Could not read this as JSON or JSONL. Expected llm-optimizer output."
      )
    }
    rawRecords = parsed
  }

  if (rawRecords.length === 0) {
    throw new Error("No benchmark records found in this input.")
  }

  const records = rawRecords.map(normalizeRecord)

  // A JSONL file carries metadata and constraints on every record rather than
  // in a header, so take them from the first record that has them.
  if (!metadata.gpu_type) {
    const withMetadata = rawRecords.find((r) => isRecord(r.metadata))
    if (withMetadata) metadata = { ...(withMetadata.metadata as BenchmarkMetadata), ...metadata }
  }
  if (constraints.length === 0) {
    const withConstraints = rawRecords.find((r) => readConstraints(r.constraints).length > 0)
    if (withConstraints) constraints = readConstraints(withConstraints.constraints)
  }

  if (metadata.total_tests === undefined) {
    metadata.total_tests = records.length
  }

  return { records, metadata, constraints }
}

/** True when these results came from AWS Trainium/Inferentia. */
export function isNeuronRun(metadata: BenchmarkMetadata, records: BenchmarkRecord[]): boolean {
  if (metadata.framework?.includes("neuron")) return true
  if (/^(trn|inf)/i.test(metadata.gpu_type ?? "")) return true
  return records.some((r) => r.config.neuron !== undefined)
}


/** A metric that can be ranked, with the direction that counts as "better". */
export interface RankableMetric {
  key: string
  label: string
  betterWhen: "higher" | "lower"
}

/** Metrics where a bigger number is the better outcome. Everything else is a cost. */
const HIGHER_IS_BETTER = /throughput|completed|concurrency|accept_length/

/**
 * List the numeric result metrics present in these records.
 *
 * Driven by the data rather than a fixed list, so any metric llm-optimizer
 * grows shows up without a change here.
 */
export function rankableMetrics(records: BenchmarkRecord[]): RankableMetric[] {
  const keys = new Set<string>()
  for (const record of records) {
    for (const [key, value] of Object.entries(record.results)) {
      if (typeof value === "number" && Number.isFinite(value)) keys.add(key)
    }
  }

  return [...keys].sort().map((key) => ({
    key,
    label: key.replace(/_/g, " ").replace(/ ms$/, " (ms)"),
    betterWhen: HIGHER_IS_BETTER.test(key) ? "higher" : "lower",
  }))
}

/** Whether one record satisfies every SLO constraint. */
export function meetsConstraints(
  record: BenchmarkRecord,
  constraints: Constraint[]
): boolean {
  return constraints.every((c) => {
    const value = record.results[c.metric]
    if (typeof value !== "number") return false
    switch (c.operator) {
      case "<":
        return value < c.value
      case "<=":
        return value <= c.value
      case ">":
        return value > c.value
      case ">=":
        return value >= c.value
      case "=":
      case "==":
        return value === c.value
      default:
        return false
    }
  })
}

export interface WinnerSelection {
  /** Best run overall, ignoring SLOs. */
  overall?: BenchmarkRecord
  /** Best run that also satisfies every constraint. */
  withinSlo?: BenchmarkRecord
  /** How many runs satisfied the constraints. */
  passingCount: number
}

/**
 * Pick the best run by whichever metric matters to the caller.
 *
 * llm-optimizer's own `best_configurations` only ranks by input/output
 * throughput and is only written once a sweep completes, so ranking is done
 * here instead: it works on a partial JSONL and on any metric in the data.
 *
 * @param records Runs to rank
 * @param metricKey Result metric to rank by
 * @param direction Whether higher or lower wins
 * @param constraints SLOs used to compute the within-SLO winner
 */
export function pickWinner(
  records: BenchmarkRecord[],
  metricKey: string,
  direction: "higher" | "lower",
  constraints: Constraint[] = []
): WinnerSelection {
  const scored = records.filter(
    (r) => typeof r.results[metricKey] === "number" && Number.isFinite(r.results[metricKey] as number)
  )

  const best = (pool: BenchmarkRecord[]) =>
    pool.reduce<BenchmarkRecord | undefined>((winner, candidate) => {
      if (!winner) return candidate
      const a = candidate.results[metricKey] as number
      const b = winner.results[metricKey] as number
      return direction === "higher" ? (a > b ? candidate : winner) : a < b ? candidate : winner
    }, undefined)

  const passing = constraints.length > 0 ? scored.filter((r) => meetsConstraints(r, constraints)) : scored

  return {
    overall: best(scored),
    withinSlo: constraints.length > 0 ? best(passing) : undefined,
    passingCount: passing.length,
  }
}

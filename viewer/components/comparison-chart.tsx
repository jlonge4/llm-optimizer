"use client"

import { useState, useMemo } from "react"
import {
  Bar,
  BarChart,
  Line,
  LineChart,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  Legend,
} from "recharts"
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart"
import { ChevronDown } from "lucide-react"

interface ComparisonChartProps {
  data: Record<string, unknown>[]
}

// All numeric result keys we can compare
function getNumericResultKeys(data: Record<string, unknown>[]): string[] {
  const keys = new Set<string>()
  for (const entry of data) {
    const results = entry.results as Record<string, unknown> | undefined
    if (results) {
      for (const [k, v] of Object.entries(results)) {
        if (typeof v === "number") keys.add(k)
      }
    }
  }
  return Array.from(keys).sort()
}

function getNumericConfigKeys(data: Record<string, unknown>[]): string[] {
  const keys = new Set<string>()
  for (const entry of data) {
    const config = entry.config as Record<string, unknown> | undefined
    if (config) {
      for (const section of ["client", "server"]) {
        const sub = config[section] as Record<string, unknown> | undefined
        if (sub) {
          for (const [k, v] of Object.entries(sub)) {
            if (typeof v === "number") keys.add(`${section}.${k}`)
          }
        }
      }
    }
  }
  return Array.from(keys).sort()
}

function getNestedValue(obj: Record<string, unknown>, path: string): unknown {
  const parts = path.split(".")
  let current: unknown = obj
  for (const part of parts) {
    if (current && typeof current === "object" && !Array.isArray(current)) {
      current = (current as Record<string, unknown>)[part]
    } else {
      return undefined
    }
  }
  return current
}

// Friendly labels for metrics
const METRIC_LABELS: Record<string, string> = {
  request_throughput: "Req Throughput (req/s)",
  input_throughput: "Input Throughput (tok/s)",
  output_throughput: "Output Throughput (tok/s)",
  mean_e2e_latency_ms: "Mean E2E Latency (ms)",
  median_e2e_latency_ms: "Median E2E Latency (ms)",
  std_e2e_latency_ms: "Std E2E Latency (ms)",
  p99_e2e_latency_ms: "P99 E2E Latency (ms)",
  mean_ttft_ms: "Mean TTFT (ms)",
  median_ttft_ms: "Median TTFT (ms)",
  std_ttft_ms: "Std TTFT (ms)",
  p99_ttft_ms: "P99 TTFT (ms)",
  mean_tpot_ms: "Mean TPOT (ms)",
  median_tpot_ms: "Median TPOT (ms)",
  std_tpot_ms: "Std TPOT (ms)",
  p99_tpot_ms: "P99 TPOT (ms)",
  mean_itl_ms: "Mean ITL (ms)",
  median_itl_ms: "Median ITL (ms)",
  std_itl_ms: "Std ITL (ms)",
  p99_itl_ms: "P99 ITL (ms)",
  total_input_tokens: "Total Input Tokens",
  total_output_tokens: "Total Output Tokens",
  max_concurrency: "Max Concurrency",
}

// Grouped metric categories for easier selection
const METRIC_GROUPS: { label: string; keys: string[] }[] = [
  {
    label: "Throughput",
    keys: ["request_throughput", "input_throughput", "output_throughput"],
  },
  {
    label: "E2E Latency",
    keys: [
      "mean_e2e_latency_ms",
      "median_e2e_latency_ms",
      "std_e2e_latency_ms",
      "p99_e2e_latency_ms",
    ],
  },
  {
    label: "TTFT",
    keys: ["mean_ttft_ms", "median_ttft_ms", "std_ttft_ms", "p99_ttft_ms"],
  },
  {
    label: "TPOT",
    keys: ["mean_tpot_ms", "median_tpot_ms", "std_tpot_ms", "p99_tpot_ms"],
  },
  {
    label: "ITL",
    keys: ["mean_itl_ms", "median_itl_ms", "std_itl_ms", "p99_itl_ms"],
  },
]

// Palette of distinguishable bar colors
const BAR_COLORS = [
  "#2dd4bf", // teal-400
  "#60a5fa", // blue-400
  "#f59e0b", // amber-500
  "#f472b6", // pink-400
  "#a78bfa", // violet-400
  "#34d399", // emerald-400
  "#fb923c", // orange-400
  "#38bdf8", // sky-400
]

type ChartMode = "bar" | "line"

export function ComparisonChart({ data }: ComparisonChartProps) {
  const numericKeys = useMemo(() => getNumericResultKeys(data), [data])
  const configKeys = useMemo(() => getNumericConfigKeys(data), [data])
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>(() => {
    // Default to p99 e2e + mean ttft for a nice starting comparison
    const defaults = ["p99_e2e_latency_ms", "mean_ttft_ms"]
    return defaults.filter((d) => numericKeys.includes(d))
  })
  const [groupByKey, setGroupByKey] = useState<string>(() => {
    // Default group-by: a config param that varies (concurrency, TP, etc.)
    const preferred = [
      "client.max_concurrency",
      "server.tensor_parallel_size",
      "server.max_num_seqs",
    ]
    return preferred.find((p) => configKeys.includes(p)) || configKeys[0] || ""
  })
  const [chartMode, setChartMode] = useState<ChartMode>("bar")
  const [metricPickerOpen, setMetricPickerOpen] = useState(false)
  const [groupByPickerOpen, setGroupByPickerOpen] = useState(false)

  // Build chart data: each row is a benchmark run with group-by label + metric values
  const chartData = useMemo(() => {
    return data.map((entry, i) => {
      const results = entry.results as Record<string, unknown> | undefined
      const groupVal = getNestedValue(
        entry,
        groupByKey.startsWith("client.") || groupByKey.startsWith("server.")
          ? `config.${groupByKey}`
          : groupByKey
      )
      const row: Record<string, unknown> = {
        label: groupVal !== undefined ? String(groupVal) : `Run ${i + 1}`,
        _index: i,
      }
      for (const metric of selectedMetrics) {
        if (results && typeof results[metric] === "number") {
          row[metric] = results[metric]
        }
      }
      return row
    })
  }, [data, selectedMetrics, groupByKey])

  // Build recharts config
  const chartConfig = useMemo(() => {
    const config: Record<string, { label: string; color: string }> = {}
    selectedMetrics.forEach((m, i) => {
      config[m] = {
        label: METRIC_LABELS[m] || m,
        color: BAR_COLORS[i % BAR_COLORS.length],
      }
    })
    return config
  }, [selectedMetrics])

  function toggleMetric(key: string) {
    setSelectedMetrics((prev) =>
      prev.includes(key)
        ? prev.filter((k) => k !== key)
        : [...prev, key]
    )
  }

  if (data.length < 2) {
    return (
      <div className="flex flex-col items-center justify-center gap-3 rounded-lg border border-border bg-card py-16">
        <p className="text-sm text-muted-foreground">
          Load at least 2 benchmark runs to compare
        </p>
        <p className="text-xs text-muted-foreground">
          Paste a JSON array with multiple benchmark entries
        </p>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-4">
      {/* Controls */}
      <div className="flex flex-col gap-3 rounded-lg border border-border bg-card p-4 sm:flex-row sm:items-end">
        {/* Group By selector */}
        <div className="relative flex-1">
          <label className="mb-1 block text-xs font-medium text-muted-foreground">
            Group By (X-Axis)
          </label>
          <button
            onClick={() => {
              setGroupByPickerOpen(!groupByPickerOpen)
              setMetricPickerOpen(false)
            }}
            className="flex h-9 w-full items-center justify-between rounded-md border border-border bg-secondary px-3 text-sm text-foreground"
          >
            <span className="truncate font-mono text-xs">
              {groupByKey || "Select..."}
            </span>
            <ChevronDown className="h-4 w-4 flex-shrink-0 text-muted-foreground" />
          </button>
          {groupByPickerOpen && (
            <div className="absolute left-0 top-[calc(100%+4px)] z-50 max-h-56 w-full overflow-y-auto rounded-md border border-border bg-popover shadow-lg">
              {configKeys.map((key) => (
                <button
                  key={key}
                  className={`w-full px-3 py-2 text-left font-mono text-xs hover:bg-muted ${
                    groupByKey === key
                      ? "bg-primary/10 text-primary"
                      : "text-popover-foreground"
                  }`}
                  onClick={() => {
                    setGroupByKey(key)
                    setGroupByPickerOpen(false)
                  }}
                >
                  {key}
                </button>
              ))}
              {/* Also allow grouping by result fields */}
              <div className="border-t border-border px-3 py-1.5 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
                Result Fields
              </div>
              {numericKeys.slice(0, 10).map((key) => (
                <button
                  key={key}
                  className={`w-full px-3 py-2 text-left font-mono text-xs hover:bg-muted ${
                    groupByKey === key
                      ? "bg-primary/10 text-primary"
                      : "text-popover-foreground"
                  }`}
                  onClick={() => {
                    setGroupByKey(key)
                    setGroupByPickerOpen(false)
                  }}
                >
                  {key}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Metric selector */}
        <div className="relative flex-[2]">
          <label className="mb-1 block text-xs font-medium text-muted-foreground">
            Metrics to Compare
          </label>
          <button
            onClick={() => {
              setMetricPickerOpen(!metricPickerOpen)
              setGroupByPickerOpen(false)
            }}
            className="flex h-9 w-full items-center justify-between rounded-md border border-border bg-secondary px-3 text-sm text-foreground"
          >
            <span className="truncate text-xs">
              {selectedMetrics.length === 0
                ? "Select metrics..."
                : `${selectedMetrics.length} metric${selectedMetrics.length > 1 ? "s" : ""} selected`}
            </span>
            <ChevronDown className="h-4 w-4 flex-shrink-0 text-muted-foreground" />
          </button>
          {metricPickerOpen && (
            <div className="absolute left-0 top-[calc(100%+4px)] z-50 max-h-72 w-full overflow-y-auto rounded-md border border-border bg-popover p-2 shadow-lg">
              {/* Grouped metrics */}
              {METRIC_GROUPS.map((group) => {
                const groupMetrics = group.keys.filter((k) =>
                  numericKeys.includes(k)
                )
                if (groupMetrics.length === 0) return null
                return (
                  <div key={group.label} className="mb-2">
                    <div className="flex items-center justify-between px-2 py-1">
                      <span className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
                        {group.label}
                      </span>
                      <button
                        className="text-[10px] text-primary hover:underline"
                        onClick={() => {
                          const allSelected = groupMetrics.every((k) =>
                            selectedMetrics.includes(k)
                          )
                          if (allSelected) {
                            setSelectedMetrics((prev) =>
                              prev.filter(
                                (p) => !groupMetrics.includes(p)
                              )
                            )
                          } else {
                            setSelectedMetrics((prev) => [
                              ...prev,
                              ...groupMetrics.filter(
                                (k) => !prev.includes(k)
                              ),
                            ])
                          }
                        }}
                      >
                        {groupMetrics.every((k) =>
                          selectedMetrics.includes(k)
                        )
                          ? "Deselect all"
                          : "Select all"}
                      </button>
                    </div>
                    {groupMetrics.map((key) => {
                      const colorIdx = selectedMetrics.indexOf(key)
                      return (
                        <label
                          key={key}
                          className="flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 hover:bg-muted"
                        >
                          <input
                            type="checkbox"
                            checked={selectedMetrics.includes(key)}
                            onChange={() => toggleMetric(key)}
                            className="h-3.5 w-3.5 rounded border-border accent-primary"
                          />
                          {colorIdx >= 0 && (
                            <span
                              className="inline-block h-2.5 w-2.5 rounded-sm"
                              style={{
                                backgroundColor:
                                  BAR_COLORS[colorIdx % BAR_COLORS.length],
                              }}
                            />
                          )}
                          <span className="font-mono text-xs text-foreground">
                            {METRIC_LABELS[key] || key}
                          </span>
                        </label>
                      )
                    })}
                  </div>
                )
              })}
              {/* Ungrouped metrics */}
              {(() => {
                const groupedKeys = METRIC_GROUPS.flatMap((g) => g.keys)
                const ungrouped = numericKeys.filter(
                  (k) => !groupedKeys.includes(k)
                )
                if (ungrouped.length === 0) return null
                return (
                  <div className="mb-2">
                    <div className="px-2 py-1 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
                      Other
                    </div>
                    {ungrouped.map((key) => (
                      <label
                        key={key}
                        className="flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 hover:bg-muted"
                      >
                        <input
                          type="checkbox"
                          checked={selectedMetrics.includes(key)}
                          onChange={() => toggleMetric(key)}
                          className="h-3.5 w-3.5 rounded border-border accent-primary"
                        />
                        <span className="font-mono text-xs text-foreground">
                          {METRIC_LABELS[key] || key}
                        </span>
                      </label>
                    ))}
                  </div>
                )
              })()}
              <div className="border-t border-border pt-2">
                <button
                  onClick={() => setMetricPickerOpen(false)}
                  className="w-full rounded-md bg-secondary py-1.5 text-xs text-secondary-foreground hover:bg-muted"
                >
                  Done
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Chart mode toggle */}
        <div className="flex gap-1 rounded-md border border-border bg-secondary p-0.5">
          <button
            onClick={() => setChartMode("bar")}
            className={`rounded px-3 py-1.5 text-xs font-medium transition-colors ${
              chartMode === "bar"
                ? "bg-card text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            Bar
          </button>
          <button
            onClick={() => setChartMode("line")}
            className={`rounded px-3 py-1.5 text-xs font-medium transition-colors ${
              chartMode === "line"
                ? "bg-card text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            Line
          </button>
        </div>
      </div>

      {/* Selected metric chips */}
      {selectedMetrics.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {selectedMetrics.map((m, i) => (
            <button
              key={m}
              onClick={() => toggleMetric(m)}
              className="flex items-center gap-1.5 rounded-md border border-border bg-card px-2.5 py-1 text-xs text-foreground transition-colors hover:bg-muted"
            >
              <span
                className="inline-block h-2 w-2 rounded-sm"
                style={{
                  backgroundColor: BAR_COLORS[i % BAR_COLORS.length],
                }}
              />
              <span className="font-mono">
                {METRIC_LABELS[m]?.split(" (")[0] || m}
              </span>
              <span className="ml-0.5 text-muted-foreground">&times;</span>
            </button>
          ))}
        </div>
      )}

      {/* Chart */}
      {selectedMetrics.length === 0 ? (
        <div className="flex items-center justify-center rounded-lg border border-dashed border-border py-20 text-sm text-muted-foreground">
          Select at least one metric to start comparing
        </div>
      ) : (
        <div className="rounded-lg border border-border bg-card p-4">
          <ChartContainer config={chartConfig} className="h-[400px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              {chartMode === "bar" ? (
                <BarChart
                  data={chartData}
                  margin={{ top: 10, right: 10, left: 10, bottom: 10 }}
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="hsl(240 6% 16%)"
                    vertical={false}
                  />
                  <XAxis
                    dataKey="label"
                    tick={{ fontSize: 12, fill: "hsl(215 12% 55%)" }}
                    axisLine={{ stroke: "hsl(240 6% 16%)" }}
                    tickLine={false}
                  />
                  <YAxis
                    tick={{ fontSize: 11, fill: "hsl(215 12% 55%)" }}
                    axisLine={false}
                    tickLine={false}
                    width={70}
                    tickFormatter={(v: number) => {
                      if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`
                      if (v >= 1_000) return `${(v / 1_000).toFixed(1)}K`
                      return v.toFixed(1)
                    }}
                  />
                  <ChartTooltip
                    content={
                      <ChartTooltipContent
                        className="bg-popover text-popover-foreground border-border"
                      />
                    }
                  />
                  <Legend
                    wrapperStyle={{ fontSize: 11, paddingTop: 12 }}
                  />
                  {selectedMetrics.map((metric, i) => (
                    <Bar
                      key={metric}
                      dataKey={metric}
                      name={METRIC_LABELS[metric] || metric}
                      fill={BAR_COLORS[i % BAR_COLORS.length]}
                      radius={[4, 4, 0, 0]}
                      maxBarSize={60}
                    />
                  ))}
                </BarChart>
              ) : (
                <LineChart
                  data={chartData}
                  margin={{ top: 10, right: 10, left: 10, bottom: 10 }}
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="hsl(240 6% 16%)"
                    vertical={false}
                  />
                  <XAxis
                    dataKey="label"
                    tick={{ fontSize: 12, fill: "hsl(215 12% 55%)" }}
                    axisLine={{ stroke: "hsl(240 6% 16%)" }}
                    tickLine={false}
                  />
                  <YAxis
                    tick={{ fontSize: 11, fill: "hsl(215 12% 55%)" }}
                    axisLine={false}
                    tickLine={false}
                    width={70}
                    tickFormatter={(v: number) => {
                      if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`
                      if (v >= 1_000) return `${(v / 1_000).toFixed(1)}K`
                      return v.toFixed(1)
                    }}
                  />
                  <ChartTooltip
                    content={
                      <ChartTooltipContent
                        className="bg-popover text-popover-foreground border-border"
                      />
                    }
                  />
                  <Legend
                    wrapperStyle={{ fontSize: 11, paddingTop: 12 }}
                  />
                  {selectedMetrics.map((metric, i) => (
                    <Line
                      key={metric}
                      type="monotone"
                      dataKey={metric}
                      name={METRIC_LABELS[metric] || metric}
                      stroke={BAR_COLORS[i % BAR_COLORS.length]}
                      strokeWidth={2}
                      dot={{
                        fill: BAR_COLORS[i % BAR_COLORS.length],
                        r: 4,
                      }}
                      activeDot={{ r: 6 }}
                    />
                  ))}
                </LineChart>
              )}
            </ResponsiveContainer>
          </ChartContainer>
        </div>
      )}
    </div>
  )
}

"use client"

import { Zap, Clock, ArrowUpRight, ArrowDownRight, Gauge, Timer } from "lucide-react"
import type { BenchmarkRecord } from "@/lib/benchmark-data"

interface MetricCardsProps {
  /** Runs currently in view, used for the range each card sits in. */
  data: BenchmarkRecord[]
  /** The run these cards describe — the winner from the block above. */
  winner?: BenchmarkRecord
  /** Metric being optimized, pinned as the first card. */
  objectiveKey: string
}

function formatNumber(value: unknown): string {
  if (typeof value !== "number") return "—"
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`
  if (value >= 1_000) return `${(value / 1_000).toFixed(2)}K`
  if (value % 1 !== 0) return value.toFixed(2)
  return String(value)
}

function range(data: BenchmarkRecord[], key: string): { min: number; max: number } | null {
  const values = data
    .map((entry) => entry.results?.[key])
    .filter((v): v is number => typeof v === "number" && Number.isFinite(v))
  if (values.length === 0) return null
  return { min: Math.min(...values), max: Math.max(...values) }
}

const UNITS: Record<string, string> = {
  request_throughput: "req/s",
  input_throughput: "tok/s",
  output_throughput: "tok/s",
  total_throughput: "tok/s",
  concurrency: "req",
  duration: "s",
}

function unitFor(key: string): string {
  if (UNITS[key]) return UNITS[key]
  if (key.endsWith("_ms")) return "ms"
  if (key.includes("tokens")) return "tokens"
  return ""
}

function labelFor(key: string): string {
  return key.replace(/_/g, " ").replace(/ ms$/, "").replace(/\b\w/g, (c) => c.toUpperCase())
}

/** The metrics worth seeing about the winning run, beyond the objective itself. */
const SUPPORTING_METRICS = [
  { key: "output_throughput", icon: ArrowUpRight, color: "text-success", bg: "bg-success/10" },
  { key: "request_throughput", icon: Zap, color: "text-primary", bg: "bg-primary/10" },
  { key: "mean_ttft_ms", icon: ArrowDownRight, color: "text-chart-2", bg: "bg-chart-2/10" },
  { key: "p99_ttft_ms", icon: Timer, color: "text-chart-2", bg: "bg-chart-2/10" },
  { key: "mean_e2e_latency_ms", icon: Clock, color: "text-warning", bg: "bg-warning/10" },
  { key: "p99_e2e_latency_ms", icon: Clock, color: "text-warning", bg: "bg-warning/10" },
]

/**
 * Metrics for the selected winning run.
 *
 * These used to average every run in view, which for a config sweep is a
 * number nobody wants — the mean throughput across candidate configs describes
 * no config you can actually deploy. They now describe the run chosen above,
 * with the spread across the other runs underneath for context.
 */
export function MetricCards({ data, winner, objectiveKey }: MetricCardsProps) {
  if (!winner) return null

  // Objective first, then the supporting metrics that are not already it.
  const cards = [
    { key: objectiveKey, icon: Gauge, color: "text-primary", bg: "bg-primary/10", isObjective: true },
    ...SUPPORTING_METRICS.filter((m) => m.key !== objectiveKey).map((m) => ({
      ...m,
      isObjective: false,
    })),
  ].slice(0, 6)

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
      {cards.map((card) => {
        const value = winner.results?.[card.key]
        if (typeof value !== "number") return null

        const span = range(data, card.key)
        const Icon = card.icon

        return (
          <div
            key={card.key}
            className={`flex flex-col gap-2 rounded-lg border bg-card p-4 ${
              card.isObjective ? "border-primary/30" : "border-border"
            }`}
          >
            <div className="flex items-center gap-2">
              <div className={`flex h-7 w-7 items-center justify-center rounded-md ${card.bg}`}>
                <Icon className={`h-3.5 w-3.5 ${card.color}`} />
              </div>
              <span className="truncate text-xs text-muted-foreground">
                {labelFor(card.key)}
              </span>
            </div>

            <div>
              <p className={`text-xl font-semibold tracking-tight ${card.color}`}>
                {formatNumber(value)}
              </p>
              <p className="text-[11px] text-muted-foreground">
                {card.isObjective ? "objective" : unitFor(card.key)}
              </p>
            </div>

            {span && data.length > 1 && (
              <div className="border-t border-border pt-2 text-[11px] text-muted-foreground">
                across runs{" "}
                <span className="font-medium text-foreground">
                  {formatNumber(span.min)}–{formatNumber(span.max)}
                </span>
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

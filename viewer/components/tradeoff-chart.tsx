"use client"

import { useMemo } from "react"
import {
  CartesianGrid,
  Cell,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts"
import { Crosshair } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { meetsConstraints, rankableMetrics, type BenchmarkRecord } from "@/lib/benchmark-data"
import type { Budget } from "@/components/winner-card"

interface TradeoffChartProps {
  records: BenchmarkRecord[]
  /** Metric on the Y axis — what you are trying to get more (or less) of. */
  yKey: string
  /** Metric on the X axis — the cost you are paying for it. */
  xKey: string
  onXKeyChange: (key: string) => void
  budgets: Budget[]
  winner?: BenchmarkRecord
  selected?: BenchmarkRecord
  onSelect: (record: BenchmarkRecord) => void
}

interface Point {
  x: number
  y: number
  record: BenchmarkRecord
  passes: boolean
  isWinner: boolean
  isSelected: boolean
  label: string
}

function labelFor(key: string): string {
  return key.replace(/_/g, " ").replace(/ ms$/, " (ms)")
}

function shortConfig(record: BenchmarkRecord): string {
  const s = record.config.server
  return [
    s.tensor_parallel_size !== undefined && `TP${s.tensor_parallel_size}`,
    s.data_parallel_size !== undefined && `DP${s.data_parallel_size}`,
    s.max_num_seqs !== undefined && `seqs${s.max_num_seqs}`,
  ]
    .filter(Boolean)
    .join("/")
}

/**
 * Every run plotted as cost against benefit, with the budget drawn on it.
 *
 * The stock viewer plots the runs but leaves you to work out which point you
 * can actually ship. Here the budget is a shaded region: points outside it are
 * greyed, the best point inside it is ringed, and clicking any point drives
 * the detail cards below — so picking a config is looking and clicking rather
 * than cross-referencing a table.
 */
export function TradeoffChart({
  records,
  yKey,
  xKey,
  onXKeyChange,
  budgets,
  winner,
  selected,
  onSelect,
}: TradeoffChartProps) {
  const metrics = useMemo(() => rankableMetrics(records), [records])

  const points = useMemo<Point[]>(() => {
    return records
      .map((record) => {
        const x = record.results[xKey]
        const y = record.results[yKey]
        if (typeof x !== "number" || typeof y !== "number") return null
        return {
          x,
          y,
          record,
          passes: budgets.length === 0 || meetsConstraints(record, budgets),
          isWinner: record === winner,
          isSelected: record === selected,
          label: shortConfig(record),
        }
      })
      .filter((p): p is Point => p !== null)
  }, [records, xKey, yKey, budgets, winner, selected])

  // Budget lines that apply to the axis actually on screen.
  const xBudgets = budgets.filter((b) => b.metric === xKey)
  const yBudgets = budgets.filter((b) => b.metric === yKey)

  if (points.length === 0) {
    return (
      <div className="rounded-lg border border-border bg-card p-8 text-center text-sm text-muted-foreground">
        No runs have both {labelFor(xKey)} and {labelFor(yKey)}.
      </div>
    )
  }

  const xs = points.map((p) => p.x)
  const ys = points.map((p) => p.y)
  const pad = (values: number[]) => {
    const min = Math.min(...values)
    const max = Math.max(...values)
    const margin = (max - min || Math.abs(max) || 1) * 0.15
    return [min - margin, max + margin] as [number, number]
  }
  const [xMin, xMax] = pad(xs)
  const [yMin, yMax] = pad(ys)

  // Shade the half-plane the budget allows, per budget on the X axis.
  const shaded = xBudgets.map((b) => {
    const allowsLower = b.operator === "<" || b.operator === "<="
    return {
      id: b.id,
      x1: allowsLower ? xMin : b.value,
      x2: allowsLower ? b.value : xMax,
    }
  })

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-md bg-primary/10">
            <Crosshair className="h-3.5 w-3.5 text-primary" />
          </div>
          <span className="text-sm font-medium text-foreground">
            {labelFor(yKey)} vs {labelFor(xKey)}
          </span>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">cost axis</span>
          <select
            value={xKey}
            onChange={(e) => onXKeyChange(e.target.value)}
            className="h-8 rounded-md border border-border bg-secondary px-2 text-xs text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
          >
            {metrics
              .filter((m) => m.key !== yKey)
              .map((m) => (
                <option key={m.key} value={m.key}>
                  {m.label}
                </option>
              ))}
          </select>
        </div>
      </div>

      <div className="mt-2 flex flex-wrap items-center gap-3 text-[11px] text-muted-foreground">
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-full bg-success" /> within budget
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-full bg-muted-foreground/40" /> over
          budget
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-full bg-primary ring-2 ring-primary/30" />{" "}
          best
        </span>
        <span>· click a point to load it below</span>
      </div>

      <div className="mt-3 h-[340px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 12, right: 20, bottom: 34, left: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />

            {shaded.map((s) => (
              <ReferenceArea
                key={s.id}
                x1={s.x1}
                x2={s.x2}
                fill="hsl(var(--success))"
                fillOpacity={0.07}
                ifOverflow="extendDomain"
              />
            ))}
            {xBudgets.map((b) => (
              <ReferenceLine
                key={`xl-${b.id}`}
                x={b.value}
                stroke="hsl(var(--warning))"
                strokeDasharray="4 4"
                label={{
                  value: `${b.operator}${b.value}`,
                  position: "top",
                  fill: "hsl(var(--warning))",
                  fontSize: 11,
                }}
              />
            ))}
            {yBudgets.map((b) => (
              <ReferenceLine
                key={`yl-${b.id}`}
                y={b.value}
                stroke="hsl(var(--warning))"
                strokeDasharray="4 4"
              />
            ))}

            <XAxis
              type="number"
              dataKey="x"
              domain={[xMin, xMax]}
              tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
              stroke="hsl(var(--border))"
              label={{
                value: labelFor(xKey),
                position: "insideBottom",
                offset: -20,
                fill: "hsl(var(--muted-foreground))",
                fontSize: 12,
              }}
            />
            <YAxis
              type="number"
              dataKey="y"
              domain={[yMin, yMax]}
              tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
              stroke="hsl(var(--border))"
              width={70}
              label={{
                value: labelFor(yKey),
                angle: -90,
                position: "insideLeft",
                fill: "hsl(var(--muted-foreground))",
                fontSize: 12,
              }}
            />
            <ZAxis range={[140, 140]} />

            <Scatter
              data={points}
              onClick={(entry: unknown) => {
                const point = entry as { payload?: Point } & Partial<Point>
                const record = point.payload?.record ?? point.record
                if (record) onSelect(record)
              }}
              cursor="pointer"
              isAnimationActive={false}
            >
              {points.map((p, i) => (
                <Cell
                  key={i}
                  fill={
                    p.isWinner
                      ? "hsl(var(--primary))"
                      : p.passes
                        ? "hsl(var(--success))"
                        : "hsl(var(--muted-foreground))"
                  }
                  fillOpacity={p.passes ? 0.95 : 0.35}
                  stroke={p.isSelected ? "hsl(var(--foreground))" : "transparent"}
                  strokeWidth={p.isSelected ? 3 : 0}
                />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* A click target per run as well, since small scatters are fiddly and
          this doubles as the legend of which point is which config. */}
      <div className="mt-2 flex flex-wrap gap-1.5">
        {points.map((p, i) => (
          <button key={i} onClick={() => onSelect(p.record)}>
            <Badge
              variant="outline"
              className={
                p.isSelected
                  ? "cursor-pointer border-foreground/40 bg-foreground/10 font-mono text-[11px] text-foreground"
                  : p.passes
                    ? "cursor-pointer border-success/20 bg-success/10 font-mono text-[11px] text-success hover:bg-success/20"
                    : "cursor-pointer border-border bg-muted font-mono text-[11px] text-muted-foreground hover:bg-muted/70"
              }
            >
              {p.isWinner && "★ "}
              {p.label}
            </Badge>
          </button>
        ))}
      </div>
    </div>
  )
}

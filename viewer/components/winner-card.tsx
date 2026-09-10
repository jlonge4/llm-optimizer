"use client"

import { useMemo, useState } from "react"
import {
  Trophy,
  ArrowUp,
  ArrowDown,
  Copy,
  Check,
  Plus,
  X,
  ShieldCheck,
  ShieldAlert,
} from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  pickWinner,
  rankableMetrics,
  type BenchmarkRecord,
  type Constraint,
} from "@/lib/benchmark-data"

/** A budget the viewer sets here, as opposed to one baked into the run. */
export interface Budget extends Constraint {
  id: string
}

export interface Objective {
  metricKey: string
  direction: "higher" | "lower"
}

interface WinnerCardProps {
  records: BenchmarkRecord[]
  objective: Objective
  onObjectiveChange: (objective: Objective) => void
  budgets: Budget[]
  onBudgetsChange: (budgets: Budget[]) => void
}

const OPERATORS = ["<", "<=", ">", ">="] as const

function formatValue(value: unknown): string {
  if (typeof value !== "number") return String(value ?? "—")
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`
  if (value >= 1_000) return `${(value / 1_000).toFixed(2)}K`
  if (value % 1 !== 0) return value.toFixed(2)
  return String(value)
}

/** The parallelism and batching settings that distinguish one run from another. */
function describeConfig(record: BenchmarkRecord): string {
  const server = record.config.server
  const parts = [
    server.tensor_parallel_size !== undefined && `TP ${server.tensor_parallel_size}`,
    server.data_parallel_size !== undefined && `DP ${server.data_parallel_size}`,
    server.max_num_seqs !== undefined && `seqs ${server.max_num_seqs}`,
    server.max_num_batched_tokens !== undefined && `tokens ${server.max_num_batched_tokens}`,
    server.block_size !== undefined && `block ${server.block_size}`,
  ].filter(Boolean)
  return parts.join(" · ") || "default configuration"
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <Button
      variant="ghost"
      size="sm"
      className="h-7 shrink-0 gap-1.5 px-2 text-xs text-muted-foreground hover:text-foreground"
      onClick={() => {
        navigator.clipboard?.writeText(text)
        setCopied(true)
        setTimeout(() => setCopied(false), 1500)
      }}
    >
      {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
      {copied ? "Copied" : "Copy"}
    </Button>
  )
}

function Winner({
  record,
  metricKey,
  budgets,
  label,
  tone,
}: {
  record: BenchmarkRecord
  metricKey: string
  budgets: Budget[]
  label: string
  tone: "primary" | "success"
}) {
  const accent = tone === "success" ? "text-success" : "text-primary"
  const surface =
    tone === "success" ? "border-success/20 bg-success/5" : "border-primary/20 bg-primary/5"

  return (
    <div className={`rounded-lg border p-4 ${surface}`}>
      <p className="text-[11px] uppercase tracking-wide text-muted-foreground">{label}</p>
      <p className={`mt-1 text-2xl font-semibold tracking-tight ${accent}`}>
        {formatValue(record.results[metricKey])}
      </p>
      <p className="mt-1 font-mono text-xs text-foreground">{describeConfig(record)}</p>

      {/* What this run scored on each budgeted metric, so a near-miss is visible. */}
      {budgets.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {budgets.map((b) => {
            const value = record.results[b.metric]
            const ok =
              typeof value === "number" &&
              (b.operator === "<"
                ? value < b.value
                : b.operator === "<="
                  ? value <= b.value
                  : b.operator === ">"
                    ? value > b.value
                    : value >= b.value)
            return (
              <Badge
                key={b.id}
                variant="outline"
                className={
                  ok
                    ? "border-success/20 bg-success/10 font-mono text-[11px] text-success"
                    : "border-destructive/20 bg-destructive/10 font-mono text-[11px] text-destructive"
                }
              >
                {b.metric.replace(/_ms$/, "")} {formatValue(value)}
                {ok ? " ✓" : ` ✗ (${b.operator}${b.value})`}
              </Badge>
            )
          })}
        </div>
      )}

      {record.cmd && (
        <div className="mt-3 flex items-start gap-2 border-t border-border pt-3">
          <code className="min-w-0 flex-1 break-all font-mono text-[11px] leading-relaxed text-muted-foreground">
            {record.cmd}
          </code>
          <CopyButton text={record.cmd} />
        </div>
      )}
    </div>
  )
}

/**
 * "Give me max throughput inside this latency budget."
 *
 * The objective and the budgets are both set here rather than coming from the
 * benchmark run: which trade-off matters is a question you ask of a finished
 * sweep, and the answer changes without re-running anything. Constraints
 * recorded by the CLI only seed the initial budget.
 */
export function WinnerCard({
  records,
  objective,
  onObjectiveChange,
  budgets,
  onBudgetsChange,
}: WinnerCardProps) {
  const metrics = useMemo(() => rankableMetrics(records), [records])

  const metricKey = objective.metricKey
  const effectiveDirection = objective.direction
  const selected = metrics.find((m) => m.key === metricKey)

  const winner = useMemo(
    () => pickWinner(records, metricKey, effectiveDirection, budgets),
    [records, metricKey, effectiveDirection, budgets]
  )

  if (records.length === 0 || !metricKey) return null

  const setBudgets = (next: Budget[] | ((prev: Budget[]) => Budget[])) =>
    onBudgetsChange(typeof next === "function" ? next(budgets) : next)

  function addBudget() {
    // Default to the tightest thing most people budget on, unless it is
    // already the objective.
    const preferred = metrics.find(
      (m) => m.key === "mean_ttft_ms" && m.key !== metricKey
    )
    const fallback = metrics.find((m) => m.betterWhen === "lower" && m.key !== metricKey)
    const target = preferred ?? fallback ?? metrics[0]

    setBudgets((prev) => [
      ...prev,
      {
        id: `b-${Date.now()}`,
        metric: target.key,
        operator: target.betterWhen === "lower" ? "<" : ">",
        value: 0,
      },
    ])
  }

  function updateBudget(id: string, patch: Partial<Budget>) {
    setBudgets((prev) => prev.map((b) => (b.id === id ? { ...b, ...patch } : b)))
  }

  // With a budget set, the winner is the best run that fits it — a run that
  // blows the budget is not a candidate, so it does not get a card. What the
  // budget cost you is worth one line, not a competing headline.
  const champion = budgets.length > 0 ? winner.withinSlo : winner.overall
  const unconstrained = winner.overall
  const budgetCost =
    budgets.length > 0 &&
    champion &&
    unconstrained &&
    champion !== unconstrained &&
    typeof champion.results[metricKey] === "number" &&
    typeof unconstrained.results[metricKey] === "number"
      ? {
          value: unconstrained.results[metricKey] as number,
          percent:
            Math.abs(
              ((champion.results[metricKey] as number) -
                (unconstrained.results[metricKey] as number)) /
                (unconstrained.results[metricKey] as number)
            ) * 100,
          config: describeConfig(unconstrained),
        }
      : null

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="flex items-center gap-2">
        <div className="flex h-7 w-7 items-center justify-center rounded-md bg-primary/10">
          <Trophy className="h-3.5 w-3.5 text-primary" />
        </div>
        <span className="text-sm font-medium text-foreground">Best configuration</span>
      </div>

      {/* Objective */}
      <div className="mt-3 flex flex-wrap items-center gap-2">
        <Button
          variant="outline"
          size="sm"
          className="h-8 gap-1.5 border-border bg-secondary px-2 text-xs"
          onClick={() =>
            onObjectiveChange({
              metricKey,
              direction: effectiveDirection === "higher" ? "lower" : "higher",
            })
          }
        >
          {effectiveDirection === "higher" ? (
            <>
              <ArrowUp className="h-3.5 w-3.5" /> Maximize
            </>
          ) : (
            <>
              <ArrowDown className="h-3.5 w-3.5" /> Minimize
            </>
          )}
        </Button>
        <select
          value={metricKey}
          onChange={(e) => {
            const next = e.target.value
            // Reset to the new metric's natural direction rather than carrying
            // over one that makes no sense for it.
            const natural =
              metrics.find((m) => m.key === next)?.betterWhen ?? "higher"
            onObjectiveChange({ metricKey: next, direction: natural })
          }}
          className="h-8 rounded-md border border-border bg-secondary px-2 text-xs text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
        >
          {metrics.map((m) => (
            <option key={m.key} value={m.key}>
              {m.label}
            </option>
          ))}
        </select>

        <span className="text-xs text-muted-foreground">subject to</span>

        <Button
          variant="outline"
          size="sm"
          className="h-8 gap-1.5 border-border bg-secondary px-2 text-xs"
          onClick={addBudget}
        >
          <Plus className="h-3.5 w-3.5" /> Add budget
        </Button>
      </div>

      {/* Budgets */}
      {budgets.length > 0 && (
        <div className="mt-3 flex flex-col gap-2">
          {budgets.map((b) => (
            <div key={b.id} className="flex flex-wrap items-center gap-2">
              <select
                value={b.metric}
                onChange={(e) => updateBudget(b.id, { metric: e.target.value })}
                className="h-8 rounded-md border border-border bg-secondary px-2 text-xs text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
              >
                {metrics.map((m) => (
                  <option key={m.key} value={m.key}>
                    {m.label}
                  </option>
                ))}
              </select>
              <select
                value={b.operator}
                onChange={(e) => updateBudget(b.id, { operator: e.target.value })}
                className="h-8 w-16 rounded-md border border-border bg-secondary px-2 text-xs text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
              >
                {OPERATORS.map((op) => (
                  <option key={op} value={op}>
                    {op}
                  </option>
                ))}
              </select>
              <input
                type="number"
                value={b.value}
                onChange={(e) =>
                  updateBudget(b.id, { value: parseFloat(e.target.value) || 0 })
                }
                className="h-8 w-28 rounded-md border border-border bg-secondary px-2 font-mono text-xs text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
              />
              <Button
                variant="ghost"
                size="sm"
                className="h-8 w-8 p-0 text-muted-foreground hover:text-destructive"
                onClick={() => setBudgets((prev) => prev.filter((x) => x.id !== b.id))}
              >
                <X className="h-3.5 w-3.5" />
              </Button>
            </div>
          ))}

          <div className="flex flex-wrap items-center gap-2">
            {winner.passingCount > 0 ? (
              <Badge
                variant="outline"
                className="gap-1 border-success/20 bg-success/10 text-success"
              >
                <ShieldCheck className="h-3 w-3" />
                {winner.passingCount} of {records.length} within budget
              </Badge>
            ) : (
              <Badge
                variant="outline"
                className="gap-1 border-destructive/20 bg-destructive/10 text-destructive"
              >
                <ShieldAlert className="h-3 w-3" />
                No run fits this budget
              </Badge>
            )}
          </div>
        </div>
      )}

      <div className="mt-4">
        {champion ? (
          <Winner
            record={champion}
            metricKey={metricKey}
            budgets={budgets}
            label={budgets.length > 0 ? "Best within budget" : "Best run"}
            tone={budgets.length > 0 ? "success" : "primary"}
          />
        ) : (
          <div className="rounded-lg border border-destructive/20 bg-destructive/5 p-4">
            <p className="text-sm font-medium text-destructive">
              <ShieldAlert className="mr-1.5 inline h-4 w-4" />
              No run fits this budget.
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              Loosen a budget above, or sweep more configurations. The closest run
              is {unconstrained ? describeConfig(unconstrained) : "unavailable"}.
            </p>
          </div>
        )}

        {budgetCost && (
          <p className="mt-2 text-xs text-muted-foreground">
            The budget costs{" "}
            <span className="font-medium text-foreground">
              {budgetCost.percent.toFixed(1)}%
            </span>{" "}
            of {selected?.label ?? metricKey} — without it,{" "}
            <span className="font-mono text-foreground">{budgetCost.config}</span> reaches{" "}
            <span className="font-medium text-foreground">
              {formatValue(budgetCost.value)}
            </span>
            .
          </p>
        )}
      </div>
    </div>
  )
}

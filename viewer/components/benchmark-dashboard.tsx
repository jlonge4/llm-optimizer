"use client"

import { useState, useMemo, useEffect } from "react"
import { Activity } from "lucide-react"
import { JsonInput } from "@/components/json-input"
import { MetricCards } from "@/components/metric-cards"
import { FilterBar, applyFilters, type FilterRule } from "@/components/filter-bar"
import { ResultsTable } from "@/components/results-table"
import { ConfigPanel } from "@/components/config-panel"
import { ComparisonChart } from "@/components/comparison-chart"
import { RunMetadata } from "@/components/run-metadata"
import { WinnerCard, type Budget, type Objective } from "@/components/winner-card"
import { TradeoffChart } from "@/components/tradeoff-chart"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  parseBenchmarkInput,
  pickWinner,
  rankableMetrics,
  type BenchmarkMetadata,
  type BenchmarkRecord,
  type Constraint,
} from "@/lib/benchmark-data"

export function BenchmarkDashboard() {
  const [rawData, setRawData] = useState<BenchmarkRecord[]>([])
  const [metadata, setMetadata] = useState<BenchmarkMetadata>({})
  const [constraints, setConstraints] = useState<Constraint[]>([])
  const [filters, setFilters] = useState<FilterRule[]>([])

  // What "best" means here, and the budget it has to fit inside. Owned by the
  // dashboard so the chart and the detail cards key off the same choice.
  const [objective, setObjective] = useState<Objective>({
    metricKey: "output_throughput",
    direction: "higher",
  })
  const [budgets, setBudgets] = useState<Budget[]>([])
  const [xKey, setXKey] = useState("mean_ttft_ms")
  // The run the detail cards describe. Null means "follow the winner".
  const [pinned, setPinned] = useState<BenchmarkRecord | null>(null)

  const filteredData = useMemo(
    () => applyFilters(rawData as Record<string, unknown>[], filters) as BenchmarkRecord[],
    [rawData, filters]
  )

  const winner = useMemo(
    () => pickWinner(filteredData, objective.metricKey, objective.direction, budgets),
    [filteredData, objective, budgets]
  )

  // Prefer the run that fits the budget: that is the one you would deploy.
  const best = winner.withinSlo ?? winner.overall
  // A pinned run that filtering removed should fall back to the winner.
  const selected = pinned && filteredData.includes(pinned) ? pinned : best

  const hasData = rawData.length > 0

  function handleDataLoaded(parsed: {
    records: BenchmarkRecord[]
    metadata: BenchmarkMetadata
    constraints: Constraint[]
  }) {
    setRawData(parsed.records)
    setMetadata(parsed.metadata)
    setConstraints(parsed.constraints)
    setFilters([])
    setPinned(null)

    // Seed the budget from the constraints the run was benchmarked against;
    // they stay editable from here.
    setBudgets(parsed.constraints.map((c, i) => ({ ...c, id: `seed-${i}-${c.metric}` })))

    // Fall back if this data has no throughput metric to optimize.
    const metrics = rankableMetrics(parsed.records)
    if (!metrics.some((m) => m.key === objective.metricKey) && metrics[0]) {
      setObjective({ metricKey: metrics[0].key, direction: metrics[0].betterWhen })
    }
    if (!metrics.some((m) => m.key === xKey)) {
      const cost = metrics.find((m) => m.betterWhen === "lower")
      if (cost) setXKey(cost.key)
    }
  }

  function handleClear() {
    setRawData([])
    setMetadata({})
    setConstraints([])
    setFilters([])
    setBudgets([])
    setPinned(null)
  }

  // ?data=/path.jsonl loads a results file straight from this origin, so a run
  // can be shared as a link instead of a file to drag in. Restricted to
  // same-origin paths: this should not become a fetcher for arbitrary URLs.
  useEffect(() => {
    const path = new URLSearchParams(window.location.search).get("data")
    if (!path || !path.startsWith("/")) return

    let cancelled = false
    fetch(path)
      .then((res) => (res.ok ? res.text() : Promise.reject(new Error(String(res.status)))))
      .then((text) => {
        if (!cancelled) handleDataLoaded(parseBenchmarkInput(text))
      })
      .catch(() => {
        /* Bad path or malformed file: fall back to the normal empty state. */
      })
    return () => {
      cancelled = true
    }
  }, [])

  return (
    <div className="flex min-h-screen flex-col bg-background">
      {/* Header */}
      <header className="sticky top-0 z-40 border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/10">
              <Activity className="h-5 w-5 text-primary" />
            </div>
            <div>
              <h1 className="text-base font-semibold tracking-tight text-foreground sm:text-lg">
                vLLM Benchmark Viewer
              </h1>
              <p className="hidden text-xs text-muted-foreground sm:block">
                Visualize and compare inference benchmark results
              </p>
            </div>
          </div>
          <JsonInput
            onDataLoaded={handleDataLoaded}
            hasData={hasData}
            onClear={handleClear}
          />
        </div>
      </header>

      <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6">
        {!hasData ? (
          <div className="flex flex-col items-center justify-center gap-8 py-20">
            <div className="flex flex-col items-center gap-4 text-center">
              <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10">
                <Activity className="h-8 w-8 text-primary" />
              </div>
              <div>
                <h2 className="text-xl font-semibold text-foreground sm:text-2xl">
                  Load Benchmark Results
                </h2>
                <p className="mt-1 text-sm text-muted-foreground">
                  Paste llm-optimizer results (.jsonl or .json) to get started
                </p>
              </div>
            </div>
            <div className="w-full max-w-2xl">
              <JsonInput
                onDataLoaded={handleDataLoaded}
                hasData={false}
                onClear={handleClear}
              />
            </div>
          </div>
        ) : (
          <div className="flex flex-col gap-6">
            {/* Run metadata: accelerator, model, framework, SLO constraints */}
            <RunMetadata
              metadata={metadata}
              constraints={constraints}
              records={rawData}
            />

            {/* Objective + budget: what "best" means for this deployment */}
            <WinnerCard
              records={filteredData}
              objective={objective}
              onObjectiveChange={setObjective}
              budgets={budgets}
              onBudgetsChange={setBudgets}
            />

            {/* Every run as cost vs benefit; click a point to inspect it */}
            <TradeoffChart
              records={filteredData}
              yKey={objective.metricKey}
              xKey={xKey}
              onXKeyChange={setXKey}
              budgets={budgets}
              winner={best}
              selected={selected}
              onSelect={setPinned}
            />

            {/* Details for the selected run */}
            <MetricCards
              data={filteredData}
              winner={selected}
              objectiveKey={objective.metricKey}
            />

            {/* Filter Bar */}
            <FilterBar
              data={rawData as Record<string, unknown>[]}
              filters={filters}
              onFiltersChange={setFilters}
            />

            {/* Filtered count */}
            {filters.length > 0 && (
              <p className="text-xs text-muted-foreground">
                Showing{" "}
                <span className="font-medium text-foreground">
                  {filteredData.length}
                </span>{" "}
                of{" "}
                <span className="font-medium text-foreground">
                  {rawData.length}
                </span>{" "}
                results
              </p>
            )}

            {/* Tabs: Table / Config */}
            <Tabs defaultValue="table">
              <TabsList className="bg-secondary">
                <TabsTrigger
                  value="table"
                  className="text-secondary-foreground data-[state=active]:bg-card data-[state=active]:text-foreground"
                >
                  Results Table
                </TabsTrigger>
                <TabsTrigger
                  value="compare"
                  className="text-secondary-foreground data-[state=active]:bg-card data-[state=active]:text-foreground"
                >
                  Compare
                </TabsTrigger>
                <TabsTrigger
                  value="config"
                  className="text-secondary-foreground data-[state=active]:bg-card data-[state=active]:text-foreground"
                >
                  Config Overview
                </TabsTrigger>
              </TabsList>
              <TabsContent value="table">
                <ResultsTable data={filteredData} />
              </TabsContent>
              <TabsContent value="compare">
                <ComparisonChart data={filteredData} />
              </TabsContent>
              <TabsContent value="config">
                <ConfigPanel data={filteredData} />
              </TabsContent>
            </Tabs>
          </div>
        )}
      </main>
    </div>
  )
}

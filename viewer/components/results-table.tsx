"use client"

import { useState, useMemo, Fragment } from "react"
import { ArrowUpDown, ArrowUp, ArrowDown, ChevronDown, ChevronRight, Terminal } from "lucide-react"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"

interface ResultsTableProps {
  data: Record<string, unknown>[]
}

function flattenObject(
  obj: Record<string, unknown>,
  prefix = ""
): Record<string, unknown> {
  const result: Record<string, unknown> = {}
  for (const key of Object.keys(obj)) {
    const fullKey = prefix ? `${prefix}.${key}` : key
    const val = obj[key]
    if (val && typeof val === "object" && !Array.isArray(val)) {
      Object.assign(result, flattenObject(val as Record<string, unknown>, fullKey))
    } else {
      result[fullKey] = val
    }
  }
  return result
}

function formatValue(value: unknown): string {
  if (typeof value === "number") {
    if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`
    if (value >= 10_000) return `${(value / 1_000).toFixed(2)}K`
    if (value % 1 !== 0) return value.toFixed(4)
    return String(value)
  }
  if (Array.isArray(value)) return value.join(", ")
  return String(value ?? "")
}

function getColumnCategory(key: string): string {
  if (key.startsWith("config.client")) return "client"
  if (key.startsWith("config.neuron")) return "neuron"
  if (key.startsWith("config.server")) return "server"
  if (key.startsWith("results.")) return "results"
  if (key.startsWith("metadata.")) return "metadata"
  if (key === "cmd") return "cmd"
  return "other"
}

function getCategoryColor(cat: string): string {
  switch (cat) {
    case "client":
      return "bg-primary/10 text-primary border-primary/20"
    case "server":
      return "bg-chart-2/10 text-chart-2 border-chart-2/20"
    case "neuron":
      return "bg-chart-3/10 text-chart-3 border-chart-3/20"
    case "results":
      return "bg-warning/10 text-warning border-warning/20"
    default:
      return "bg-muted text-muted-foreground border-border"
  }
}

// Columns to show by default for a clean initial view
const DEFAULT_VISIBLE_COLUMNS = new Set([
  "config.client.max_concurrency",
  "config.server.tensor_parallel_size",
  "config.server.data_parallel_size",
  "config.server.max_num_batched_tokens",
  "config.server.max_num_seqs",
  // Neuron-only; absent on NVIDIA runs, where they simply never appear.
  "config.server.block_size",
  "config.neuron.prefill_buckets",
  "config.neuron.decode_buckets",
  "results.request_throughput",
  "results.output_throughput",
  // Tail latency matters more than the mean for serving, so p95/p99 are on by
  // default. std/median and the rest are in the column picker.
  "results.mean_e2e_latency_ms",
  "results.p95_e2e_latency_ms",
  "results.p99_e2e_latency_ms",
  "results.mean_ttft_ms",
  "results.p95_ttft_ms",
  "results.p99_ttft_ms",
  "results.mean_itl_ms",
  "results.p95_itl_ms",
  "results.p99_itl_ms",
])

export function ResultsTable({ data }: ResultsTableProps) {
  const [sortKey, setSortKey] = useState<string | null>(null)
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc")
  const [expandedRows, setExpandedRows] = useState<Set<number>>(new Set())
  const [visibleColumns, setVisibleColumns] = useState<Set<string>>(DEFAULT_VISIBLE_COLUMNS)
  const [columnPickerOpen, setColumnPickerOpen] = useState(false)

  const allColumns = useMemo(() => {
    if (data.length === 0) return []
    const allKeys = new Set<string>()
    for (const entry of data) {
      const flat = flattenObject(entry)
      Object.keys(flat).forEach((k) => {
        const isBucketList =
          k === "config.neuron.num_batched_tokens_buckets" ||
          k === "config.neuron.num_seqs_buckets"
        if (
          k !== "cmd" &&
          !k.startsWith("config.server_args") &&
          !k.startsWith("metadata.") &&
          !isBucketList
        ) {
          allKeys.add(k)
        }
      })
    }
    return Array.from(allKeys).sort((a, b) => {
      const catA = getColumnCategory(a)
      const catB = getColumnCategory(b)
      if (catA !== catB) return catA.localeCompare(catB)
      return a.localeCompare(b)
    })
  }, [data])

  const displayColumns = useMemo(() => {
    return allColumns.filter((c) => visibleColumns.has(c))
  }, [allColumns, visibleColumns])

  const flatData = useMemo(() => {
    return data.map((entry) => flattenObject(entry))
  }, [data])

  const sortedData = useMemo(() => {
    if (!sortKey) return flatData
    return [...flatData].sort((a, b) => {
      const valA = a[sortKey]
      const valB = b[sortKey]
      if (typeof valA === "number" && typeof valB === "number") {
        return sortDir === "asc" ? valA - valB : valB - valA
      }
      const strA = String(valA ?? "")
      const strB = String(valB ?? "")
      return sortDir === "asc"
        ? strA.localeCompare(strB)
        : strB.localeCompare(strA)
    })
  }, [flatData, sortKey, sortDir])

  function handleSort(key: string) {
    if (sortKey === key) {
      setSortDir(sortDir === "asc" ? "desc" : "asc")
    } else {
      setSortKey(key)
      setSortDir("asc")
    }
  }

  function toggleRow(index: number) {
    setExpandedRows((prev) => {
      const next = new Set(prev)
      if (next.has(index)) {
        next.delete(index)
      } else {
        next.add(index)
      }
      return next
    })
  }

  function toggleColumn(col: string) {
    setVisibleColumns((prev) => {
      const next = new Set(prev)
      if (next.has(col)) {
        next.delete(col)
      } else {
        next.add(col)
      }
      return next
    })
  }

  function shortLabel(key: string): string {
    const parts = key.split(".")
    return parts[parts.length - 1]
  }

  return (
    <div className="flex flex-col gap-3">
      {/* Column picker */}
      <div className="relative">
        <button
          onClick={() => setColumnPickerOpen(!columnPickerOpen)}
          className="flex items-center gap-2 rounded-md border border-border bg-secondary px-3 py-2 text-xs text-secondary-foreground hover:bg-muted"
        >
          Columns ({displayColumns.length}/{allColumns.length})
          <ChevronDown className="h-3 w-3" />
        </button>
        {columnPickerOpen && (
          <div className="absolute left-0 top-10 z-50 max-h-64 w-80 overflow-y-auto rounded-lg border border-border bg-popover p-2 shadow-xl sm:w-96">
            <div className="mb-2 flex items-center justify-between px-2">
              <span className="text-xs font-medium text-foreground">
                Toggle Columns
              </span>
              <div className="flex gap-2">
                <button
                  className="text-xs text-primary hover:underline"
                  onClick={() => setVisibleColumns(new Set(allColumns))}
                >
                  Show All
                </button>
                <button
                  className="text-xs text-muted-foreground hover:underline"
                  onClick={() => setVisibleColumns(new Set(DEFAULT_VISIBLE_COLUMNS))}
                >
                  Reset
                </button>
              </div>
            </div>
            <div className="grid grid-cols-1 gap-0.5 sm:grid-cols-2">
              {allColumns.map((col) => {
                const cat = getColumnCategory(col)
                return (
                  <label
                    key={col}
                    className="flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 hover:bg-muted"
                  >
                    <input
                      type="checkbox"
                      checked={visibleColumns.has(col)}
                      onChange={() => toggleColumn(col)}
                      className="h-3.5 w-3.5 rounded border-border accent-primary"
                    />
                    <Badge
                      variant="outline"
                      className={`text-[10px] ${getCategoryColor(cat)}`}
                    >
                      {cat}
                    </Badge>
                    <span className="truncate font-mono text-xs text-foreground">
                      {shortLabel(col)}
                    </span>
                  </label>
                )
              })}
            </div>
            <div className="mt-2 border-t border-border pt-2">
              <button
                onClick={() => setColumnPickerOpen(false)}
                className="w-full rounded-md bg-secondary py-1.5 text-xs text-secondary-foreground hover:bg-muted"
              >
                Done
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Table */}
      <div className="overflow-x-auto rounded-lg border border-border">
        <Table>
          <TableHeader>
            <TableRow className="border-border hover:bg-transparent">
              <TableHead className="w-10 bg-card text-muted-foreground" />
              {displayColumns.map((col) => {
                const cat = getColumnCategory(col)
                return (
                  <TableHead
                    key={col}
                    className="cursor-pointer select-none whitespace-nowrap bg-card text-xs text-muted-foreground hover:text-foreground"
                    onClick={() => handleSort(col)}
                  >
                    <div className="flex items-center gap-1">
                      <Badge
                        variant="outline"
                        className={`mr-1 text-[9px] ${getCategoryColor(cat)}`}
                      >
                        {cat === "client" ? "C" : cat === "server" ? "S" : "R"}
                      </Badge>
                      <span>{shortLabel(col)}</span>
                      {sortKey === col ? (
                        sortDir === "asc" ? (
                          <ArrowUp className="h-3 w-3" />
                        ) : (
                          <ArrowDown className="h-3 w-3" />
                        )
                      ) : (
                        <ArrowUpDown className="h-3 w-3 opacity-30" />
                      )}
                    </div>
                  </TableHead>
                )
              })}
            </TableRow>
          </TableHeader>
          <TableBody>
            {sortedData.map((row, idx) => (
              <Fragment key={idx}>
                <TableRow
                  className="border-border hover:bg-muted/30"
                >
                  <TableCell className="w-10 p-2">
                    <button
                      onClick={() => toggleRow(idx)}
                      className="flex h-7 w-7 items-center justify-center rounded-md hover:bg-muted"
                      aria-label={expandedRows.has(idx) ? "Collapse row" : "Expand row"}
                    >
                      {expandedRows.has(idx) ? (
                        <ChevronDown className="h-4 w-4 text-muted-foreground" />
                      ) : (
                        <ChevronRight className="h-4 w-4 text-muted-foreground" />
                      )}
                    </button>
                  </TableCell>
                  {displayColumns.map((col) => (
                    <TableCell
                      key={col}
                      className="whitespace-nowrap font-mono text-xs"
                    >
                      {formatValue(row[col])}
                    </TableCell>
                  ))}
                </TableRow>
                {expandedRows.has(idx) && (
                  <TableRow key={`${idx}-expanded`} className="border-border">
                    <TableCell
                      colSpan={displayColumns.length + 1}
                      className="bg-secondary/50 p-4"
                    >
                      <div className="flex items-start gap-2">
                        <Terminal className="mt-0.5 h-4 w-4 flex-shrink-0 text-primary" />
                        <div className="flex flex-col gap-1">
                          <span className="text-xs font-medium text-muted-foreground">
                            Command
                          </span>
                          <code className="break-all font-mono text-xs text-foreground">
                            {String(row["cmd"] ?? "N/A")}
                          </code>
                        </div>
                      </div>
                      {/* Show all hidden columns */}
                      {allColumns.filter((c) => !visibleColumns.has(c)).length > 0 && (
                        <div className="mt-4">
                          <span className="text-xs font-medium text-muted-foreground">
                            All Fields
                          </span>
                          <div className="mt-2 grid grid-cols-1 gap-x-6 gap-y-1 sm:grid-cols-2 lg:grid-cols-3">
                            {allColumns.map((col) => (
                              <div key={col} className="flex items-baseline gap-2 py-0.5">
                                <span className="text-[11px] text-muted-foreground">
                                  {shortLabel(col)}
                                </span>
                                <span className="font-mono text-xs text-foreground">
                                  {formatValue(row[col])}
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </TableCell>
                  </TableRow>
                )}
              </Fragment>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  )
}

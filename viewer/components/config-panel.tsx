"use client"

import { Settings2, Server, Monitor } from "lucide-react"

interface ConfigPanelProps {
  data: Record<string, unknown>[]
}

function getUniqueConfigValues(
  data: Record<string, unknown>[],
  path: string[]
): unknown[] {
  const values = new Set<string>()
  for (const entry of data) {
    let current: unknown = entry
    for (const key of path) {
      if (current && typeof current === "object") {
        current = (current as Record<string, unknown>)[key]
      } else {
        current = undefined
        break
      }
    }
    if (current !== undefined) {
      values.add(JSON.stringify(current))
    }
  }
  return Array.from(values).map((v) => JSON.parse(v))
}

export function ConfigPanel({ data }: ConfigPanelProps) {
  if (data.length === 0) return null

  const clientConfig = (data[0] as Record<string, unknown>)?.config as
    | Record<string, unknown>
    | undefined
  const clientKeys = clientConfig
    ? Object.keys(
        (clientConfig.client as Record<string, unknown>) ?? {}
      )
    : []
  const serverKeys = clientConfig
    ? Object.keys(
        (clientConfig.server as Record<string, unknown>) ?? {}
      )
    : []

  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
      {/* Client Config */}
      <div className="rounded-lg border border-border bg-card p-4">
        <div className="mb-3 flex items-center gap-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-md bg-primary/10">
            <Monitor className="h-3.5 w-3.5 text-primary" />
          </div>
          <h3 className="text-sm font-medium text-foreground">
            Client Config
          </h3>
        </div>
        <div className="flex flex-col gap-2">
          {clientKeys.map((key) => {
            const values = getUniqueConfigValues(data, [
              "config",
              "client",
              key,
            ])
            return (
              <div
                key={key}
                className="flex items-center justify-between rounded-md bg-secondary px-3 py-2"
              >
                <span className="font-mono text-xs text-muted-foreground">
                  {key}
                </span>
                <span className="font-mono text-xs font-medium text-foreground">
                  {values.length === 1
                    ? String(values[0])
                    : values.map(String).join(", ")}
                </span>
              </div>
            )
          })}
        </div>
      </div>

      {/* Server Config */}
      <div className="rounded-lg border border-border bg-card p-4">
        <div className="mb-3 flex items-center gap-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-md bg-chart-2/10">
            <Server className="h-3.5 w-3.5 text-chart-2" />
          </div>
          <h3 className="text-sm font-medium text-foreground">
            Server Config
          </h3>
        </div>
        <div className="flex flex-col gap-2">
          {serverKeys.map((key) => {
            const values = getUniqueConfigValues(data, [
              "config",
              "server",
              key,
            ])
            return (
              <div
                key={key}
                className="flex items-center justify-between rounded-md bg-secondary px-3 py-2"
              >
                <span className="font-mono text-xs text-muted-foreground">
                  {key}
                </span>
                <span className="font-mono text-xs font-medium text-foreground">
                  {values.length === 1
                    ? String(values[0])
                    : values.map(String).join(", ")}
                </span>
              </div>
            )
          })}
        </div>
      </div>

      {/* Run Count */}
      <div className="rounded-lg border border-border bg-card p-4 sm:col-span-2">
        <div className="flex items-center gap-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-md bg-muted">
            <Settings2 className="h-3.5 w-3.5 text-muted-foreground" />
          </div>
          <div>
            <h3 className="text-sm font-medium text-foreground">
              {data.length} Benchmark Run{data.length > 1 ? "s" : ""} Loaded
            </h3>
            <p className="text-xs text-muted-foreground">
              {data.length > 1
                ? "Showing aggregated metrics across all runs. Use filters to narrow down."
                : "Single benchmark run loaded. Paste more results to compare."}
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

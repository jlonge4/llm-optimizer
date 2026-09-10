"use client"

import { Cpu, Box, Layers, Ruler, Target } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import {
  isNeuronRun,
  type BenchmarkMetadata,
  type BenchmarkRecord,
  type Constraint,
} from "@/lib/benchmark-data"

interface RunMetadataProps {
  metadata: BenchmarkMetadata
  constraints: Constraint[]
  records: BenchmarkRecord[]
}

function Item({
  icon: Icon,
  label,
  value,
}: {
  icon: typeof Cpu
  label: string
  value: string
}) {
  return (
    <div className="flex items-center gap-2">
      <Icon className="h-4 w-4 shrink-0 text-muted-foreground" />
      <div className="min-w-0">
        <p className="text-[11px] uppercase tracking-wide text-muted-foreground">
          {label}
        </p>
        <p className="truncate text-sm font-medium text-foreground">{value}</p>
      </div>
    </div>
  )
}

/**
 * The run-level header: which accelerator, which serving stack, what shape of
 * request. llm-optimizer records this per result, and without it a Trainium
 * sweep is indistinguishable from an H100 one in the table.
 */
export function RunMetadata({ metadata, constraints, records }: RunMetadataProps) {
  const hasAny =
    metadata.gpu_type ||
    metadata.model_tag ||
    metadata.framework ||
    constraints.length > 0

  if (!hasAny) return null

  const neuron = isNeuronRun(metadata, records)

  const accelerator = metadata.gpu_type
    ? metadata.gpu_count
      ? `${metadata.gpu_count} × ${metadata.gpu_type}`
      : metadata.gpu_type
    : "unknown"

  const tokens =
    metadata.input_tokens !== undefined && metadata.output_tokens !== undefined
      ? `${metadata.input_tokens} in / ${metadata.output_tokens} out`
      : undefined

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
        <Item
          icon={Cpu}
          label={neuron ? "Neuron devices" : "Accelerator"}
          value={accelerator}
        />
        {metadata.model_tag && (
          <Item icon={Box} label="Model" value={metadata.model_tag} />
        )}
        {metadata.framework && (
          <Item icon={Layers} label="Framework" value={metadata.framework} />
        )}
        {tokens && <Item icon={Ruler} label="Tokens" value={tokens} />}
        {metadata.total_tests !== undefined && (
          <Item icon={Target} label="Runs" value={String(metadata.total_tests)} />
        )}
      </div>

      {(neuron || constraints.length > 0) && (
        <div className="mt-4 flex flex-wrap items-center gap-2 border-t border-border pt-3">
          {neuron && (
            <Badge
              variant="outline"
              className="border-chart-2/20 bg-chart-2/10 text-chart-2"
            >
              vLLM Neuron plugin
            </Badge>
          )}
          {constraints.map((c) => (
            <Badge
              key={`${c.metric}${c.operator}${c.value}`}
              variant="outline"
              className="border-warning/20 bg-warning/10 font-mono text-xs text-warning"
            >
              {c.metric} {c.operator} {c.value}
            </Badge>
          ))}
        </div>
      )}
    </div>
  )
}

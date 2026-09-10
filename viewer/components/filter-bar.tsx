"use client"

import { useState, useMemo } from "react"
import { Filter, X, Plus, ChevronDown } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"

export interface FilterRule {
  id: string
  field: string
  operator: "eq" | "neq" | "gt" | "gte" | "lt" | "lte" | "contains"
  value: string
}

interface FilterBarProps {
  data: Record<string, unknown>[]
  filters: FilterRule[]
  onFiltersChange: (filters: FilterRule[]) => void
}

function flattenKeys(
  obj: Record<string, unknown>,
  prefix = ""
): string[] {
  const keys: string[] = []
  for (const key of Object.keys(obj)) {
    const fullKey = prefix ? `${prefix}.${key}` : key
    const val = obj[key]
    if (
      val &&
      typeof val === "object" &&
      !Array.isArray(val)
    ) {
      keys.push(...flattenKeys(val as Record<string, unknown>, fullKey))
    } else {
      keys.push(fullKey)
    }
  }
  return keys
}

function getNestedValue(
  obj: Record<string, unknown>,
  path: string
): unknown {
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

const OPERATORS = [
  { value: "eq", label: "=" },
  { value: "neq", label: "!=" },
  { value: "gt", label: ">" },
  { value: "gte", label: ">=" },
  { value: "lt", label: "<" },
  { value: "lte", label: "<=" },
  { value: "contains", label: "contains" },
] as const

export function applyFilters(
  data: Record<string, unknown>[],
  filters: FilterRule[]
): Record<string, unknown>[] {
  if (filters.length === 0) return data
  return data.filter((entry) => {
    return filters.every((filter) => {
      const raw = getNestedValue(entry, filter.field)
      if (raw === undefined) return false

      const numericValue =
        typeof raw === "number" ? raw : parseFloat(String(raw))
      const filterNum = parseFloat(filter.value)
      const isNumeric = !isNaN(numericValue) && !isNaN(filterNum)

      switch (filter.operator) {
        case "eq":
          return isNumeric
            ? numericValue === filterNum
            : String(raw) === filter.value
        case "neq":
          return isNumeric
            ? numericValue !== filterNum
            : String(raw) !== filter.value
        case "gt":
          return isNumeric ? numericValue > filterNum : false
        case "gte":
          return isNumeric ? numericValue >= filterNum : false
        case "lt":
          return isNumeric ? numericValue < filterNum : false
        case "lte":
          return isNumeric ? numericValue <= filterNum : false
        case "contains":
          return String(raw)
            .toLowerCase()
            .includes(filter.value.toLowerCase())
        default:
          return true
      }
    })
  })
}

export function FilterBar({ data, filters, onFiltersChange }: FilterBarProps) {
  const [isAdding, setIsAdding] = useState(false)
  const [newField, setNewField] = useState("")
  const [newOperator, setNewOperator] = useState<FilterRule["operator"]>("gt")
  const [newValue, setNewValue] = useState("")
  const [fieldDropdownOpen, setFieldDropdownOpen] = useState(false)
  const [operatorDropdownOpen, setOperatorDropdownOpen] = useState(false)

  const availableFields = useMemo(() => {
    if (data.length === 0) return []
    const allKeys = new Set<string>()
    for (const entry of data) {
      flattenKeys(entry).forEach((k) => allKeys.add(k))
    }
    return Array.from(allKeys).sort()
  }, [data])

  const filteredFieldOptions = useMemo(() => {
    if (!newField) return availableFields
    return availableFields.filter((f) =>
      f.toLowerCase().includes(newField.toLowerCase())
    )
  }, [availableFields, newField])

  function addFilter() {
    if (!newField || !newValue) return
    const filter: FilterRule = {
      id: crypto.randomUUID(),
      field: newField,
      operator: newOperator,
      value: newValue,
    }
    onFiltersChange([...filters, filter])
    setNewField("")
    setNewValue("")
    setNewOperator("gt")
    setIsAdding(false)
  }

  function removeFilter(id: string) {
    onFiltersChange(filters.filter((f) => f.id !== id))
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Filter className="h-4 w-4" />
          <span>Filters</span>
        </div>

        {filters.map((filter) => (
          <Badge
            key={filter.id}
            variant="secondary"
            className="gap-1.5 bg-secondary text-secondary-foreground"
          >
            <span className="font-mono text-xs text-primary">
              {filter.field}
            </span>
            <span className="text-muted-foreground">
              {OPERATORS.find((o) => o.value === filter.operator)?.label}
            </span>
            <span className="font-mono text-xs">{filter.value}</span>
            <button
              onClick={() => removeFilter(filter.id)}
              className="ml-0.5 rounded-sm p-0.5 hover:bg-muted"
              aria-label={`Remove filter ${filter.field}`}
            >
              <X className="h-3 w-3" />
            </button>
          </Badge>
        ))}

        {!isAdding && (
          <Button
            variant="outline"
            size="sm"
            onClick={() => setIsAdding(true)}
            className="h-7 gap-1 border-dashed border-border bg-transparent text-xs text-muted-foreground hover:bg-secondary hover:text-foreground"
          >
            <Plus className="h-3 w-3" />
            Add Filter
          </Button>
        )}
      </div>

      {isAdding && (
        <div className="flex flex-col gap-2 rounded-lg border border-border bg-card p-3 sm:flex-row sm:items-end">
          {/* Field selector */}
          <div className="relative flex-1">
            <label className="mb-1 block text-xs text-muted-foreground">
              Field
            </label>
            <div className="relative">
              <input
                type="text"
                placeholder="Search fields..."
                value={newField}
                onChange={(e) => {
                  setNewField(e.target.value)
                  setFieldDropdownOpen(true)
                }}
                onFocus={() => setFieldDropdownOpen(true)}
                className="h-9 w-full rounded-md border border-border bg-secondary px-3 font-mono text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
              />
              <ChevronDown className="pointer-events-none absolute right-2 top-2.5 h-4 w-4 text-muted-foreground" />
              {fieldDropdownOpen && filteredFieldOptions.length > 0 && (
                <div className="absolute left-0 top-10 z-50 max-h-48 w-full overflow-y-auto rounded-md border border-border bg-popover shadow-lg">
                  {filteredFieldOptions.map((field) => (
                    <button
                      key={field}
                      className="w-full px-3 py-2 text-left font-mono text-xs text-popover-foreground hover:bg-muted"
                      onClick={() => {
                        setNewField(field)
                        setFieldDropdownOpen(false)
                      }}
                    >
                      {field}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Operator selector */}
          <div className="relative w-full sm:w-28">
            <label className="mb-1 block text-xs text-muted-foreground">
              Operator
            </label>
            <div className="relative">
              <button
                onClick={() => setOperatorDropdownOpen(!operatorDropdownOpen)}
                className="flex h-9 w-full items-center justify-between rounded-md border border-border bg-secondary px-3 text-sm text-foreground"
              >
                {OPERATORS.find((o) => o.value === newOperator)?.label}
                <ChevronDown className="h-4 w-4 text-muted-foreground" />
              </button>
              {operatorDropdownOpen && (
                <div className="absolute left-0 top-10 z-50 w-full rounded-md border border-border bg-popover shadow-lg">
                  {OPERATORS.map((op) => (
                    <button
                      key={op.value}
                      className="w-full px-3 py-2 text-left text-sm text-popover-foreground hover:bg-muted"
                      onClick={() => {
                        setNewOperator(op.value)
                        setOperatorDropdownOpen(false)
                      }}
                    >
                      {op.label}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Value input */}
          <div className="flex-1">
            <label className="mb-1 block text-xs text-muted-foreground">
              Value
            </label>
            <input
              type="text"
              placeholder="Value..."
              value={newValue}
              onChange={(e) => setNewValue(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && addFilter()}
              className="h-9 w-full rounded-md border border-border bg-secondary px-3 font-mono text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
          </div>

          {/* Actions */}
          <div className="flex gap-2">
            <Button
              size="sm"
              onClick={addFilter}
              disabled={!newField || !newValue}
              className="h-9 bg-primary text-primary-foreground hover:bg-primary/90"
            >
              Apply
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                setIsAdding(false)
                setNewField("")
                setNewValue("")
                setFieldDropdownOpen(false)
                setOperatorDropdownOpen(false)
              }}
              className="h-9 border-border bg-transparent text-foreground hover:bg-secondary"
            >
              Cancel
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}

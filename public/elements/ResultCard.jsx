// Result card of a SQL answer: KPI tiles, chart, sortable table, SQL, CSV export.
// Props are built by src/lib/result_card.py (build_result_card_props).
import { useEffect, useMemo, useRef, useState } from "react"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Markdown } from "@/components/markdown"
import { ArrowDown, ArrowUp, ChartColumn, ChartLine, Code, Database, Download, Table2, Wrench } from "lucide-react"

const MAX_BARS = 25

const C = {
  fg:      "hsl(var(--foreground))",
  muted:   "hsl(var(--muted-foreground))",
  border:  "hsl(var(--border))",
  primary: "hsl(var(--primary))",
  card:    "hsl(var(--card))",
  stripe:  "hsl(var(--accent) / 0.45)",
}

const nf       = new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 })
const isNum    = (v) => typeof v === "number" && Number.isFinite(v)
// Dates come as ISO strings: drop a midnight time ("2021-01-01T00:00:00" → "2021-01-01")
const fmt      = (v) =>
  v === null || v === undefined ? "—" : isNum(v) ? nf.format(v) : String(v).replace(/[T ]00:00:00(\.0+)?$/, "")
const humanize = (s) => {
  const t = String(s).replace(/([a-z])([A-Z])/g, "$1 $2").replace(/_/g, " ").trim()
  return t.charAt(0).toUpperCase() + t.slice(1)
}
const pad2     = (v) => (isNum(v) && v < 100 ? String(v).padStart(2, "0") : String(v))
const labelOf  = (row, idx, sep) =>
  idx.map((i) => row[i]).filter((v) => v !== null && v !== undefined).map(sep === "-" ? pad2 : fmt).join(sep)


// ── KPI tiles ─────────────────────────────────────────────────────────────────

function CountUp({ value }) {
  const [shown, setShown] = useState(isNum(value) ? 0 : value)
  useEffect(() => {
    if (!isNum(value)) return
    let frame
    const start = performance.now()
    const tick = (now) => {
      const p = Math.min((now - start) / 800, 1)
      setShown(value * (1 - Math.pow(1 - p, 3)))
      if (p < 1) frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [value])
  return fmt(isNum(value) && Number.isInteger(value) ? Math.round(shown) : shown)
}

function Kpis({ kpis }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))", gap: 12, padding: 16 }}>
      {kpis.map((k, i) => (
        <div key={i} style={{ border: `1px solid ${C.border}`, borderRadius: 12, padding: "14px 16px", background: C.stripe }}>
          <div style={{ fontSize: 12, color: C.muted, textTransform: "uppercase", letterSpacing: ".05em" }}>{humanize(k.label)}</div>
          <div style={{ fontSize: 32, fontWeight: 700, lineHeight: 1.15, marginTop: 6, fontVariantNumeric: "tabular-nums", color: C.fg }}>
            <CountUp value={k.value} />
          </div>
          {k.caption && <div style={{ fontSize: 13, color: C.muted, marginTop: 4 }}>{k.caption}</div>}
        </div>
      ))}
    </div>
  )
}


// ── Charts ────────────────────────────────────────────────────────────────────

function BarChart({ rows, x, y }) {
  const [grown, setGrown] = useState(false)
  useEffect(() => {
    const frame = requestAnimationFrame(() => setGrown(true))
    return () => cancelAnimationFrame(frame)
  }, [])

  // Largest first: a ranking reads better than the query order (often alphabetical)
  const data = rows
    .map((r) => ({ label: labelOf(r, x, " "), value: r[y] }))
    .sort((a, b) => (isNum(b.value) ? b.value : -Infinity) - (isNum(a.value) ? a.value : -Infinity))
    .slice(0, MAX_BARS)
  const max  = Math.max(...data.map((d) => (isNum(d.value) ? Math.abs(d.value) : 0)), 0) || 1

  return (
    <div style={{ padding: "8px 4px" }}>
      {data.map((d, i) => (
        <div key={i} style={{ display: "grid", gridTemplateColumns: "minmax(90px, 32%) 1fr", alignItems: "center", gap: 12, padding: "3px 0" }}>
          <div title={d.label} style={{ fontSize: 13, color: C.fg, textAlign: "right", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
            {d.label || "—"}
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
            <div
              style={{
                height: 18,
                borderRadius: 4,
                background: C.primary,
                opacity: 0.35 + 0.65 * (isNum(d.value) ? Math.abs(d.value) / max : 0),
                width: grown && isNum(d.value) ? `${Math.max((Math.abs(d.value) / max) * 82, 0.5)}%` : "0%",
                transition: `width 700ms cubic-bezier(.2,.8,.2,1) ${i * 30}ms`,
              }}
            />
            <span style={{ fontSize: 12, color: C.muted, fontVariantNumeric: "tabular-nums", whiteSpace: "nowrap" }}>{fmt(d.value)}</span>
          </div>
        </div>
      ))}
      {rows.length > MAX_BARS && (
        <div style={{ fontSize: 12, color: C.muted, textAlign: "center", marginTop: 8 }}>
          Top {MAX_BARS} of {rows.length} rows — the Table tab shows them all
        </div>
      )}
    </div>
  )
}

function niceTicks(lo, hi, count) {
  const span = hi - lo
  const mag  = Math.pow(10, Math.floor(Math.log10(span / count)))
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= count)
  const out  = []
  for (let v = Math.floor(lo / step) * step; v <= Math.ceil(hi / step) * step + step / 2; v += step) out.push(+v.toFixed(10))
  return out
}

function LineChart({ rows, x, y }) {
  const ref = useRef(null)
  const [width, setWidth] = useState(0)
  const [hover, setHover] = useState(null)
  const [drawn, setDrawn] = useState(false)
  const gradientId = useMemo(() => `rc-grad-${Math.random().toString(36).slice(2)}`, [])

  useEffect(() => {
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width))
    observer.observe(ref.current)
    return () => observer.disconnect()
  }, [])
  useEffect(() => {
    if (!width || drawn) return
    const frame = requestAnimationFrame(() => setDrawn(true))
    return () => cancelAnimationFrame(frame)
  }, [width, drawn])

  const H = 280, P = { l: 60, r: 20, t: 20, b: 34 }
  const pts    = rows.map((r) => ({ label: labelOf(r, x, "-"), value: isNum(r[y]) ? r[y] : null }))
  const values = pts.map((p) => p.value).filter((v) => v !== null)
  let lo = Math.min(0, ...values), hi = Math.max(0, ...values)
  if (hi === lo) hi = lo + 1
  const ticks = niceTicks(lo, hi, 4)
  lo = ticks[0]; hi = ticks[ticks.length - 1]

  const iw = Math.max(width - P.l - P.r, 1), ih = H - P.t - P.b
  const sx = (i) => P.l + (pts.length === 1 ? iw / 2 : (i * iw) / (pts.length - 1))
  const sy = (v) => P.t + ih - ((v - lo) / (hi - lo)) * ih

  const defined = pts.map((p, i) => ({ ...p, i })).filter((p) => p.value !== null)
  const line    = defined.map((p, k) => `${k ? "L" : "M"}${sx(p.i)},${sy(p.value)}`).join(" ")
  const area    = defined.length > 1
    ? `${line} L${sx(defined[defined.length - 1].i)},${sy(Math.max(lo, 0))} L${sx(defined[0].i)},${sy(Math.max(lo, 0))} Z`
    : ""
  const labelStep = Math.ceil(pts.length / Math.max(Math.floor(iw / 90), 1))

  const onMove = (e) => {
    const rect = ref.current.getBoundingClientRect()
    const i = Math.round(((e.clientX - rect.left - P.l) / iw) * (pts.length - 1))
    setHover(Math.min(Math.max(i, 0), pts.length - 1))
  }
  const hovered = hover !== null && pts[hover]

  return (
    <div ref={ref} style={{ position: "relative", height: H }} onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
      {width > 0 && (
        <svg width={width} height={H} style={{ display: "block", overflow: "visible" }}>
          <defs>
            <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={C.primary} stopOpacity="0.28" />
              <stop offset="100%" stopColor={C.primary} stopOpacity="0" />
            </linearGradient>
          </defs>
          {ticks.map((t) => (
            <g key={t}>
              <line x1={P.l} x2={P.l + iw} y1={sy(t)} y2={sy(t)} stroke={C.border} strokeDasharray={t === 0 ? "" : "3 4"} />
              <text x={P.l - 10} y={sy(t)} dy="0.32em" textAnchor="end" fontSize="11" fill={C.muted}>{nf.format(t)}</text>
            </g>
          ))}
          {pts.map((p, i) => i % labelStep === 0 && (
            <text key={i} x={sx(i)} y={H - 10} textAnchor="middle" fontSize="11" fill={C.muted}>{p.label}</text>
          ))}
          <path d={area} fill={`url(#${gradientId})`} style={{ opacity: drawn ? 1 : 0, transition: "opacity 900ms ease 300ms" }} />
          <path
            d={line} fill="none" stroke={C.primary} strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round"
            pathLength="1" strokeDasharray="1" strokeDashoffset={drawn ? 0 : 1}
            style={{ transition: "stroke-dashoffset 1100ms cubic-bezier(.4,0,.2,1)" }}
          />
          {hovered && hovered.value !== null && (
            <g>
              <line x1={sx(hover)} x2={sx(hover)} y1={P.t} y2={P.t + ih} stroke={C.muted} strokeDasharray="3 3" />
              <circle cx={sx(hover)} cy={sy(hovered.value)} r="5" fill={C.card} stroke={C.primary} strokeWidth="2.5" />
            </g>
          )}
        </svg>
      )}
      {hovered && hovered.value !== null && (
        <div
          style={{
            position: "absolute", left: sx(hover), top: sy(hovered.value) - 12, transform: "translate(-50%, -100%)",
            background: C.card, border: `1px solid ${C.border}`, borderRadius: 8, padding: "6px 10px",
            boxShadow: "0 6px 20px rgba(0,0,0,.12)", pointerEvents: "none", whiteSpace: "nowrap",
          }}
        >
          <div style={{ fontSize: 11, color: C.muted }}>{hovered.label}</div>
          <div style={{ fontSize: 14, fontWeight: 600, color: C.fg, fontVariantNumeric: "tabular-nums" }}>{fmt(hovered.value)}</div>
        </div>
      )}
    </div>
  )
}

function Chart({ chart, columns, rows }) {
  const [series, setSeries] = useState(chart.y[0])
  const Plot = chart.type === "line" ? LineChart : BarChart
  return (
    <div style={{ padding: "4px 16px 16px" }}>
      {chart.y.length > 1 && (
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 8 }}>
          {chart.y.map((i) => (
            <button
              key={i}
              onClick={() => setSeries(i)}
              style={{
                fontSize: 12, padding: "3px 10px", borderRadius: 999, cursor: "pointer",
                border: `1px solid ${i === series ? C.primary : C.border}`,
                color: i === series ? C.primary : C.muted, background: "transparent",
              }}
            >
              {humanize(columns[i])}
            </button>
          ))}
        </div>
      )}
      {chart.y.length === 1 && (
        <div style={{ fontSize: 12, color: C.muted, marginBottom: 8 }}>{humanize(columns[series])}</div>
      )}
      <Plot key={series} rows={rows} x={chart.x} y={series} />
    </div>
  )
}


// ── Table ─────────────────────────────────────────────────────────────────────

function DataTable({ columns, rows, truncated }) {
  const [sort, setSort] = useState(null) // { i, dir }
  const numeric = useMemo(() => columns.map((_, i) => rows.every((r) => r[i] === null || isNum(r[i]))), [columns, rows])
  const sorted  = useMemo(() => {
    if (!sort) return rows
    return [...rows].sort((a, b) => {
      const u = a[sort.i], v = b[sort.i]
      if (u === null) return 1
      if (v === null) return -1
      return sort.dir * (isNum(u) && isNum(v) ? u - v : String(u).localeCompare(String(v)))
    })
  }, [rows, sort])
  const onSort = (i) =>
    setSort((s) => (s && s.i === i ? { i, dir: -s.dir } : { i, dir: numeric[i] ? -1 : 1 }))

  const cell = { padding: "7px 12px", fontSize: 13, whiteSpace: "nowrap", maxWidth: 320, overflow: "hidden", textOverflow: "ellipsis" }
  return (
    <div>
      <div style={{ maxHeight: 380, overflow: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr>
              {columns.map((c, i) => (
                <th
                  key={i}
                  onClick={() => onSort(i)}
                  style={{
                    ...cell, position: "sticky", top: 0, background: C.card, cursor: "pointer", userSelect: "none",
                    textAlign: numeric[i] ? "right" : "left", fontWeight: 600, color: C.muted, fontSize: 12,
                    borderBottom: `1px solid ${C.border}`,
                  }}
                >
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                    {c}
                    {sort && sort.i === i && (sort.dir > 0 ? <ArrowUp size={12} /> : <ArrowDown size={12} />)}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((r, k) => (
              <tr key={k} style={{ background: k % 2 ? C.stripe : "transparent" }}>
                {r.map((v, i) => (
                  <td
                    key={i}
                    title={v === null ? "" : String(v)}
                    style={{
                      ...cell, textAlign: numeric[i] ? "right" : "left", fontVariantNumeric: "tabular-nums",
                      color: v === null ? C.muted : C.fg,
                    }}
                  >
                    {fmt(v)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {truncated && (
        <div style={{ fontSize: 12, color: C.muted, padding: "8px 12px", borderTop: `1px solid ${C.border}` }}>
          Only the first {rows.length} rows were read (safety cap) — the query returned more.
        </div>
      )}
    </div>
  )
}


// ── Card ──────────────────────────────────────────────────────────────────────

function toCsv(columns, rows) {
  const esc = (v) => {
    if (v === null || v === undefined) return ""
    const s = String(v)
    return /[",\n;]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
  }
  return [columns, ...rows].map((r) => r.map(esc).join(",")).join("\n")
}

export default function ResultCard() {
  const { columns, rows, rowCount, truncated, sql, tables, retries, dbName, chart, kpis } = props
  // A single-row answer is already told by its KPI tiles: keep the tabs collapsed
  const [tab, setTab] = useState(chart ? "chart" : kpis ? "" : "table")
  const [visible, setVisible] = useState(false)
  useEffect(() => {
    const frame = requestAnimationFrame(() => setVisible(true))
    return () => cancelAnimationFrame(frame)
  }, [])

  const downloadCsv = () => {
    const blob = new Blob(["﻿" + toCsv(columns, rows)], { type: "text/csv;charset=utf-8" })
    const url  = URL.createObjectURL(blob)
    const a    = Object.assign(document.createElement("a"), { href: url, download: `${dbName || "result"}.csv` })
    a.click()
    URL.revokeObjectURL(url)
  }

  const ChartIcon = chart && chart.type === "line" ? ChartLine : ChartColumn
  const trigger   = { display: "inline-flex", alignItems: "center", gap: 6 }

  return (
    <Card
      style={{
        width: "100%", overflow: "hidden", marginTop: 4,
        opacity: visible ? 1 : 0, transform: visible ? "none" : "translateY(6px)",
        transition: "opacity 400ms ease, transform 400ms ease",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "10px 16px", borderBottom: `1px solid ${C.border}`, flexWrap: "wrap" }}>
        <Database size={15} color={C.muted} />
        <span style={{ fontWeight: 600, fontSize: 14, color: C.fg }}>{dbName}</span>
        <span style={{ fontSize: 13, color: C.muted }}>
          · {nf.format(rowCount)}{truncated ? "+" : ""} {rowCount === 1 ? "row" : "rows"} · {columns.length} {columns.length === 1 ? "column" : "columns"}
        </span>
        {tables && tables.length > 0 && (
          <span title="Tables read by the query" style={{ display: "inline-flex", alignItems: "center", gap: 4, fontSize: 13, color: C.muted, flexWrap: "wrap" }}>
            · from
            {tables.map((t) => (
              <code key={t} style={{ fontSize: 12, padding: "1px 6px", borderRadius: 6, border: `1px solid ${C.border}`, color: C.fg }}>{t}</code>
            ))}
          </span>
        )}
        {retries > 0 && (
          <Badge
            variant="outline"
            title="The first query failed validation; the agent read the error and rewrote it."
            style={{ display: "inline-flex", alignItems: "center", gap: 4, fontWeight: 500 }}
          >
            <Wrench size={12} /> Self-corrected{retries > 1 ? ` ×${retries}` : ""}
          </Badge>
        )}
        <div style={{ flex: 1 }} />
        <Button variant="ghost" size="sm" onClick={downloadCsv} style={trigger}>
          <Download size={14} /> CSV
        </Button>
      </div>

      {kpis && <Kpis kpis={kpis} />}

      <Tabs value={tab} onValueChange={setTab}>
        <div style={{ padding: "10px 16px 0" }}>
          <TabsList>
            {chart && <TabsTrigger value="chart" style={trigger}><ChartIcon size={14} /> Chart</TabsTrigger>}
            <TabsTrigger value="table" style={trigger}><Table2 size={14} /> Table</TabsTrigger>
            <TabsTrigger value="sql" style={trigger}><Code size={14} /> SQL</TabsTrigger>
          </TabsList>
        </div>
        {chart && (
          <TabsContent value="chart">
            <Chart chart={chart} columns={columns} rows={rows} />
          </TabsContent>
        )}
        <TabsContent value="table">
          <DataTable columns={columns} rows={rows} truncated={truncated} />
        </TabsContent>
        <TabsContent value="sql">
          <div style={{ padding: "0 16px 8px" }}>
            <Markdown>{"```sql\n" + sql + "\n```"}</Markdown>
          </div>
        </TabsContent>
      </Tabs>
      {!tab && <div style={{ height: 12 }} />}
    </Card>
  )
}

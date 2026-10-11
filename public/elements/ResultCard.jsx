// Result card of a SQL answer: KPI tiles, chart, sortable table, SQL, CSV export.
// Props are built by src/ui/result_card.py (build_result_card_props).
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

// Separator of the card header items: its own element, so the flex gap spaces it evenly
const Dot = () => <span style={{ fontSize: 13, color: C.muted }}>·</span>

const nf       = new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 })
const isNum    = (v) => typeof v === "number" && Number.isFinite(v)
// Dates come as ISO strings: drop a midnight time ("2021-01-01T00:00:00" → "2021-01-01")
const fmt      = (v) =>
  v === null || v === undefined ? "—" : isNum(v) ? nf.format(v) : String(v).replace(/[T ]00:00:00(\.0+)?$/, "")


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
          <div style={{ fontSize: 12, color: C.muted, textTransform: "uppercase", letterSpacing: ".05em" }}>{k.label}</div>
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
// The chart comes ready to draw from src/ui/charts.py: x labels in order, then one list of
// values per series. These components only draw it.

const SERIES_COLORS = ["hsl(215 65% 55%)", "hsl(165 50% 40%)", "hsl(265 50% 62%)"]
const pctFmt        = new Intl.NumberFormat(undefined, { style: "percent", maximumFractionDigits: 1, signDisplay: "exceptZero" })

// Series color: the last series (e.g. the latest year) takes the primary color
const seriesColor = (k, count) => (k === count - 1 ? C.primary : SERIES_COLORS[k % SERIES_COLORS.length])

function Pills({ options, value, onChange }) {
  return (
    <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
      {options.map((o) => (
        <button
          key={o.value}
          onClick={() => onChange(o.value)}
          style={{
            fontSize: 12, padding: "3px 10px", borderRadius: 999, cursor: "pointer",
            border: `1px solid ${o.value === value ? C.primary : C.border}`,
            color: o.value === value ? C.primary : C.muted, background: "transparent",
          }}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

function Legend({ series }) {
  return (
    <div style={{ display: "flex", gap: 14, flexWrap: "wrap", marginBottom: 8 }}>
      {series.map((s) => (
        <span key={s.name} style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12, color: C.fg }}>
          <span style={{ width: 10, height: 10, borderRadius: 3, background: s.color }} />
          {s.name}
        </span>
      ))}
    </div>
  )
}

function BarChart({ labels, series }) {
  const [grown, setGrown] = useState(false)
  useEffect(() => {
    const frame = requestAnimationFrame(() => setGrown(true))
    return () => cancelAnimationFrame(frame)
  }, [])

  // Bars come largest first: the top MAX_BARS make the ranking
  const multi = series.length > 1
  const shown = labels.slice(0, MAX_BARS)
  const max   = Math.max(...series.flatMap((s) => s.values.slice(0, MAX_BARS).map((v) => (isNum(v) ? Math.abs(v) : 0))), 0) || 1

  return (
    <div style={{ padding: "8px 4px" }}>
      {shown.map((label, i) => (
        <div key={i} style={{ display: "grid", gridTemplateColumns: "minmax(90px, 32%) 1fr", alignItems: "center", gap: 12, padding: multi ? "5px 0" : "3px 0" }}>
          <div title={label} style={{ fontSize: 13, color: C.fg, textAlign: "right", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
            {label}
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 2, minWidth: 0 }}>
            {series.map((s) => {
              const v = s.values[i]
              return (
                <div key={s.name} style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
                  <div
                    title={multi ? `${s.name} : ${fmt(v)}` : undefined}
                    style={{
                      height: multi ? 10 : 18,
                      borderRadius: multi ? 3 : 4,
                      background: s.color,
                      // A single series shades its bars by value; several series keep their color solid
                      opacity: multi ? 1 : 0.35 + 0.65 * (isNum(v) ? Math.abs(v) / max : 0),
                      width: grown && isNum(v) ? `${Math.max((Math.abs(v) / max) * 82, 0.5)}%` : "0%",
                      transition: `width 700ms cubic-bezier(.2,.8,.2,1) ${i * 30}ms`,
                    }}
                  />
                  <span style={{ fontSize: multi ? 11 : 12, color: C.muted, fontVariantNumeric: "tabular-nums", whiteSpace: "nowrap" }}>{fmt(v)}</span>
                </div>
              )
            })}
          </div>
        </div>
      ))}
      {labels.length > MAX_BARS && (
        <div style={{ fontSize: 12, color: C.muted, textAlign: "center", marginTop: 8 }}>
          {MAX_BARS} premières sur {labels.length}. L'onglet Table les affiche toutes.
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

function LineChart({ labels, series }) {
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
  const n      = labels.length
  const values = series.flatMap((s) => s.values).filter((v) => v !== null)
  let lo = Math.min(0, ...values), hi = Math.max(0, ...values)
  if (hi === lo) hi = lo + 1
  const ticks = niceTicks(lo, hi, 4)
  lo = ticks[0]; hi = ticks[ticks.length - 1]

  const iw = Math.max(width - P.l - P.r, 1), ih = H - P.t - P.b
  const sx = (i) => P.l + (n === 1 ? iw / 2 : (i * iw) / (n - 1))
  const sy = (v) => P.t + ih - ((v - lo) / (hi - lo)) * ih

  const paths = series.map((s) => {
    const defined = s.values.map((v, i) => ({ v, i })).filter((p) => p.v !== null)
    const line    = defined.map((p, k) => `${k ? "L" : "M"}${sx(p.i)},${sy(p.v)}`).join(" ")
    const area    = defined.length > 1
      ? `${line} L${sx(defined[defined.length - 1].i)},${sy(Math.max(lo, 0))} L${sx(defined[0].i)},${sy(Math.max(lo, 0))} Z`
      : ""
    return { line, area }
  })
  const labelStep = Math.ceil(n / Math.max(Math.floor(iw / 90), 1))

  const onMove = (e) => {
    const rect = ref.current.getBoundingClientRect()
    const i = Math.round(((e.clientX - rect.left - P.l) / iw) * (n - 1))
    setHover(Math.min(Math.max(i, 0), n - 1))
  }
  const hovered = hover === null ? [] : series.filter((s) => s.values[hover] !== null)
  const tipTop  = hovered.length ? Math.min(...hovered.map((s) => sy(s.values[hover]))) : 0
  // Two series (e.g. 2022 vs 2023): show the change of the last one against the first
  const [first, last] = [series[0], series[series.length - 1]]
  const delta = hover !== null && series.length === 2 && isNum(first.values[hover]) && isNum(last.values[hover]) && first.values[hover] !== 0
    ? (last.values[hover] - first.values[hover]) / Math.abs(first.values[hover])
    : null

  return (
    <div ref={ref} style={{ position: "relative", height: H }} onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
      {width > 0 && (
        <svg width={width} height={H} style={{ display: "block", overflow: "visible" }}>
          {/* One fill per line; faint under several lines so overlapping areas don't blend into a muddy color */}
          <defs>
            {series.map((s, k) => (
              <linearGradient key={k} id={`${gradientId}-${k}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={s.color} stopOpacity={series.length === 1 ? 0.28 : 0.1} />
                <stop offset="100%" stopColor={s.color} stopOpacity="0" />
              </linearGradient>
            ))}
          </defs>
          {ticks.map((t) => (
            <g key={t}>
              <line x1={P.l} x2={P.l + iw} y1={sy(t)} y2={sy(t)} stroke={C.border} strokeDasharray={t === 0 ? "" : "3 4"} />
              <text x={P.l - 10} y={sy(t)} dy="0.32em" textAnchor="end" fontSize="11" fill={C.muted}>{nf.format(t)}</text>
            </g>
          ))}
          {labels.map((label, i) => i % labelStep === 0 && (
            <text key={i} x={sx(i)} y={H - 10} textAnchor="middle" fontSize="11" fill={C.muted}>{label}</text>
          ))}
          {series.map((s, k) => (
            <path
              key={k}
              d={paths[k].area} fill={`url(#${gradientId}-${k})`}
              style={{ opacity: drawn ? 1 : 0, transition: "opacity 900ms ease 300ms" }}
            />
          ))}
          {series.map((s, k) => (
            <path
              key={k}
              d={paths[k].line} fill="none" stroke={s.color} strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round"
              pathLength="1" strokeDasharray="1" strokeDashoffset={drawn ? 0 : 1}
              style={{ transition: "stroke-dashoffset 1100ms cubic-bezier(.4,0,.2,1)" }}
            />
          ))}
          {hovered.length > 0 && (
            <g>
              <line x1={sx(hover)} x2={sx(hover)} y1={P.t} y2={P.t + ih} stroke={C.muted} strokeDasharray="3 3" />
              {hovered.map((s, k) => (
                <circle key={k} cx={sx(hover)} cy={sy(s.values[hover])} r="5" fill={C.card} stroke={s.color} strokeWidth="2.5" />
              ))}
            </g>
          )}
        </svg>
      )}
      {hovered.length > 0 && (
        <div
          style={{
            position: "absolute", left: sx(hover), top: tipTop - 12, transform: "translate(-50%, -100%)",
            background: C.card, border: `1px solid ${C.border}`, borderRadius: 8, padding: "6px 10px",
            boxShadow: "0 6px 20px rgba(0,0,0,.12)", pointerEvents: "none", whiteSpace: "nowrap",
          }}
        >
          <div style={{ fontSize: 11, color: C.muted }}>{labels[hover]}</div>
          {hovered.map((s, k) => (
            <div key={k} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 14, fontWeight: 600, color: C.fg, fontVariantNumeric: "tabular-nums" }}>
              {s.name !== null && <span style={{ width: 8, height: 8, borderRadius: 2, background: s.color }} />}
              {s.name !== null && <span style={{ fontSize: 12, fontWeight: 400, color: C.muted }}>{s.name}</span>}
              {fmt(s.values[hover])}
            </div>
          ))}
          {delta !== null && (
            <div style={{ fontSize: 12, color: C.muted, marginTop: 2, fontVariantNumeric: "tabular-nums" }}>
              {last.name} vs {first.name} : <span style={{ fontWeight: 600, color: C.fg }}>{pctFmt.format(delta)}</span>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

const VIEWS = [{ value: "compare", label: "Comparer" }, { value: "timeline", label: "Chronologique" }]

function Chart({ chart }) {
  const [measure, setMeasure] = useState(0)
  const [view, setView]       = useState("compare")
  const data   = (view === "timeline" ? chart.timeline : chart.measures)[measure]
  const series = data.series.map((s, k) => ({ ...s, color: seriesColor(k, data.series.length) }))
  const Plot   = chart.type === "line" ? LineChart : BarChart
  return (
    <div style={{ padding: "4px 16px 16px" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, flexWrap: "wrap", marginBottom: 8 }}>
        {chart.measures.length > 1
          ? <Pills options={chart.measures.map((m, i) => ({ value: i, label: m.name }))} value={measure} onChange={setMeasure} />
          : <div style={{ fontSize: 12, color: C.muted }}>{data.name}</div>}
        {chart.timeline && <Pills options={VIEWS} value={view} onChange={setView} />}
      </div>
      {series.length > 1 && <Legend series={series} />}
      <Plot key={`${measure}-${view}`} labels={data.labels} series={series} />
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
          Seules les {rows.length} premières lignes ont été lues. La requête en a renvoyé davantage.
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
        <Dot />
        <span style={{ fontSize: 13, color: C.muted }}>
          {nf.format(rowCount)}{truncated ? "+" : ""} {rowCount === 1 ? "ligne" : "lignes"}
        </span>
        <Dot />
        <span style={{ fontSize: 13, color: C.muted }}>
          {columns.length} {columns.length === 1 ? "colonne" : "colonnes"}
        </span>
        {tables && tables.length > 0 && (
          <>
            <Dot />
            <span title="Tables lues par la requête" style={{ display: "inline-flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
              <code style={{ fontSize: 12, color: C.muted, letterSpacing: ".04em" }}>FROM</code>
              {tables.map((t) => (
                <code key={t} style={{ fontSize: 12, padding: "1px 6px", borderRadius: 6, border: "1px solid hsl(var(--border) / 0.6)", color: C.muted }}>{t}</code>
              ))}
            </span>
          </>
        )}
        {retries > 0 && (
          <Badge
            variant="outline"
            title="La première requête a échoué. L'agent a lu l'erreur et l'a corrigée."
            style={{ display: "inline-flex", alignItems: "center", gap: 4, fontWeight: 500 }}
          >
            <Wrench size={12} /> Auto-corrigée{retries > 1 ? ` ×${retries}` : ""}
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
            <Chart chart={chart} />
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

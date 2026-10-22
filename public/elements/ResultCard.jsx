// Result card of a SQL answer: KPI tiles, chart, sortable table, SQL, lineage graph (Calcul), CSV export.
// Props are built by src/ui/result_card.py (build_result_card_props).
import { Children, Fragment, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Markdown } from "@/components/markdown"
import {
  ArrowDown, ArrowUp, BadgeCheck, ChartColumn, ChartLine, Code, CornerDownRight, Database, Download, Funnel, KeyRound, Layers, List,
  Sigma, Table2, Workflow, Wrench,
} from "lucide-react"

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
// Pie slices, largest first: the primary color, then the series colors, then two more
const SLICE_COLORS  = [C.primary, ...SERIES_COLORS, "hsl(38 80% 52%)", "hsl(330 55% 58%)"]
const pctFmt        = new Intl.NumberFormat(undefined, { style: "percent", maximumFractionDigits: 1, signDisplay: "exceptZero" })
const shareFmt      = new Intl.NumberFormat(undefined, { style: "percent", maximumFractionDigits: 1 })
const compactFmt    = new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 })
// A number in a narrow space: in full when short, else compact ("448 386 633,17" → "448,4 M")
const shortFmt      = (v) => (fmt(v).length <= 9 ? fmt(v) : compactFmt.format(v))

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

// Donut of shares: one slice per category, largest first, then the folded tail ("Autres (20)")
// in a neutral color. Slices are arcs of a circle's stroke that sweep in. The legend gives each
// slice's value and share; hovering either highlights the slice.
function PieChart({ parts }) {
  const [grown, setGrown] = useState(false)
  const [hover, setHover] = useState(null)
  useEffect(() => {
    const frame = requestAnimationFrame(() => setGrown(true))
    return () => cancelAnimationFrame(frame)
  }, [])

  const slices = [
    ...parts.labels.map((label, i) => ({ label, value: parts.values[i], color: SLICE_COLORS[i % SLICE_COLORS.length] })),
    ...(parts.others ? [{ label: `Autres (${parts.others.count})`, value: parts.others.value, color: C.muted }] : []),
  ]
  const total = slices.reduce((sum, s) => sum + s.value, 0)
  let start = 0
  for (const s of slices) {
    s.share = s.value / total
    s.start = start
    start  += s.share
  }

  const S = 200, R = 74, W = 30, CIRC = 2 * Math.PI * R
  const GAP = slices.length > 1 ? 2 : 0  // px of card background between slices
  const shown = hover === null ? null : slices[hover]

  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 28, flexWrap: "wrap", padding: "8px 4px" }}>
      <div style={{ position: "relative", width: S, height: S, flexShrink: 0 }}>
        <svg width={S} height={S} viewBox={`0 0 ${S} ${S}`} style={{ display: "block" }}>
          {slices.map((s, i) => (
            <circle
              key={i}
              cx={S / 2} cy={S / 2} r={R} fill="none" stroke={s.color}
              strokeWidth={hover === i ? W + 8 : W}
              strokeDasharray={`${grown ? Math.max(s.share * CIRC - GAP, 0.5) : 0} ${CIRC}`}
              transform={`rotate(${-90 + s.start * 360} ${S / 2} ${S / 2})`}
              opacity={hover === null || hover === i ? 1 : 0.45}
              style={{ cursor: "default", transition: `stroke-dasharray 800ms cubic-bezier(.2,.8,.2,1) ${i * 60}ms, stroke-width 150ms, opacity 150ms` }}
              onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}
            />
          ))}
        </svg>
        {/* Center: the total, or the hovered slice */}
        <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", pointerEvents: "none", textAlign: "center" }}>
          <div style={{ fontSize: 11, color: C.muted, maxWidth: 2 * R - W, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {shown ? shown.label : "Total"}
          </div>
          <div style={{ fontSize: 20, fontWeight: 700, color: C.fg, fontVariantNumeric: "tabular-nums" }}>
            {shown ? shareFmt.format(shown.share) : shortFmt(total)}
          </div>
          {shown && <div style={{ fontSize: 12, color: C.muted, fontVariantNumeric: "tabular-nums" }}>{shortFmt(shown.value)}</div>}
        </div>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 4, minWidth: 0, flex: "1 1 280px", maxWidth: 380 }}>
        {slices.map((s, i) => (
          <div
            key={i}
            onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}
            style={{
              display: "grid", gridTemplateColumns: "10px minmax(0, 1fr) auto auto", alignItems: "center", gap: 10,
              padding: "4px 8px", borderRadius: 6, fontSize: 13, background: hover === i ? C.stripe : "transparent",
            }}
          >
            <span style={{ width: 10, height: 10, borderRadius: 3, background: s.color }} />
            <span title={s.label} style={{ color: C.fg, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{s.label}</span>
            <span style={{ color: C.muted, fontVariantNumeric: "tabular-nums", textAlign: "right" }}>{fmt(s.value)}</span>
            <span style={{ color: C.fg, fontWeight: 600, fontVariantNumeric: "tabular-nums", textAlign: "right", minWidth: 48 }}>{shareFmt.format(s.share)}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

const VIEWS  = [{ value: "compare", label: "Comparer" }, { value: "timeline", label: "Chronologique" }]
const SHAPES = [{ value: "bars", label: "Barres" }, { value: "parts", label: "Parts" }]

function Chart({ chart }) {
  const [measure, setMeasure] = useState(0)
  const [view, setView]       = useState("compare")
  const [shape, setShape]     = useState(chart.parts_first ? "parts" : "bars")
  const data   = (view === "timeline" ? chart.timeline : chart.measures)[measure]
  const series = data.series.map((s, k) => ({ ...s, color: seriesColor(k, data.series.length) }))
  // The pie choice is kept across measures, drawn only for those that read as shares
  const pie    = shape === "parts" && !!data.parts
  const Plot   = chart.type === "line" ? LineChart : BarChart
  return (
    <div style={{ padding: "4px 16px 16px" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, flexWrap: "wrap", marginBottom: 8 }}>
        {chart.measures.length > 1
          ? <Pills options={chart.measures.map((m, i) => ({ value: i, label: m.name }))} value={measure} onChange={setMeasure} />
          : <div style={{ fontSize: 12, color: C.muted }}>{data.name}</div>}
        {chart.timeline && <Pills options={VIEWS} value={view} onChange={setView} />}
        {data.parts && <Pills options={SHAPES} value={shape} onChange={setShape} />}
      </div>
      {series.length > 1 && <Legend series={series} />}
      {pie
        ? <PieChart key={`${measure}-pie`} parts={data.parts} />
        : <Plot key={`${measure}-${view}`} labels={data.labels} series={series} />}
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


// ── Lineage ───────────────────────────────────────────────────────────────────
// The steps of the query come from src/lib/query_lineage.py, physical names: one row per clause.

const STEP_KINDS = { cte: "CTE", derived: "table dérivée", subquery: "sous-requête", branch: "branche", result: "requête finale" }
const mono       = { fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace", fontSize: 12 }

const Chip = ({ children, title }) => (
  <code title={title} style={{ ...mono, padding: "1px 6px", borderRadius: 6, border: `1px solid ${C.border}`, color: C.fg, overflowWrap: "anywhere" }}>
    {children}
  </code>
)

function Row({ label, children }) {
  return (
    <>
      <div style={{ fontSize: 12, color: C.muted, paddingTop: 2 }}>{label}</div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6, alignItems: "baseline", minWidth: 0 }}>{children}</div>
    </>
  )
}

// Columns an output comes from, grouped by table; "*" is a COUNT(*) of the table rows
function Origins({ origins }) {
  const byTable = {}
  origins.forEach((o) => (byTable[o.table] = [...(byTable[o.table] || []), o.column === "*" ? "nombre de lignes" : o.column]))
  return Object.entries(byTable).map(([table, columns], i) => (
    <span key={table} style={{ fontSize: 12, color: C.muted }}>
      {i > 0 && "+ "}<span style={{ color: C.fg }}>{table}</span> · {columns.join(", ")}
    </span>
  ))
}

function LineageStep({ step, titled }) {
  const sort = [...step.order_by, ...(step.limit ? [`LIMIT ${step.limit}`] : [])]
  return (
    <div style={{ display: "grid", gridTemplateColumns: "110px minmax(0, 1fr)", gap: "8px 12px", alignItems: "start" }}>
      {titled && (
        <div style={{ gridColumn: "1 / -1", display: "flex", alignItems: "center", gap: 6, fontSize: 13, fontWeight: 600, color: C.fg }}>
          <Layers size={14} color={C.muted} /> {step.kind === "result" ? "Requête finale" : step.id}
          {step.kind !== "result" && <span style={{ fontWeight: 400, color: C.muted }}>· {STEP_KINDS[step.kind]}</span>}
        </div>
      )}
      <Row label="Sources">
        {step.sources.map((s) => (
          <span key={s.alias} style={{ display: "inline-flex", alignItems: "baseline", gap: 6, flexWrap: "wrap" }}>
            <Chip>{s.kind === "step" ? `↳ ${s.name}` : s.name}{s.alias.toLowerCase() !== s.name.toLowerCase() ? ` ${s.alias}` : ""}</Chip>
            {Object.keys(s.columns).length > 0 && (
              <span style={{ fontSize: 12, color: C.muted }}>
                {Object.entries(s.columns).map(([c, roles]) => <span key={c} title={roles.join(", ")}>{c}</span>)
                  .reduce((all, c, i) => (i ? [...all, ", ", c] : [c]), [])}
              </span>
            )}
          </span>
        ))}
      </Row>
      {step.operation !== "SELECT" && <Row label="Opération"><Chip>{step.operation}</Chip></Row>}
      {step.joins.length > 0 && (
        <Row label="Liaisons">
          {step.joins.map((j, i) => (
            <Chip key={i} title={j.implicit ? "Condition écrite dans le WHERE" : undefined}>
              {j.type !== "INNER" ? `${j.type} · ` : ""}{j.condition || `${j.sources.join(", ")} sans condition`}
            </Chip>
          ))}
        </Row>
      )}
      {step.filters.length > 0 && (
        <Row label="Filtres">
          {step.filters.map((f, i) => <Chip key={i}>{f.condition}</Chip>)}
        </Row>
      )}
      {step.group_by.length > 0 && <Row label="Regroupement">{step.group_by.map((g, i) => <Chip key={i}>{g}</Chip>)}</Row>}
      {step.having.length > 0 && <Row label="Après calcul">{step.having.map((h, i) => <Chip key={i}>{h}</Chip>)}</Row>}
      {step.distinct && <Row label="Doublons"><span style={{ fontSize: 12, color: C.fg }}>retirés (DISTINCT)</span></Row>}
      {sort.length > 0 && <Row label="Tri">{sort.map((o, i) => <Chip key={i}>{o}</Chip>)}</Row>}
      <Row label="Colonnes">
        {step.outputs.map((o, i) => (
          <div key={i} style={{ width: "100%", display: "flex", flexWrap: "wrap", alignItems: "baseline", gap: 6 }}>
            <Chip title={o.expression}>{o.name}</Chip>
            {o.aggregated && <Badge variant="outline" style={{ fontWeight: 500 }}>agrégée</Badge>}
            {o.window && <Badge variant="outline" style={{ fontWeight: 500 }}>fenêtre</Badge>}
            {o.origins.length > 0 && <CornerDownRight size={12} color={C.muted} style={{ alignSelf: "center" }} />}
            <Origins origins={o.origins} />
          </div>
        ))}
      </Row>
    </div>
  )
}

function LineageDetail({ lineage }) {
  const steps = lineage.steps
  return (
    <div style={{ padding: "8px 16px 16px", display: "flex", flexDirection: "column", gap: 16 }}>
      {steps.map((step, i) => (
        <div key={step.id} style={i ? { borderTop: `1px dashed ${C.border}`, paddingTop: 14 } : undefined}>
          <LineageStep step={step} titled={steps.length > 1} />
        </div>
      ))}
    </div>
  )
}


// ── Lineage graph ─────────────────────────────────────────────────────────────
// The final step of the query, from src/lib/query_lineage.py, told as a graph in three columns:
//   sources   each table (or intermediate step): its row count, the columns read, join keys first;
//             the joins are arcs between keys
//   clauses   Filtres (which rows are kept), Calcul (what is computed, for each what), chained
//   result    the row count, the sort, the output columns
// The edges go column by column: a column leaves its table at its own height and reaches the
// clause that uses it; without a calculation, the filters pass the columns read on to the result.
// The phrases are built here from the summary of the lineage; the exact SQL, the descriptions of
// the semantic layer and the metric definitions are in the tooltips. Hovering an output, a source
// column, a clause or a join lights its path.

const GRAINS    = { year: "année", quarter: "trimestre", month: "mois", week: "semaine", day: "jour", month_of_year: "mois" }
const OPS       = { "=": "=", "<>": "≠", ">": ">", ">=": "≥", "<": "<", "<=": "≤" }
const FUNCS     = { sum: "somme", avg: "moyenne", min: "minimum", max: "maximum" }
const LIKES     = { starts: ["commence par", "ne commence pas par"], ends: ["finit par", "ne finit pas par"],
                    contains: ["contient", "ne contient pas"], pattern: ["ressemble à", "ne ressemble pas à"] }
const EDGE      = "hsl(var(--muted-foreground) / 0.4)"
const ARC       = "hsl(var(--muted-foreground) / 0.6)"
const LIT       = { soft: "hsl(var(--primary) / 0.15)", border: "hsl(var(--primary) / 0.55)" }
// The cards: flat, an even tint by role (slate for the sources, near neutral for the clauses, the
// primary color for the result); lit by the hover with a border and, in the dark theme, a glow
const CARD_STYLES = `
  .lg-node { --lg-tint: 215 22% 62%; --lg-glow: 215 22% 62%; border: 1px solid hsl(var(--lg-tint) / 0.32);
             background: linear-gradient(hsl(var(--lg-tint) / 0.05), hsl(var(--lg-tint) / 0.05)), hsl(var(--card));
             box-shadow: 0 1px 2px rgb(0 0 0 / 0.06); transition: opacity 150ms, border-color 200ms, box-shadow 250ms; }
  .lg-node.lg-stage { --lg-tint: 215 6% 60%; --lg-glow: var(--primary); }
  .lg-node.lg-res { --lg-tint: var(--primary); --lg-glow: var(--primary); }
  .lg-node .lg-hr { background: hsl(var(--lg-tint) / 0.22); }
  .lg-node.lg-on { border-color: hsl(var(--lg-glow) / 0.75); box-shadow: 0 6px 18px -8px hsl(var(--lg-glow) / 0.45); }
  .dark .lg-node { border-color: hsl(var(--lg-tint) / 0.28);
                   background: linear-gradient(hsl(var(--lg-tint) / 0.07), hsl(var(--lg-tint) / 0.07)), hsl(var(--card));
                   box-shadow: inset 0 1px 0 hsl(0 0% 100% / 0.04), 0 1px 2px rgb(0 0 0 / 0.3); }
  .dark .lg-node.lg-on { border-color: hsl(var(--lg-glow) / 0.7);
                         box-shadow: inset 0 1px 0 hsl(0 0% 100% / 0.05), 0 0 0 1px hsl(var(--lg-glow) / 0.25), 0 0 22px -4px hsl(var(--lg-glow) / 0.45); }
`
const rowsFmt   = new Intl.NumberFormat("fr-FR", { notation: "compact", maximumSignificantDigits: 2 })
const fullFmt   = new Intl.NumberFormat("fr-FR")
const monthFmt  = new Intl.DateTimeFormat("fr-FR", { month: "long", year: "numeric" })
const monthOnly = new Intl.DateTimeFormat("fr-FR", { month: "long" })
const countOf   = (n, one, many) => `${fullFmt.format(n)} ${n > 1 ? many : one}`

// Tables of the database a step reads, through the steps it reads
function tablesOf(steps, id, seen = new Set()) {
  if (seen.has(id) || !steps[id]) return []
  seen.add(id)
  return [...new Set(steps[id].sources.flatMap((s) => (s.kind === "table" ? [s.name] : tablesOf(steps, s.name, seen))))]
}

// ── Words ──

const Label = ({ children, style, title }) => <span title={title} style={{ fontSize: 11.5, color: C.muted, ...style }}>{children}</span>
const Mono  = ({ children }) => <span style={{ ...mono, fontSize: 11.5 }}>{children}</span>

// A formula wraps after an operator, never inside its innermost parentheses; a subtraction shows the minus sign
function Formula({ text, size = 11.5 }) {
  const parts = String(text).replace(/ - /g, " − ").split(/(\([^()]*\))/)
  return (
    <span style={{ ...mono, fontSize: size }}>
      {parts.map((part, i) => (i % 2 ? <span key={i} style={{ whiteSpace: "nowrap" }}>{part}</span> : part))}
    </span>
  )
}

// An identifier on its own, lit by the hover; a join key carries a key
function Pill({ port, lit, keyed, title, onHover, children }) {
  const on = !!lit && !!port && lit.has(port)
  return (
    <span
      data-port={port}
      title={title}
      onMouseEnter={onHover && (() => onHover(port))}
      onMouseLeave={onHover && (() => onHover(null))}
      style={{
        ...mono, fontSize: 11.5, lineHeight: 1.65, padding: "0 7px", borderRadius: 5, display: "inline-flex", alignItems: "center",
        gap: 4, whiteSpace: "nowrap", transition: "color 150ms, background 150ms", cursor: onHover ? "default" : undefined,
        background: on ? LIT.soft : "hsl(var(--muted))", color: on ? C.primary : C.fg,
      }}
    >
      {keyed && <KeyRound size={10} style={{ opacity: 0.65, flex: "none" }} />}
      {children}
    </span>
  )
}

// How an output is computed: "somme de unitprice × quantity, arrondie à 2 décimales"
function AggPhrase({ output }) {
  const a = output.aggregate
  if (!a) return <Formula text={output.formula} size={11} />
  const operand = <Formula text={a.operand} size={11} />
  return (
    <>
      {a.function === "count"
        ? (a.operand === "*" ? "nombre de lignes" : <>nombre de {operand} {a.distinct ? "distincts" : "renseignés"}</>)
        : <>{FUNCS[a.function]} de {a.distinct && "valeurs distinctes de "}{operand}</>}
      {a.round !== null && a.round !== undefined && <Label>, arrondie à {a.round} décimale{a.round > 1 ? "s" : ""}</Label>}
    </>
  )
}

// What a filter keeps, after its column: { text, detail }; text null for a condition shown as SQL
function filterPhrase(f, steps) {
  switch (f.kind) {
    case "period": {
      const m = /^(\d{4})-(\d{2})$/.exec(f.value)
      if (f.grain === "year")          return { text: `est en ${f.value}` }
      if (f.grain === "month" && m)    return { text: `est en ${monthFmt.format(new Date(Number(m[1]), Number(m[2]) - 1, 1))}` }
      if (f.grain === "month_of_year") return { text: `est en ${monthOnly.format(new Date(2000, Number(f.value) - 1, 1))}` }
      if (f.grain === "quarter")       return { text: `est au trimestre ${f.value}` }
      return { text: `${GRAINS[f.grain] || f.grain} = ${f.value}` }
    }
    case "compare":  return { text: `${f.grain ? `${GRAINS[f.grain]} ` : ""}${OPS[f.op]} ${f.value}` }
    case "range":    return { text: `entre ${f.low} et ${f.high}`, detail: f.high_op === "<" ? "fin exclue" : null }
    case "in":
      return f.values.length <= 3
        ? { text: `${f.negated ? "hors" : "parmi"} ${f.values.join(", ")}` }
        : { text: `${f.negated ? "hors" : "parmi"} ${f.values.length} valeurs`, detail: `${f.values.slice(0, 4).join(", ")}…` }
    case "like":     return { text: `${LIKES[f.mode][f.negated ? 1 : 0]} « ${f.value} »` }
    case "null":     return { text: f.negated ? "est renseigné" : "est vide" }
    case "subquery": {
      const origin = steps[f.steps[0]]?.outputs[0]?.origins[0]
      return origin
        ? { text: <>{f.negated ? "absent de" : "figure dans"} <Mono>{origin.table}</Mono></>, detail: `colonne ${origin.column}` }
        : { text: f.negated ? "hors d'une sous-requête" : "dans une sous-requête" }
    }
    default:         return { text: null }
  }
}

// "top 5 par chiffre_affaires ↓", "trié par mois ↑", "limité à 10 lignes"
function SortPhrase({ step }) {
  const keys = step.summary.order_by.map((o, i) => (
    <span key={i}>{i > 0 && ", puis "}<Mono>{o.label}</Mono> {o.descending ? "↓" : "↑"}</span>
  ))
  if (step.limit && keys.length) return <>top {step.limit} par {keys}</>
  if (keys.length) return <>trié par {keys}</>
  return step.limit ? <>limité à {step.limit} lignes</> : null
}

// ── Model ──

// The joins as arcs between two ports: a key column of each side, or the tables when a join has no key
function linksOf(step) {
  return step.joins.flatMap((j) => {
    const sql = `${j.type} JOIN · ${j.condition || "sans condition"}`, outer = j.type !== "INNER"
    if (j.pairs.length) {
      return j.pairs.map(([x, y]) => ({ a: `c:${x.source}.${x.column}`, b: `c:${y.source}.${y.column}`, aliases: [x.source, y.source], outer, sql }))
    }
    return j.sources.length > 1 ? [{ a: `t:${j.sources[0]}`, b: `t:${j.sources[1]}`, aliases: j.sources.slice(0, 2), outer, sql }] : []
  })
}

// The sources in join order, joined tables next to each other: a walk from one end of the longest path
function orderSources(step, links) {
  const adj = Object.fromEntries(step.sources.map((s) => [s.alias, []]))
  links.forEach(({ aliases: [a, b] }) => { if (adj[a] && adj[b] && a !== b) { adj[a].push(b); adj[b].push(a) } })
  const farthest = (start) => {
    const dist = { [start]: 0 }, queue = [start]
    while (queue.length) {
      const a = queue.shift()
      adj[a].forEach((b) => { if (!(b in dist)) { dist[b] = dist[a] + 1; queue.push(b) } })
    }
    return Object.keys(dist).reduce((x, y) => (dist[y] > dist[x] ? y : x))
  }
  const order = [], seen = new Set()
  const walk = (a) => { seen.add(a); order.push(a); adj[a].forEach((b) => { if (!seen.has(b)) walk(b) }) }
  step.sources.forEach((s) => { if (!seen.has(s.alias)) walk(farthest(s.alias)) })
  return order.map((a) => step.sources.find((s) => s.alias === a))
}

// Ports: "t:alias" (a table), "c:alias.column", "filter", "calc", "o:i" (an output); each edge lists
// the outputs it feeds (outs)
function modelOf(lineage) {
  const steps    = Object.fromEntries(lineage.steps.map((s) => [s.id, s]))
  const step     = lineage.steps[lineage.steps.length - 1]
  const computed = step.outputs.filter((o) => o.aggregated || o.window)
  const has      = { filter: step.filters.length > 0,
                     calc: computed.length > 0 || step.group_by.length > 0 || step.having.length > 0 || step.distinct }
  const readBy   = (alias, column) => step.outputs.flatMap((o, i) => (o.reads.some((r) => r.source === alias && r.column === column) ? [i] : []))
  const edges    = []
  step.sources.forEach((s) => {
    Object.entries(s.columns).forEach(([column, roles]) => {
      const port = `c:${s.alias}.${column}`, outs = readBy(s.alias, column)
      // Without a calculation, the filters pass the columns read on to the result
      if (has.filter && (roles.includes("where") || (!has.calc && outs.length))) edges.push({ from: port, to: "filter", outs: has.calc ? [] : outs })
      if (has.calc && (outs.length || roles.includes("group_by") || roles.includes("having"))) edges.push({ from: port, to: "calc", outs })
      if (!has.calc && !has.filter) outs.forEach((i) => edges.push({ from: port, to: `o:${i}`, outs: [i] }))
    })
    const counting = readBy(s.alias, "*")   // COUNT(*) reads the rows of the table, not a column
    if (counting.length) edges.push({ from: `t:${s.alias}`, to: has.calc ? "calc" : `o:${counting[0]}`, outs: counting })
  })
  const last = has.calc ? "calc" : has.filter ? "filter" : null
  if (last) step.outputs.forEach((_, i) => edges.push({ from: last, to: `o:${i}`, outs: [i] }))
  if (has.filter && has.calc) edges.push({ from: "filter", to: "calc", outs: [], chain: true })
  const links = linksOf(step)
  return { steps, step, computed, has, edges, links, sources: orderSources(step, links) }
}

// What a hover lights: edges (on), ports (lit), join arcs (arcs)
function focusOf(key, { edges, links }) {
  if (key === null) return null
  const on = new Set(), lit = new Set([key]), arcs = new Set()
  const light = (k) => { on.add(k); lit.add(edges[k].from); lit.add(edges[k].to) }
  if (key.startsWith("arc:")) {
    const k = Number(key.slice(4))
    arcs.add(k); lit.add(links[k].a); lit.add(links[k].b)
  } else if (key.startsWith("o:")) {
    const i = Number(key.slice(2))
    edges.forEach((e, k) => { if (e.outs.includes(i)) light(k) })
  } else if (key === "filter" || key === "calc") {
    edges.forEach((e, k) => { if (e.to === key || e.from === key) light(k) })
  } else {   // a source column: the clauses it reaches, the outputs it feeds
    const outs = new Set()
    edges.forEach((e, k) => { if (e.from === key) { light(k); e.outs.forEach((i) => outs.add(i)) } })
    edges.forEach((e, k) => { if (e.to.startsWith("o:") && !e.from.includes(":") && e.outs.some((i) => outs.has(i))) light(k) })
  }
  return { on, lit, arcs }
}

// ── Nodes ──

// A card of a role (src, stage, res, see CARD_STYLES): its header (title, aside, sub: what the card
// is), a hairline from border to border, its body
function Node({ id, role, on, focus, icon: Icon, title, code, aside, sub, tip, onHover, children }) {
  const body = Children.toArray(children).length > 0
  return (
    <div
      data-node={id}
      title={tip}
      className={`lg-node lg-${role}${on ? " lg-on" : ""}`}
      onMouseEnter={onHover && (() => onHover(id))}
      onMouseLeave={onHover && (() => onHover(null))}
      style={{ borderRadius: 10, padding: "9px 10px 10px", minWidth: 0, opacity: focus && !on ? 0.45 : 1, cursor: onHover ? "default" : undefined }}
    >
      <div data-port={id.startsWith("t:") ? id : undefined} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, fontWeight: 600, color: C.fg }}>
        <Icon size={13} color={on ? C.primary : C.muted} style={{ flex: "none" }} />
        <span style={code ? { ...mono, fontWeight: 500 } : undefined}>{title}</span>
        {aside && <Label style={{ marginLeft: "auto", fontWeight: 400, whiteSpace: "nowrap" }}>{aside}</Label>}
      </div>
      {sub && <div style={{ display: "flex", flexDirection: "column", marginTop: 2 }}>{sub}</div>}
      {body && <div className="lg-hr" style={{ height: 1, margin: "8px -10px 0" }} />}
      {children}
    </div>
  )
}

function SourceNode({ source, model, context, focus, onHover }) {
  const { steps } = model
  const columns  = Object.entries(source.columns)
  const isKey    = (roles) => roles.includes("join")
  const ordered  = [...columns.filter(([, r]) => isKey(r)), ...columns.filter(([, r]) => !isKey(r))]
  const isStep   = source.kind === "step"
  const sub      = isStep ? steps[source.name] : null
  const facts    = (!isStep && context.tables[source.name]) || {}
  // Its role (a bridge, a join that brings nothing) shows in the arcs and the columns: no label.
  // Its row count, an order of magnitude measured by the last profiling, discreet beside its name
  const rows = facts.rows !== null && facts.rows !== undefined && (
    <span title={`${countOf(facts.rows, "ligne", "lignes")}, mesurées au dernier profilage`} style={{ fontSize: 10.5, opacity: 0.75 }}>
      {rowsFmt.format(facts.rows)}
    </span>
  )
  const lit  = focus?.lit
  const on   = !!focus && (lit.has(`t:${source.alias}`) || columns.some(([c]) => lit.has(`c:${source.alias}.${c}`)))
  const groups = sub?.summary.group_by || []
  return (
    <Node id={`t:${source.alias}`} role="src" on={on} focus={focus} icon={isStep ? Layers : Table2} title={source.name} code aside={rows}
          tip={facts.description || (source.alias !== source.name ? `${source.name} AS ${source.alias}` : undefined)}
          sub={isStep && (
            <>
              <Label>{STEP_KINDS[sub?.kind] || "étape"} sur {tablesOf(steps, source.name).join(", ")}</Label>
              {groups.length > 0 && <Label>pour chaque {groups.map((g, i) => <span key={i}>{i > 0 && ", "}<Mono>{g.column || g.sql}</Mono></span>)}</Label>}
            </>
          )}>
      {ordered.length > 0 && (
        <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-start", gap: 4, marginTop: 8 }}>
          {ordered.map(([column, roles]) => {
            const made = sub?.outputs.find((o) => o.name === column && o.aggregate)   // how a step computes the column
            return (
              <Fragment key={column}>
                <Pill port={`c:${source.alias}.${column}`} lit={lit} keyed={isKey(roles)} title={(facts.columns || {})[column]} onHover={onHover}>
                  {column}
                </Pill>
                {made && <div style={{ fontSize: 11, color: C.muted, margin: "-1px 0 3px 2px" }}><AggPhrase output={made} /></div>}
              </Fragment>
            )
          })}
        </div>
      )}
    </Node>
  )
}

function FilterNode({ model, focus, onHover }) {
  const { step, steps } = model
  return (
    <Node id="filter" role="stage" on={!!focus && focus.lit.has("filter")} focus={focus} icon={Funnel} title="Filtres" onHover={onHover}
          aside={countOf(step.summary.filters.length, "condition", "conditions")}
          tip={step.filters.map((f) => f.condition).join("\nAND ")}>
      <div style={{ marginTop: 7 }}><Label>garde les lignes où</Label></div>
      <div style={{ display: "flex", flexDirection: "column", gap: 6, marginTop: 5 }}>
        {step.summary.filters.map((f, i) => {
          const { text, detail } = filterPhrase(f, steps)
          return (
            <div key={i} style={{ display: "flex", flexWrap: "wrap", alignItems: "baseline", gap: "2px 7px", minWidth: 0, fontSize: 12.5, color: C.fg }}>
              {text === null ? <Formula text={f.sql} /> : (
                <>
                  {f.column && <Pill>{f.column}</Pill>}
                  <span>{text}</span>
                  {detail && <span style={{ ...mono, fontSize: 11, color: C.muted, flexBasis: "100%", overflowWrap: "anywhere" }}>{detail}</span>}
                </>
              )}
            </div>
          )
        })}
      </div>
    </Node>
  )
}

// Between two sections of a card's body: from border to border like the hairline under the header,
// dashed (3px dashes, 3px gaps) and paler
const SectionRule = () => (
  <div style={{
    height: 1, margin: "9px -10px",
    backgroundImage: "linear-gradient(to right, hsl(var(--muted-foreground) / 0.4) 50%, transparent 50%)",
    backgroundSize: "6px 1px", backgroundRepeat: "repeat-x",
  }} />
)

function CalcNode({ model, context, focus, onHover }) {
  const { step, computed } = model
  const groups = step.summary.group_by
  // HAVING written with the output names: "nb_commandes > 50" rather than "COUNT(DISTINCT orderid) > 50"
  const having = step.summary.having.map((h) => computed.reduce((text, o) => text.split(o.formula).join(o.name), h))
  // Its sections, those the query has: what is computed, for each what, which groups are kept
  const sections = [
    computed.length > 0 && computed.map((o, i) => (
      <div key={i} style={{ marginTop: i ? 8 : 0 }}>
        <div style={{ ...mono, fontSize: 11, fontWeight: 500, color: C.fg }}>{o.name}</div>
        <div style={{ fontSize: 11.5, color: C.fg }}>
          {o.window ? <><Label>fenêtre : </Label><Formula text={o.formula} size={11} /></> : <>= <AggPhrase output={o} /></>}
        </div>
        {o.metric && (
          <div
            title={context.metrics[o.metric] || "Métrique de la couche sémantique"}
            style={{ display: "inline-flex", alignItems: "center", gap: 4, marginTop: 4, fontSize: 11.5, color: C.primary }}
          >
            <BadgeCheck size={12} style={{ flex: "none" }} /> métrique {o.metric}
          </div>
        )}
      </div>
    )),
    groups.length > 0 && (
      <div style={{ display: "flex", flexWrap: "wrap", alignItems: "baseline", gap: "4px 5px", fontSize: 12.5, color: C.fg }}>
        <Label>pour chaque</Label>
        {groups.map((g, i) => (
          <span key={i} style={{ display: "inline-flex", alignItems: "baseline", gap: 5 }}>
            {g.grain ? <>{GRAINS[g.grain]} de <Pill>{g.column}</Pill></> : g.column ? <Pill>{g.column}</Pill> : <Formula text={g.sql} />}
            {i < groups.length - 1 && ","}
          </span>
        ))}
      </div>
    ),
    having.length > 0 && (
      <div style={{ fontSize: 12.5, color: C.fg }}>
        <Label>garde les groupes où </Label>{having.map((h, i) => <span key={i}>{i > 0 && " et "}<Formula text={h} /></span>)}
      </div>
    ),
    step.distinct && <Label>retire les doublons</Label>,
  ].filter(Boolean)
  return (
    <Node id="calc" role="stage" on={!!focus && focus.lit.has("calc")} focus={focus} icon={Sigma} title="Calcul" onHover={onHover}
          aside={computed.length ? countOf(computed.length, "agrégat", "agrégats") : groups.length ? "regroupement" : "sans doublons"}
          tip={[...computed.map((o) => `${o.name} = ${o.expression}`), ...step.group_by.map((g) => `GROUP BY ${g}`),
                ...step.having.map((h) => `HAVING ${h}`)].join("\n")}>
      {sections.map((section, i) => (
        <Fragment key={i}>
          {i > 0 && <SectionRule />}
          <div style={{ marginTop: i ? 0 : 8 }}>{section}</div>
        </Fragment>
      ))}
    </Node>
  )
}

function ResultNode({ model, rowCount, truncated, focus, onHover }) {
  const { step } = model
  const on = !!focus && step.outputs.some((_, i) => focus.lit.has(`o:${i}`))
  return (
    <Node id="out" role="res" on={on} focus={focus} icon={List} title="Résultat" aside={`${countOf(rowCount, "ligne", "lignes")}${truncated ? "+" : ""}`}
          sub={(step.operation !== "SELECT" || step.summary.order_by.length > 0 || step.limit) && (
            <>
              {step.operation !== "SELECT" && <Label>{step.operation} des branches</Label>}
              {(step.summary.order_by.length > 0 || step.limit) && <Label><SortPhrase step={step} /></Label>}
            </>
          )}>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-start", gap: 4, marginTop: 8 }}>
        {step.outputs.map((o, i) => <Pill key={i} port={`o:${i}`} lit={focus?.lit} title={o.formula} onHover={onHover}>{o.name}</Pill>)}
      </div>
    </Node>
  )
}

// ── Graph ──

function LineageGraph({ lineage, rowCount, truncated }) {
  const model   = useMemo(() => modelOf(lineage), [lineage])
  const context = lineage.context || { tables: {}, metrics: {} }
  const [hover, setHover]   = useState(null)
  const focus               = useMemo(() => focusOf(hover, model), [hover, model])
  const ref                 = useRef(null)
  const [shapes, setShapes] = useState({ edges: [], arcs: [] })

  // Edges from the node boxes: a source port leaves its node by the right border, an output port is
  // reached on the left border, a clause spreads its inputs on its left border in the order of their
  // heights, the clauses chain top to bottom. Join arcs on the left of the sources, from key to key,
  // nested when they overlap. Redrawn on resize.
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const draw = () => {
      const box    = el.getBoundingClientRect()
      const port   = (p) => el.querySelector(`[data-port="${CSS.escape(p)}"]`)
      const node   = (n) => el.querySelector(`[data-node="${CSS.escape(n)}"]`)
      const mid    = (e) => { const r = e.getBoundingClientRect(); return r.top + r.height / 2 - box.top }
      const anchor = (p, side) => {
        const e = port(p) || node(p), n = e?.closest("[data-node]")?.getBoundingClientRect()
        return e && n ? { x: (side === "right" ? n.right : n.left) - box.left, y: mid(e) } : null
      }
      const { edges, links } = model
      const inputs = {}
      edges.forEach((e, k) => { if (!e.chain && !e.to.startsWith("o:")) (inputs[e.to] = inputs[e.to] || []).push(k) })
      Object.values(inputs).forEach((list) => list.sort((a, b) => (anchor(edges[a].from, "right")?.y ?? 0) - (anchor(edges[b].from, "right")?.y ?? 0)))
      const edgePaths = edges.flatMap((e, k) => {
        if (e.chain) {
          const a = node(e.from)?.getBoundingClientRect(), b = node(e.to)?.getBoundingClientRect()
          if (!a || !b) return []
          const x = a.left + a.width / 2 - box.left
          return [{ k, d: `M${x},${a.bottom - box.top} L${x},${b.top - box.top}`, ends: [] }]
        }
        const s = anchor(e.from, "right")
        let t = null
        if (e.to.startsWith("o:")) t = anchor(e.to, "left")
        else {
          const r = node(e.to)?.getBoundingClientRect(), list = inputs[e.to], i = list.indexOf(k)
          if (r) {
            const span = Math.min(r.height - 18, 7 * (list.length - 1))
            t = { x: r.left - box.left, y: r.top + r.height / 2 - box.top + (list.length > 1 ? -span / 2 + (span * i) / (list.length - 1) : 0) }
          }
        }
        if (!s || !t) return []
        const dx = (t.x - s.x) * 0.5
        return [{ k, d: `M${s.x},${s.y} C${s.x + dx},${s.y} ${t.x - dx},${t.y} ${t.x},${t.y}`, ends: [s, t] }]
      })
      const arcs = links.flatMap((l, k) => {
        const a = port(l.a), b = port(l.b)
        if (!a || !b) return []
        const ya = mid(a), yb = mid(b)
        return [{ k, outer: l.outer, sql: l.sql, top: Math.min(ya, yb), bottom: Math.max(ya, yb),
                  x: a.closest("[data-node]").getBoundingClientRect().left - box.left }]
      })
      const placed = []
      ;[...arcs].sort((x, y) => (x.bottom - x.top) - (y.bottom - y.top)).forEach((arc) => {
        const used = new Set(placed.filter((p) => p.top < arc.bottom && arc.top < p.bottom).map((p) => p.level))
        arc.level = 0
        while (used.has(arc.level)) arc.level++
        placed.push(arc)
      })
      setShapes({
        edges: edgePaths,
        arcs: arcs.map((arc) => {
          const { x, top, bottom } = arc, b = Math.min(10 + 7 * arc.level, 30), r = Math.min(5, (bottom - top) / 2)
          return { ...arc, d: `M${x},${top} L${x - b + r},${top} Q${x - b},${top} ${x - b},${top + r} L${x - b},${bottom - r} Q${x - b},${bottom} ${x - b + r},${bottom} L${x},${bottom}` }
        }),
      })
    }
    draw()
    const observer = new ResizeObserver(draw)
    observer.observe(el)
    el.querySelectorAll("[data-node]").forEach((n) => observer.observe(n))
    return () => observer.disconnect()
  }, [model])

  const stages = [
    model.has.filter && <FilterNode key="filter" model={model} focus={focus} onHover={setHover} />,
    model.has.calc && <CalcNode key="calc" model={model} context={context} focus={focus} onHover={setHover} />,
  ].filter(Boolean)
  const widths = stages.length ? [145, 200, 150] : [145, 150]
  const grows  = stages.length ? [1, 1.3, 1] : [1, 1]
  const gap    = stages.length ? 34 : 90
  const pad    = model.links.length ? 40 : 16   // room for the join arcs
  const column = { display: "flex", flexDirection: "column", position: "relative", zIndex: 1, minWidth: 0 }

  return (
    <div style={{ overflowX: "auto" }}>
      <style>{CARD_STYLES}</style>
      <div
        ref={ref}
        style={{
          position: "relative", display: "grid", alignItems: "center", columnGap: gap, padding: `14px 16px 20px ${pad}px`,
          // The sources and the result are as wide as their longest identifier (a pill never wraps), 150px at least
          gridTemplateColumns: widths.map((w, i) => (i === 0 || i === widths.length - 1 ? `minmax(min-content, ${grows[i]}fr)` : `minmax(${w}px, ${grows[i]}fr)`)).join(" "),
          minWidth: widths.reduce((sum, w) => sum + w, 0) + gap * (widths.length - 1) + pad + 16,
        }}
      >
        <svg style={{ position: "absolute", inset: 0, width: "100%", height: "100%", overflow: "visible", pointerEvents: "none" }}>
          {shapes.edges.map((p) => {
            const on = !!focus && focus.on.has(p.k), color = on ? C.primary : EDGE
            return (
              <g key={`e${p.k}`} style={{ opacity: focus && !on ? 0.2 : 1, transition: "opacity 150ms" }}>
                <path d={p.d} fill="none" stroke={color} strokeWidth={on ? 1.5 : 1.2} style={{ transition: "stroke 150ms" }} />
                {p.ends.map((end, i) => <circle key={i} cx={end.x} cy={end.y} r={2} fill={color} />)}
              </g>
            )
          })}
          {shapes.arcs.map((a) => {
            const on = !!focus && focus.arcs.has(a.k), color = on ? C.primary : ARC
            return (
              <g key={`a${a.k}`} style={{ opacity: focus && !on ? 0.2 : 1, transition: "opacity 150ms" }}>
                <path d={a.d} fill="none" stroke={color} strokeWidth={on ? 1.5 : 1.2} strokeDasharray={a.outer ? "3 3" : undefined} />
                <circle cx={a.x} cy={a.top} r={2} fill={color} />
                <circle cx={a.x} cy={a.bottom} r={2} fill={color} />
                <path d={a.d} fill="none" stroke="transparent" strokeWidth={10} style={{ pointerEvents: "stroke" }}
                      onMouseEnter={() => setHover(`arc:${a.k}`)} onMouseLeave={() => setHover(null)}>
                  <title>{a.sql}</title>
                </path>
              </g>
            )
          })}
        </svg>
        <div style={{ ...column, gap: 12, alignSelf: "start", minWidth: widths[0] }}>
          {model.sources.map((s) => <SourceNode key={s.alias} source={s} model={model} context={context} focus={focus} onHover={setHover} />)}
        </div>
        {stages.length > 0 && <div style={{ ...column, gap: 22 }}>{stages}</div>}
        <div style={{ ...column, minWidth: widths[widths.length - 1] }}>
          <ResultNode model={model} rowCount={rowCount} truncated={truncated} focus={focus} onHover={setHover} />
        </div>
      </div>
    </div>
  )
}

const LINEAGE_VIEWS = [{ value: "graph", label: "Graphe" }, { value: "detail", label: "Détail" }]

function Lineage({ lineage, rowCount, truncated }) {
  const [view, setView] = useState("graph")
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "flex-end", padding: "12px 16px 0" }}>
        <Pills options={LINEAGE_VIEWS} value={view} onChange={setView} />
      </div>
      {view === "graph"
        ? <LineageGraph lineage={lineage} rowCount={rowCount} truncated={truncated} />
        : <LineageDetail lineage={lineage} />}
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
  const { columns, rows, rowCount, truncated, sql, tables, lineage, retries, dbName, chart, kpis } = props
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
            {lineage && <TabsTrigger value="lineage" style={trigger}><Workflow size={14} /> Calcul</TabsTrigger>}
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
        {lineage && (
          <TabsContent value="lineage">
            <Lineage lineage={lineage} rowCount={rowCount} truncated={truncated} />
          </TabsContent>
        )}
      </Tabs>
      {!tab && <div style={{ height: 12 }} />}
    </Card>
  )
}

// Clickable follow-up questions shown under an answer: one click asks the question.
// Props are built by src/ui/streaming.py (send_follow_ups): { questions: string[] }
import { useEffect, useState } from "react"
import { CornerDownRight, Sparkles } from "lucide-react"

const C = {
  fg:      "hsl(var(--foreground))",
  muted:   "hsl(var(--muted-foreground))",
  border:  "hsl(var(--border))",
  primary: "hsl(var(--primary))",
  card:    "hsl(var(--card))",
}

function Chip({ question, index, visible, disabled, onAsk }) {
  const [hover, setHover] = useState(false)
  const active = hover && !disabled
  return (
    <button
      onClick={() => onAsk(question)}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      disabled={disabled}
      style={{
        display: "inline-flex", alignItems: "center", gap: 8, textAlign: "left",
        padding: "7px 14px", borderRadius: 999, fontSize: 13, lineHeight: 1.35,
        border: `1px solid ${active ? C.primary : C.border}`,
        background: C.card, color: active ? C.primary : C.fg,
        cursor: disabled ? "default" : "pointer", opacity: visible ? (disabled ? 0.5 : 1) : 0,
        transform: visible ? "none" : "translateY(4px)",
        transition: `opacity 300ms ease ${index * 80}ms, transform 300ms ease ${index * 80}ms, border-color 150ms, color 150ms`,
      }}
    >
      <CornerDownRight size={14} color={C.primary} style={{ flexShrink: 0 }} />
      {question}
    </button>
  )
}

export default function FollowUps() {
  const [visible, setVisible] = useState(false)
  const [asked, setAsked] = useState(false)
  useEffect(() => {
    const frame = requestAnimationFrame(() => setVisible(true))
    return () => cancelAnimationFrame(frame)
  }, [])

  // Ask once: the next answer brings its own suggestions
  const onAsk = (question) => {
    if (asked) return
    setAsked(true)
    sendUserMessage(question)
  }

  return (
    <div style={{ marginTop: 4 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, color: C.muted, marginBottom: 8 }}>
        <Sparkles size={13} color={C.primary} /> Ask next
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
        {props.questions.map((q, i) => (
          <Chip key={i} question={q} index={i} visible={visible} disabled={asked} onAsk={onAsk} />
        ))}
      </div>
    </div>
  )
}

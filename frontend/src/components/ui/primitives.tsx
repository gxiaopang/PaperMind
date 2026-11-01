import type { ReactNode } from 'react'
import { CircleAlert, CircleCheck, Info, LoaderCircle } from 'lucide-react'

export function Spinner({ size = 16, label }: { size?: number; label?: string }) {
  return (
    <span className="spinner-wrap">
      <LoaderCircle size={size} className="spin" aria-hidden="true" />
      {label && <span>{label}</span>}
    </span>
  )
}

export function Notice({
  tone,
  children,
  testId,
  action,
}: {
  tone: 'error' | 'success' | 'info'
  children: ReactNode
  testId?: string
  action?: ReactNode
}) {
  const Icon = tone === 'error' ? CircleAlert : tone === 'success' ? CircleCheck : Info
  return (
    <div
      className={`notice ${tone}`}
      role={tone === 'error' ? 'alert' : tone === 'success' ? 'status' : undefined}
      data-testid={testId}
    >
      <Icon size={16} className="notice-icon" aria-hidden="true" />
      <div className="notice-body">{children}</div>
      {action}
    </div>
  )
}

export function Meter({
  label,
  used,
  reserved = 0,
  limit,
  display,
  hint,
}: {
  label: ReactNode
  used: number
  reserved?: number
  limit: number
  display: ReactNode
  hint?: ReactNode
}) {
  const usedPct = limit > 0 ? Math.min(100, (used / limit) * 100) : 0
  const reservedPct = limit > 0 ? Math.min(100 - usedPct, (reserved / limit) * 100) : 0
  const tone = usedPct + reservedPct >= 90 ? 'danger' : usedPct + reservedPct >= 70 ? 'warn' : 'ok'
  return (
    <div className={`meter meter-${tone}`}>
      <div className="meter-top">
        <span className="meter-label">{label}</span>
        <strong className="meter-value">{display}</strong>
      </div>
      <div
        className="meter-track"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(usedPct)}
      >
        <span className="meter-fill" style={{ width: `${usedPct}%` }} />
        {reservedPct > 0 && (
          <span className="meter-reserved" style={{ left: `${usedPct}%`, width: `${reservedPct}%` }} />
        )}
      </div>
      {hint && <span className="meter-hint">{hint}</span>}
    </div>
  )
}

export function EmptyState({
  icon,
  title,
  children,
  action,
  compact,
}: {
  icon: ReactNode
  title: string
  children?: ReactNode
  action?: ReactNode
  compact?: boolean
}) {
  return (
    <div className={`empty-state ${compact ? 'compact' : ''}`}>
      <div className="empty-illustration">{icon}</div>
      <h2>{title}</h2>
      {children && <p>{children}</p>}
      {action}
    </div>
  )
}

export function PageHeader({
  eyebrow,
  title,
  description,
  actions,
}: {
  eyebrow?: string
  title: ReactNode
  description?: ReactNode
  actions?: ReactNode
}) {
  return (
    <div className="page-header">
      <div className="page-header-text">
        {eyebrow && <span className="eyebrow">{eyebrow}</span>}
        <h1>{title}</h1>
        {description && <p className="muted">{description}</p>}
      </div>
      {actions && <div className="page-header-actions">{actions}</div>}
    </div>
  )
}

export function Skeleton({ lines = 3 }: { lines?: number }) {
  return (
    <div className="skeleton" aria-hidden="true">
      {Array.from({ length: lines }, (_, i) => (
        <span key={i} style={{ width: `${90 - i * 14}%` }} />
      ))}
    </div>
  )
}


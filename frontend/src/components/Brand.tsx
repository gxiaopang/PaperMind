export function BrandMark({ size = 34 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 40 40" aria-hidden="true" className="brand-mark">
      <rect width="40" height="40" rx="11" fill="currentColor" />
      <path d="M12 11.5h11.5a5.5 5.5 0 0 1 0 11H17v6h-5z" fill="none" stroke="var(--brand-mark-ink)" strokeWidth="2.6" strokeLinejoin="round" />
      <circle cx="27.5" cy="27.5" r="3" fill="var(--brand-mark-accent)" />
    </svg>
  )
}

export function Brand({ compact = false }: { compact?: boolean }) {
  return (
    <div className="brand">
      <BrandMark />
      {!compact && (
        <span className="brand-name">
          PaperMind<span className="brand-dot">.</span>
        </span>
      )}
    </div>
  )
}


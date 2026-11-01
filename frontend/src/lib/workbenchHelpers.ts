import type { CitationItem, Location } from './api'

export type PdfHighlight = {
  paperId: string
  pageNumber: number // 1-based for display
  bboxes: [number, number, number, number][] // 0..1 relative [x0, y0, x1, y1]
  label?: string
}

export function validBbox(bbox?: [number, number, number, number] | null): bbox is [number, number, number, number] {
  if (!bbox || bbox.length !== 4) return false
  const [x0, y0, x1, y1] = bbox
  return (
    Number.isFinite(x0) &&
    Number.isFinite(y0) &&
    Number.isFinite(x1) &&
    Number.isFinite(y1) &&
    x0 >= 0 &&
    y0 >= 0 &&
    x1 <= 1 &&
    y1 <= 1 &&
    x1 > x0 &&
    y1 > y0
  )
}

export function locationsToHighlight(
  paperId: string,
  locations: Location[],
  preferredPageIdx?: number,
  label?: string,
): PdfHighlight {
  if (!locations || locations.length === 0) {
    return { paperId, pageNumber: 1, bboxes: [], label }
  }
  const targetIdx =
    preferredPageIdx !== undefined && locations.some((l) => l.page_idx === preferredPageIdx)
      ? preferredPageIdx
      : locations[0].page_idx
  const bboxes = locations
    .filter((l) => l.page_idx === targetIdx)
    .map((l) => l.bbox)
    .filter(validBbox)
  return {
    paperId,
    pageNumber: Math.max(1, targetIdx + 1),
    bboxes,
    label,
  }
}

export function resolvePaperAssetSrc(src: string | undefined, paperId?: string): string | null {
  if (!src) return null
  const trimmed = src.trim()
  if (/^(javascript|data|vbscript|file):/i.test(trimmed)) return null
  if (trimmed.startsWith('/api/v1/papers/')) return trimmed
  if (paperId && !/^[a-z][a-z0-9+.-]*:/i.test(trimmed) && !trimmed.startsWith('//')) {
    const clean = trimmed.replace(/^\.?\//, '')
    if (clean.includes('..')) return null
    return `/api/v1/papers/${paperId}/assets/${clean}`
  }
  return null
}

export function isSafeExternalHref(href: string | undefined): boolean {
  if (!href) return false
  const trimmed = href.trim()
  if (trimmed.startsWith('#')) return true
  try {
    const parsed = new URL(trimmed, 'https://papermind.local')
    return parsed.protocol === 'http:' || parsed.protocol === 'https:' || parsed.protocol === 'mailto:'
  } catch {
    return false
  }
}

export type StreamCandidateSource = {
  number: number
  text: string
  chunk_id: string
  paper_id: string
  version_id: string
  title: string
  section_path: string[]
  locations: Location[]
}

export type StreamVerifyState = {
  citations_valid?: boolean
  grounding?: string
  sources_revoked?: boolean
  safe_fallback?: boolean
  citation_issues?: {
    uncited_units?: string[]
    invalid_numbers?: number[]
  }
}

export type RunProgressState = {
  runId: string
  question: string
  mode: 'fast' | 'agentic'
  status: 'streaming' | 'completed' | 'failed'
  steps: string[]
  rewrites: string[]
  retrievalCount: number
  generationCount: number
  degraded: string[]
  sources: StreamCandidateSource[]
  draftText: string
  finalText: string | null
  finalStatus: string | null
  verify: StreamVerifyState | null
  citations: CitationItem[]
  usageTokens: number | null
  usageEstimated: boolean
  error: string | null
  seenEventIds: number[]
}

export function applySseEvent(
  prev: RunProgressState,
  eventName: string,
  payload: Record<string, unknown>,
): RunProgressState {
  const eventId = typeof payload.id === 'number' ? payload.id : null
  if (eventId !== null && prev.seenEventIds.includes(eventId)) {
    return prev
  }
  const seenEventIds = eventId !== null ? [...prev.seenEventIds, eventId] : prev.seenEventIds
  const attempt = typeof payload.attempt === 'number' ? payload.attempt : prev.generationCount

  switch (eventName) {
    case 'step': {
      const node = String(payload.node || '')
      return {
        ...prev,
        seenEventIds,
        steps: node ? [...prev.steps, node] : prev.steps,
      }
    }
    case 'rewrite': {
      const q = String(payload.query || '')
      return {
        ...prev,
        seenEventIds,
        rewrites: q ? [...prev.rewrites, q] : prev.rewrites,
      }
    }
    case 'sources': {
      const rawSources = Array.isArray(payload.sources)
        ? (payload.sources as StreamCandidateSource[])
        : prev.sources
      const retrievalCount =
        typeof payload.retrieval_count === 'number' ? payload.retrieval_count : Math.max(1, prev.retrievalCount)
      const degraded = Array.isArray(payload.degraded)
        ? (payload.degraded as string[])
        : prev.degraded
      return {
        ...prev,
        seenEventIds,
        sources: rawSources,
        retrievalCount,
        degraded,
      }
    }
    case 'regenerate': {
      return {
        ...prev,
        seenEventIds,
        generationCount: Math.max(attempt, prev.generationCount + 1),
        draftText: '',
        finalText: null,
        citations: [],
      }
    }
    case 'token': {
      const chunk = String(payload.text || '')
      return {
        ...prev,
        seenEventIds,
        generationCount: Math.max(1, attempt),
        draftText: prev.draftText + chunk,
      }
    }
    case 'verify': {
      return {
        ...prev,
        seenEventIds,
        verify: {
          citations_valid: Boolean(payload.citations_valid),
          grounding: typeof payload.grounding === 'string' ? payload.grounding : undefined,
          sources_revoked: Boolean(payload.sources_revoked),
          safe_fallback: Boolean(payload.safe_fallback),
        },
      }
    }
    case 'answer': {
      return {
        ...prev,
        seenEventIds,
        finalText: String(payload.text ?? ''),
        finalStatus: typeof payload.status === 'string' ? payload.status : 'answered',
      }
    }
    case 'citations': {
      const list = Array.isArray(payload.citations) ? (payload.citations as CitationItem[]) : []
      return {
        ...prev,
        seenEventIds,
        citations: list,
      }
    }
    case 'done': {
      const usageObj = payload.usage as { tokens?: number; estimated?: boolean } | undefined
      return {
        ...prev,
        seenEventIds,
        status: 'completed',
        finalStatus: typeof payload.status === 'string' ? payload.status : prev.finalStatus,
        usageTokens: typeof usageObj?.tokens === 'number' ? usageObj.tokens : prev.usageTokens,
        usageEstimated: Boolean(usageObj?.estimated),
      }
    }
    case 'error': {
      return {
        ...prev,
        seenEventIds,
        status: 'failed',
        error: String(payload.message || '问答服务暂时不可用，请稍后重试'),
      }
    }
    default:
      return { ...prev, seenEventIds }
  }
}

export function splitAnswerWithCitations(text: string): Array<{ type: 'text'; value: string } | { type: 'cite'; number: number }> {
  const parts: Array<{ type: 'text'; value: string } | { type: 'cite'; number: number }> = []
  const regex = /\[(\d+)\]/g
  let lastIndex = 0
  let match: RegExpExecArray | null
  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push({ type: 'text', value: text.slice(lastIndex, match.index) })
    }
    parts.push({ type: 'cite', number: Number.parseInt(match[1], 10) })
    lastIndex = regex.lastIndex
  }
  if (lastIndex < text.length) {
    parts.push({ type: 'text', value: text.slice(lastIndex) })
  }
  return parts
}

import { useUI } from './store'

export type User = { id: string; email: string; email_verified_at: string | null; role: string; status: string }
export type Collection = { id: string; name: string; description: string; created_at: string }

export type Paper = {
  id: string
  collection_id: string | null
  title: string
  authors: string[]
  year: number | null
  page_count: number
  size_bytes: number
  status: string
  pending_version_id: string | null
  active_version_id: string | null
  error: string | null
  abstract: string | null
  venue: string | null
  doi: string | null
  arxiv_id: string | null
}

export type Location = {
  page_idx: number
  bbox?: [number, number, number, number] | null
}

export type CitationItem = {
  id: string
  number: number
  paper_id: string
  version_id: string
  chunk_id: string
  title: string
  text: string
  locations: Location[]
  deleted: boolean
}

export type ScopeConfig = {
  collection_ids?: string[] | null
  paper_ids?: string[] | null
}

export type Conversation = {
  project_id?: string | null
  id: string
  title: string
  scope: ScopeConfig
  active_paper_ids: string[]
  created_at: string
}

export type ChatMessage = {
  id: string
  run_id: string
  seq: number
  role: 'user' | 'assistant'
  content: string
  rewritten_query: string | null
  route: string | null
  verify_result: Record<string, unknown>
  feedback: 'up' | 'down' | null
  citations: CitationItem[]
}

export type UsageNodeSummary = {
  node: string
  model: string
  tokens: number
  calls: number
  estimated: boolean
}

export type UsageData = {
  papers: number
  storage_bytes: number
  parse_pages: number
  parse_pages_reserved: number
  model_tokens: number
  model_tokens_reserved: number
  model_tokens_estimated: boolean
  quota_timezone: string
  limits: {
    papers: number
    storage_bytes: number
    daily_parse_pages: number
    daily_model_tokens: number
  }
  by_node?: UsageNodeSummary[]
}

export type UserSettings = {
  answer_language: 'zh' | 'en'
  detail_level: 'concise' | 'standard' | 'detailed'
}

export type ParsedSection = {
  id: string
  parent_id: string | null
  title: string
  level: number
  path: string[]
}

export type ParsedBlock = {
  id: string
  kind: string
  text: string
  section_id: string | null
  section_path: string[]
  locations: Location[]
  indexable: boolean
}

export type ParsedDocumentInfo = {
  page_count: number
  sections: ParsedSection[]
  blocks: ParsedBlock[]
  assets: string[]
  warnings: string[]
}

export type ChunkInfo = {
  chunk_metadata?: {
    level?: number
    section_role?: string
    summary_method?: string
  }
  id: string
  kind: string
  text: string
  locations: Location[]
  section_path: string[]
  indexable: boolean
  unindexed_reason: string | null
  token_count: number
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message)
  }
}

export async function getCsrfToken(): Promise<string> {
  const response = await fetch('/api/v1/auth/csrf', { credentials: 'same-origin' })
  if (!response.ok) throw new ApiError(response.status, '无法建立安全连接，请稍后重试')
  const { csrf_token } = await response.json()
  return csrf_token
}

export async function api<T>(path: string, method = 'GET', body?: unknown): Promise<T> {
  const headers: Record<string, string> = {}
  if (body !== undefined) {
    headers['Content-Type'] = body instanceof Blob ? 'application/pdf' : 'application/json'
  }
  if (method !== 'GET') {
    headers['X-CSRF-Token'] = await getCsrfToken()
  }
  const response = await fetch(`/api/v1${path}`, {
    method,
    headers,
    credentials: 'same-origin',
    body: body === undefined ? undefined : body instanceof Blob ? body : JSON.stringify(body),
  })
  if (response.status === 204) return undefined as T
  const data = await response.json().catch(() => ({ detail: '服务暂时不可用，请稍后重试' }))
  if (!response.ok) {
    const message = typeof data.detail === 'string' ? data.detail
      : path === '/auth/registration-code' ? '请输入有效的邮箱地址'
      : path === '/auth/verify-email-code' ? '请输入 6 位数字验证码'
      : path === '/auth/register' ? '请检查邮箱、6 位验证码和 8–128 个字符的密码'
      : '请检查输入内容，密码需要 8–128 个字符'
    if (response.status === 401 && path !== '/auth/me' && !path.startsWith('/auth/')) {
      useUI.getState().setSessionExpired('会话已过期，请重新登录以继续操作。')
    }
    throw new ApiError(response.status, message)
  }
  return data as T
}

export async function apiText(path: string): Promise<string> {
  const response = await fetch(`/api/v1${path}`, { credentials: 'same-origin' })
  if (!response.ok) {
    if (response.status === 401) {
      useUI.getState().setSessionExpired('会话已过期，请重新登录以继续操作。')
    }
    const data = await response.json().catch(() => ({ detail: '获取内容失败' }))
    throw new ApiError(response.status, typeof data.detail === 'string' ? data.detail : '获取内容失败')
  }
  return await response.text()
}


export type ResearchProject = { id: string; name: string; description: string; created_at: string }
export type ResearchState = Record<string, Array<{ content: string; source_seq: number }>>
export type ConversationMemory = {
  summary: string; research_state: ResearchState; summarized_seq: number
  project_id: string | null; active_paper_ids: string[]
}
export type MemoryEntry = {
  id: string; level: 'user' | 'project' | 'paper' | 'conversation'
  project_id: string | null; conversation_id: string | null; paper_id: string | null
  kind: string; content: string; status: 'candidate' | 'active' | 'superseded'
  epistemic: 'user_statement' | 'hypothesis' | 'evidence_linked'; pinned: boolean
  source_message_id: string | null; citation_ids: string[]
  reading_anchor?: { version_id: string; page_idx: number }
  evidence_status: 'current' | 'outdated' | 'unavailable' | 'unlinked'
  evidence: Array<{ id: string; paper_id?: string; title?: string; status: string }>
  created_at: string; updated_at: string
}

export type CoreMemoryBlock = {
  level: 'user' | 'project'
  project_id: string | null
  project_name: string | null
  content: string
  char_count: number
  char_limit: number
  revision: number
  updated_at: string
}

export type CoreMemoryResponse = {
  profile: CoreMemoryBlock
  project: CoreMemoryBlock | null
}

export type MemoryEventItem = {
  id: string
  project_id: string | null
  conversation_id: string | null
  run_id: string | null
  memory_id: string | null
  target: 'profile' | 'project' | 'entry'
  action: 'add' | 'update' | 'supersede' | 'delete'
  actor: 'user' | 'agent' | 'review'
  summary: string
  undone_at: string | null
  created_at: string
}

export type SessionSearchResult = {
  message_id: string
  conversation_id: string
  conversation_title: string
  project_id: string | null
  seq: number
  role: 'user' | 'assistant'
  snippet: string
  created_at: string
}

export type MemorySettings = {
  memory_write_mode: 'auto' | 'review'
}

export type UsedMemoryHint = {
  id: string
  kind: string
  level: string
  content: string
  status: string
}

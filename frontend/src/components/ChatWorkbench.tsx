import { useEffect, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { fetchEventSource } from '@microsoft/fetch-event-source'
import {
  Activity,
  Brain,
  Bot,
  Check,
  CheckCircle2,
  Copy,
  FileText,
  Filter,
  MessageSquare,
  Plus,
  Send,
  Sparkles,
  ThumbsDown,
  ThumbsUp,
  Trash2,
  UserRound,
  X,
  Zap,
} from 'lucide-react'
import {
  api,
  getCsrfToken,
  type ChatMessage,
  type CitationItem,
  type Collection,
  type Conversation,
  type MemoryEventItem,
  type Paper,
  type ScopeConfig,
  type ResearchProject,
  type UsedMemoryHint,
  type UsageData,
  type User,
} from '../lib/api'
import { ResearchMemory } from './ResearchMemory'
import { formatCompact, relativeTime } from '../lib/format'
import { useUI } from '../lib/store'
import {
  applySseEvent,
  locationsToHighlight,
  memoryKindLabels,
  splitAnswerWithCitations,
  type PdfHighlight,
  type RunProgressState,
} from '../lib/workbenchHelpers'
import { PdfPreview } from './PdfPreview'
import { SafeMarkdown } from './SafeMarkdown'
import { Button } from './ui/button'
import { Notice } from './ui/primitives'

const nodeLabels: Record<string, string> = {
  load_memory: '加载会话记忆',
  route: '意图路由判断',
  rewrite_query: '改写独立检索词',
  retrieve: '选择论文阅读工具',
  paper_tools: '读取论文证据',
  grade_documents: '相关性评估',
  transform_query: '优化检索词重试',
  generate: '生成带引用回答',
  validate_citations: '引用编号与规则校验',
  verify_grounding: 'LLM 事实依据检查',
  direct_answer: '直接回答',
  insufficient_answer: '证据不足回退',
  safe_answer: '安全回答回退',
  persist: '持久化回答与快照',
}

const samplePrompts = [
  '这篇论文的核心贡献和方法框架是什么？',
  '对比文中的实验结果，主要指标有哪些提升？',
  '论文指出了哪些局限性或未来研究方向？',
]

function cleanSnippet(text: string, maxLen: number): string {
  return text.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim().slice(0, maxLen)
}

function CitationBadge({
  num,
  citation,
  onSelect,
}: {
  num: number
  citation?: CitationItem
  onSelect: (c: CitationItem) => void
}) {
  const [hovered, setHovered] = useState(false)
  if (!citation) {
    return <span className="citation-badge missing">[{num}]</span>
  }
  const pageIdx = citation.locations[0]?.page_idx
  const hasBbox = citation.locations.some((l) => l.bbox && l.bbox.length === 4)

  return (
    <span
      className="citation-badge-wrapper"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <button
        type="button"
        className={`citation-badge ${citation.deleted ? 'deleted' : ''}`}
        data-testid={`citation-badge-${num}`}
        aria-label={`引用 [${num}] ${citation.title}`}
        onClick={() => {
          if (!citation.deleted) onSelect(citation)
        }}
      >
        [{num}]
      </button>
      {hovered && (
        <span className="citation-popover" role="tooltip" data-testid={`citation-popover-${num}`}>
          <strong>
            [{num}] {citation.title}
          </strong>
          {citation.deleted ? (
            <span className="citation-deleted-note">来源论文已删除，快照不可访问</span>
          ) : (
            <>
              <span className="citation-meta">
                {pageIdx !== undefined ? `第 ${pageIdx + 1} 页` : '页码未知'} ·{' '}
                {hasBbox ? '支持坐标框高亮' : '仅页级定位'}
              </span>
              <span className="citation-snippet">{cleanSnippet(citation.text, 220)}</span>
            </>
          )}
        </span>
      )}
    </span>
  )
}

function AnswerContent({
  content,
  citations,
  isDraft,
  onSelectCitation,
}: {
  content: string
  citations: CitationItem[]
  isDraft?: boolean
  onSelectCitation: (c: CitationItem) => void
}) {
  const byNumber = new Map(citations.map((c) => [c.number, c]))
  const segments = splitAnswerWithCitations(content)

  return (
    <div className={`answer-body ${isDraft ? 'is-draft' : ''}`}>
      {isDraft && (
        <div className="draft-indicator" data-testid="draft-indicator">
          待校验草稿 · 正在流式生成…
        </div>
      )}
      <div className="answer-prose" data-testid={isDraft ? 'draft-answer-text' : 'final-answer-text'}>
        {segments.map((seg, idx) =>
          seg.type === 'text' ? (
            <SafeMarkdown key={idx} content={seg.value} className="inline-markdown" />
          ) : (
            <CitationBadge
              key={idx}
              num={seg.number}
              citation={byNumber.get(seg.number)}
              onSelect={onSelectCitation}
            />
          ),
        )}
      </div>
      {citations.length > 0 && (
        <div className="citation-footer-list" data-testid="citation-list">
          <div className="citation-footer-title">参考引用来源（点击跳转 PDF 对应页并高亮）</div>
          <div className="citation-cards">
            {citations.map((c) => {
              const firstPage = c.locations[0]?.page_idx
              const hasBbox = c.locations.some((l) => l.bbox && l.bbox.length === 4)
              return (
                <button
                  key={c.id}
                  type="button"
                  disabled={c.deleted}
                  className={`citation-card-item ${c.deleted ? 'is-deleted' : ''}`}
                  data-testid={`citation-card-${c.number}`}
                  onClick={() => onSelectCitation(c)}
                >
                  <div className="citation-card-top">
                    <span className="citation-num">[{c.number}]</span>
                    <span className="citation-paper-title">{c.title}</span>
                    {!c.deleted && firstPage !== undefined && (
                      <span className="citation-page-tag">
                        第 {firstPage + 1} 页 {hasBbox ? '· 高亮' : '· 跳页'}
                      </span>
                    )}
                  </div>
                  {!c.deleted && <p className="citation-card-snippet">{cleanSnippet(c.text, 140)}</p>}
                </button>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <button
      type="button"
      className="feedback-btn"
      aria-label="复制回答"
      onClick={() => {
        void navigator.clipboard?.writeText(text)
        setCopied(true)
        window.setTimeout(() => setCopied(false), 1500)
      }}
    >
      {copied ? <Check size={13} /> : <Copy size={13} />}
      {copied ? '已复制' : '复制'}
    </button>
  )
}

function MessageMemoryTrace({
  verifyResult,
  userId,
  conversationId,
}: {
  verifyResult: Record<string, unknown>
  userId: string
  conversationId: string | null
}) {
  const client = useQueryClient()
  const [expanded, setExpanded] = useState(false)
  const [undoneIds, setUndoneIds] = useState<string[]>([])
  const [dismissedMemoryIds, setDismissedMemoryIds] = useState<string[]>([])

  const usedMemories = (
    Array.isArray(verifyResult?.used_memories)
      ? (verifyResult.used_memories as UsedMemoryHint[])
      : []
  ).filter((m) => !dismissedMemoryIds.includes(m.id))

  const memoryEvents = Array.isArray(verifyResult?.memory_events)
    ? (verifyResult.memory_events as MemoryEventItem[])
    : []

  const undoMutation = useMutation({
    mutationFn: (eventId: string) => api(`/memory/events/${eventId}/undo`, 'POST'),
    onSuccess: (_, eventId) => {
      setUndoneIds((prev) => [...prev, eventId])
      void client.invalidateQueries({ queryKey: ['memories', userId] })
      void client.invalidateQueries({ queryKey: ['memory-core', userId] })
      if (conversationId) {
        void client.invalidateQueries({ queryKey: ['messages', conversationId] })
      }
    },
  })

  const memoryModify = useMutation({
    mutationFn: ({ id, action }: { id: string; action: 'supersede' | 'delete' }) =>
      action === 'delete'
        ? api(`/memories/${id}`, 'DELETE')
        : api(`/memories/${id}`, 'PATCH', { status: 'superseded' }),
    onSuccess: (_, { id }) => {
      setDismissedMemoryIds((prev) => [...prev, id])
      void client.invalidateQueries({ queryKey: ['memories', userId] })
    },
  })

  if (usedMemories.length === 0 && memoryEvents.length === 0) return null

  return (
    <div className="message-memory-trace" data-testid="message-memory-trace">
      <div className="memory-trace-pills">
        {usedMemories.length > 0 && (
          <button
            type="button"
            className="memory-trace-btn"
            onClick={() => setExpanded((v) => !v)}
          >
            <Brain size={13} />
            参考了 {usedMemories.length} 条研究记忆 {expanded ? '▴' : '▾'}
          </button>
        )}
        {memoryEvents.map((ev) => {
          const isUndone = Boolean(ev.undone_at) || undoneIds.includes(ev.id)
          return (
            <span key={ev.id} className="memory-event-pill" data-testid="memory-event-pill">
              <Sparkles size={12} />
              <span>{isUndone ? `已撤销：${ev.summary}` : ev.summary}</span>
              {!isUndone && (
                <button
                  type="button"
                  className="text-button"
                  disabled={undoMutation.isPending}
                  onClick={() => undoMutation.mutate(ev.id)}
                >
                  撤销
                </button>
              )}
            </span>
          )
        })}
      </div>
      {expanded && usedMemories.length > 0 && (
        <ul className="memory-trace-list" data-testid="used-memories-list">
          {usedMemories.map((m) => (
            <li key={m.id} className="memory-trace-item">
              <span>
                <strong>[{memoryKindLabels[m.kind] ?? m.kind}]</strong> {m.content}
              </span>
              <span className="memory-trace-actions">
                <button
                  type="button"
                  className="text-button"
                  onClick={() => memoryModify.mutate({ id: m.id, action: 'supersede' })}
                >
                  标为已修正
                </button>
                <button
                  type="button"
                  className="text-button"
                  onClick={() => memoryModify.mutate({ id: m.id, action: 'delete' })}
                >
                  删除
                </button>
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

export function ChatWorkbench({
  user,
  initialPaperId,
  onClearInitialPaper,
}: {
  user: User
  initialPaperId?: string | null
  onClearInitialPaper?: () => void
}) {
  const client = useQueryClient()
  const [selectedCid, setSelectedCid] = useState<string | null>(null)
  const [question, setQuestion] = useState('')
  const [mode, setMode] = useState<'fast' | 'agentic'>('agentic')
  const [scopeOpen, setScopeOpen] = useState(false)
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null)
  const [memoryOpen, setMemoryOpen] = useState(false)
  const [memorySource, setMemorySource] = useState<ChatMessage | null>(null)
  const [selectedCollections, setSelectedCollections] = useState<string[]>([])
  const [selectedPapers, setSelectedPapers] = useState<string[]>([])
  const [activeRun, setActiveRun] = useState<RunProgressState | null>(null)
  const [errorBanner, setErrorBanner] = useState('')
  const [pdfCitation, setPdfCitation] = useState<CitationItem | null>(null)
  const [pdfHighlight, setPdfHighlight] = useState<PdfHighlight | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const messagesRef = useRef<HTMLDivElement>(null)

  const conversations = useQuery({
    queryKey: ['conversations', user.id],
    queryFn: () => api<Conversation[]>('/conversations'),
  })

  const projects = useQuery({ queryKey: ['projects', user.id], queryFn: () => api<ResearchProject[]>('/projects') })

  const collections = useQuery({
    queryKey: ['collections', user.id],
    queryFn: () => api<Collection[]>('/collections'),
  })

  const allPapers = useQuery({
    queryKey: ['all-papers', user.id],
    queryFn: () => api<Paper[]>('/papers'),
  })

  const usage = useQuery({
    queryKey: ['usage', user.id],
    queryFn: () => api<UsageData>('/usage'),
  })

  useEffect(() => {
    if (!selectedCid && conversations.data && conversations.data.length > 0 && !initialPaperId) {
      const first = conversations.data[0]
      setSelectedCid(first.id)
      setSelectedProjectId(first.project_id ?? null)
      setSelectedCollections(first.scope?.collection_ids || [])
      setSelectedPapers(first.scope?.paper_ids || [])
    }
  }, [conversations.data, selectedCid, initialPaperId])

  useEffect(() => {
    if (initialPaperId) {
      setSelectedPapers([initialPaperId])
      setSelectedCollections([])
      onClearInitialPaper?.()
    }
  }, [initialPaperId, onClearInitialPaper])

  const messages = useQuery({
    queryKey: ['messages', selectedCid],
    queryFn: () => api<ChatMessage[]>(`/conversations/${selectedCid}/messages`),
    enabled: !!selectedCid,
  })

  useEffect(() => {
    const el = messagesRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [messages.data?.length, activeRun?.draftText, activeRun?.finalText])

  const feedbackMutation = useMutation({
    mutationFn: ({ mid, feedback }: { mid: string; feedback: 'up' | 'down' | null }) =>
      api(`/messages/${mid}/feedback`, 'PATCH', { feedback }),
    onSuccess: () => {
      if (selectedCid) void client.invalidateQueries({ queryKey: ['messages', selectedCid] })
    },
  })

  function selectConversation(conv: Conversation) {
    setSelectedCid(conv.id)
    setSelectedProjectId(conv.project_id ?? null)
    setSelectedCollections(conv.scope?.collection_ids || [])
    setSelectedPapers(conv.scope?.paper_ids || [])
    setErrorBanner('')
  }

  const removeConversation = useMutation({
    mutationFn: (id: string) => api(`/conversations/${id}`, 'DELETE'),
    onSuccess: (_, id) => {
      const remaining = (client.getQueryData<Conversation[]>(['conversations', user.id]) || [])
        .filter((conv) => conv.id !== id)
      client.setQueryData(['conversations', user.id], remaining)
      client.removeQueries({ queryKey: ['messages', id], exact: true })
      if (selectedCid === id) {
        abortRef.current?.abort()
        setActiveRun(null)
        setPdfCitation(null)
        setPdfHighlight(null)
        setSelectedCid(null)
        setSelectedProjectId(null)
        setSelectedCollections([])
        setSelectedPapers([])
        if (remaining[0]) selectConversation(remaining[0])
      }
      setErrorBanner('')
      void client.invalidateQueries({ queryKey: ['conversations', user.id] })
    },
    onError: (error) => setErrorBanner(error.message),
  })

  function currentScopePayload(): ScopeConfig {
    const scope: ScopeConfig = {}
    if (selectedCollections.length > 0) scope.collection_ids = selectedCollections
    if (selectedPapers.length > 0) scope.paper_ids = selectedPapers
    return scope
  }

  async function createNewConversation(customScope?: ScopeConfig): Promise<Conversation> {
    const scope = customScope ?? currentScopePayload()
    const created = await api<Conversation>('/conversations', 'POST', {
      title: '新会话',
      scope,
      project_id: selectedProjectId,
    })
    await client.invalidateQueries({ queryKey: ['conversations', user.id] })
    setSelectedCid(created.id)
    return created
  }

  async function selectProject(id: string | null) {
    if (selectedCid) await api(`/conversations/${selectedCid}`, 'PATCH', { project_id: id })
    setSelectedProjectId(id)
    await client.invalidateQueries({ queryKey: ['conversations', user.id] })
    void client.invalidateQueries({ queryKey: ['conversation-memory', selectedCid] })
    void client.invalidateQueries({ queryKey: ['memories', user.id] })
  }

  function handleSelectCitation(c: CitationItem) {
    if (c.deleted) return
    setPdfCitation(c)
    setPdfHighlight(
      locationsToHighlight(c.paper_id, c.locations, undefined, `引用 [${c.number}]`),
    )
  }

  async function subscribeRunEvents(
    cid: string,
    body: { idempotency_key: string; question: string; mode: 'fast' | 'agentic'; scope: ScopeConfig },
  ) {
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    const csrfToken = await getCsrfToken()
    let lastEventId = 0

    setActiveRun({
      runId: body.idempotency_key,
      question: body.question,
      mode: body.mode,
      status: 'streaming',
      steps: [],
      rewrites: [],
      retrievalCount: 0,
      generationCount: 0,
      degraded: [],
      sources: [],
      draftText: '',
      finalText: null,
      finalStatus: null,
      verify: null,
      citations: [],
      usageTokens: null,
      usageEstimated: false,
      error: null,
      seenEventIds: [],
    })

    await fetchEventSource(`/api/v1/conversations/${cid}/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRF-Token': csrfToken,
        ...(lastEventId > 0 ? { 'Last-Event-ID': String(lastEventId) } : {}),
      },
      credentials: 'same-origin',
      body: JSON.stringify(body),
      signal: controller.signal,
      openWhenHidden: true,
      async onopen(response) {
        if (response.ok) return
        if (response.status === 401) {
          useUI.getState().setSessionExpired('会话已过期，请重新登录以继续提问。')
          throw new Error('会话已过期，请重新登录')
        }
        const errData = await response.json().catch(() => ({ detail: '提交问答请求失败' }))
        const msg = typeof errData.detail === 'string' ? errData.detail : '请求失败，请检查输入或配额'
        throw new Error(msg)
      },
      onmessage(ev) {
        if (!ev.event || !ev.data) return
        try {
          const parsed = JSON.parse(ev.data) as Record<string, unknown>
          if (typeof parsed.id === 'number') {
            lastEventId = Math.max(lastEventId, parsed.id)
          }
          setActiveRun((prev) => (prev ? applySseEvent(prev, ev.event, parsed) : prev))
          if (ev.event === 'done' || ev.event === 'error') {
            void client.invalidateQueries({ queryKey: ['messages', cid] })
            void client.invalidateQueries({ queryKey: ['conversations', user.id] })
            void client.invalidateQueries({ queryKey: ['usage', user.id] })
          }
        } catch {
          // Ignore malformed keepalive frames
        }
      },
      onerror(err) {
        const message = err instanceof Error ? err.message : '流式连接中断'
        setErrorBanner(message)
        setActiveRun((prev) =>
          prev ? { ...prev, status: 'failed', error: message } : null,
        )
        throw err
      },
    })
  }

  async function handleSubmitQuestion(e: React.FormEvent) {
    e.preventDefault()
    const trimmed = question.trim()
    if (!trimmed) return
    setErrorBanner('')

    try {
      const scope = currentScopePayload()
      let cid = selectedCid
      if (!cid) {
        const created = await createNewConversation(scope)
        cid = created.id
      }
      const idempotencyKey = `run-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
      setQuestion('')
      await subscribeRunEvents(cid, {
        idempotency_key: idempotencyKey,
        question: trimmed,
        mode,
        scope,
      })
    } catch (err) {
      if (err instanceof Error && err.name !== 'AbortError') {
        setErrorBanner(err.message)
      }
    }
  }

  const readyPapers = (allPapers.data || []).filter((p) => p.active_version_id)
  const scopeSummary =
    selectedPapers.length > 0
      ? `已选 ${selectedPapers.length} 篇论文`
      : selectedCollections.length > 0
        ? `已选 ${selectedCollections.length} 个知识库`
        : '全部已入库论文'

  return (
    <section
      className={`chat-workbench ${pdfCitation ? 'with-pdf-drawer' : ''}`}
      data-testid="chat-workbench"
    >
      {memoryOpen && <ResearchMemory userId={user.id} conversationId={selectedCid} projectId={selectedProjectId} sourceMessage={memorySource} onClose={() => setMemoryOpen(false)} onProjectSelected={selectProject} />}
      <aside className="chat-conversations-sidebar">
        <div className="chat-sidebar-top">
          <h2>问答会话</h2>
          <Button
            variant="outline"
            size="sm"
            type="button"
            onClick={() => void createNewConversation()}
          >
            <Plus size={14} /> 新建会话
          </Button>
        </div>

        <div className="chat-project-selector">
          <label htmlFor="chat-project">研究项目</label>
          <select id="chat-project" value={selectedProjectId ?? ''} disabled={activeRun?.status === 'streaming'} onChange={e => { void selectProject(e.target.value || null).catch(e => setErrorBanner(e.message)) }}>
            <option value="">独立会话</option>
            {projects.data?.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}
          </select>
          <small className="muted">切换项目会重置当前会话的研究上下文。</small>
          <Button variant="outline" size="sm" onClick={() => { setMemorySource(null); setMemoryOpen(true) }}><Brain size={14} />研究记忆</Button>
        </div>
        <div className="conversation-list">
          {conversations.data?.map((conv) => (
            <div
              key={conv.id}
              className={`conversation-item ${selectedCid === conv.id ? 'active' : ''}`}
            >
              <button
                type="button"
                className="conversation-select"
                aria-pressed={selectedCid === conv.id}
                onClick={() => selectConversation(conv)}
              >
                <MessageSquare size={15} />
                <span className="conversation-item-text">
                  <strong>{conv.title}</strong>
                  <span>{relativeTime(conv.created_at)}</span>
                </span>
              </button>
              <button
                type="button"
                className="icon-button conversation-delete"
                aria-label={`删除会话 ${conv.title}`}
                title="删除会话"
                disabled={removeConversation.isPending}
                onClick={() => {
                  if (confirm(`确定删除会话“${conv.title}”？删除后将无法查看此会话的聊天记录。`)) {
                    removeConversation.mutate(conv.id)
                  }
                }}
              >
                <Trash2 size={15} />
              </button>
            </div>
          ))}
          {conversations.data?.length === 0 && (
            <p className="muted empty-conv-hint">输入问题自动开启你的第一个研究问答会话。</p>
          )}
        </div>

        {usage.data && (
          <div className="chat-quota-box">
            <span>今日模型额度</span>
            <strong>
              {formatCompact(usage.data.model_tokens)} / {formatCompact(usage.data.limits.daily_model_tokens)}
            </strong>
          </div>
        )}
      </aside>

      <div className="chat-center-column">
        <div className="chat-controls-bar">
          <div className="mode-toggle" role="group" aria-label="问答模式">
            <button
              type="button"
              className={`mode-btn ${mode === 'fast' ? 'active' : ''}`}
              onClick={() => setMode('fast')}
            >
              <Zap size={14} /> Fast 快速问答
            </button>
            <button
              type="button"
              className={`mode-btn ${mode === 'agentic' ? 'active' : ''}`}
              onClick={() => setMode('agentic')}
            >
              <Sparkles size={14} /> Agentic 深度核验
            </button>
          </div>

          <button
            type="button"
            className={`scope-selector-btn ${scopeOpen ? 'open' : ''}`}
            onClick={() => setScopeOpen((o) => !o)}
          >
            <Filter size={14} />
            检索范围：{scopeSummary}
          </button>
        </div>

        {scopeOpen && (
          <div className="scope-picker-drawer" data-testid="scope-picker">
            <div className="scope-picker-header">
              <strong>限定本轮问答的知识库或论文范围（不勾选默认检索全部已入库论文）</strong>
              <button type="button" className="text-button" onClick={() => { setSelectedCollections([]); setSelectedPapers([]) }}>
                重置为全部
              </button>
            </div>
            <div className="scope-picker-columns">
              <div>
                <h4>按知识库筛选</h4>
                {collections.data?.map((col) => (
                  <label key={col.id} className="scope-checkbox">
                    <input
                      type="checkbox"
                      checked={selectedCollections.includes(col.id)}
                      onChange={(e) =>
                        setSelectedCollections((prev) =>
                          e.target.checked ? [...prev, col.id] : prev.filter((id) => id !== col.id),
                        )
                      }
                    />
                    <span>{col.name}</span>
                  </label>
                ))}
                {!collections.data?.length && <p className="muted">暂无知识库</p>}
              </div>
              <div>
                <h4>按单篇论文筛选（已入库 {readyPapers.length} 篇）</h4>
                {readyPapers.map((p) => (
                  <label key={p.id} className="scope-checkbox">
                    <input
                      type="checkbox"
                      checked={selectedPapers.includes(p.id)}
                      onChange={(e) =>
                        setSelectedPapers((prev) =>
                          e.target.checked ? [...prev, p.id] : prev.filter((id) => id !== p.id),
                        )
                      }
                    />
                    <span>{p.title}</span>
                  </label>
                ))}
                {readyPapers.length === 0 && <p className="muted">暂无已入库论文</p>}
              </div>
            </div>
          </div>
        )}

        {errorBanner && <Notice tone="error" testId="chat-error-banner">{errorBanner}</Notice>}

        <div className="chat-messages-area" ref={messagesRef}>
          {messages.data?.map((msg) => (
            <article key={msg.id} className={`chat-bubble ${msg.role}`}>
              <div className="chat-bubble-meta">
                <span className="bubble-avatar" aria-hidden="true">
                  {msg.role === 'user' ? <UserRound size={13} /> : <Bot size={13} />}
                </span>
                <strong>{msg.role === 'user' ? '我的提问' : 'PaperMind 回答'}</strong>
                {msg.rewritten_query && msg.rewritten_query !== msg.content && (
                  <span className="rewritten-tag">改写检索：{msg.rewritten_query}</span>
                )}
                {msg.role === 'assistant' && Boolean(msg.verify_result?.grounding) && (
                  <span className="verify-tag">
                    {msg.verify_result.grounding === 'passed'
                      ? '事实依据核验通过'
                      : msg.verify_result.grounding === 'not_run'
                        ? 'Fast 模式（未运行 LLM 核验）'
                        : String(msg.verify_result.grounding)}
                  </span>
                )}
              </div>
              {msg.role === 'user' ? (
                <p className="user-question-text">{msg.content}</p>
              ) : (
                <>
                  <AnswerContent
                    content={msg.content}
                    citations={msg.citations}
                    onSelectCitation={handleSelectCitation}
                  />
                  <MessageMemoryTrace
                    verifyResult={msg.verify_result}
                    userId={user.id}
                    conversationId={selectedCid}
                  />
                  <div className="message-feedback-bar">
                    <button
                      type="button"
                      aria-label="回答有帮助"
                      className={`feedback-btn ${msg.feedback === 'up' ? 'active' : ''}`}
                      onClick={() =>
                        feedbackMutation.mutate({
                          mid: msg.id,
                          feedback: msg.feedback === 'up' ? null : 'up',
                        })
                      }
                    >
                      <ThumbsUp size={13} /> 有帮助
                    </button>
                    <button
                      type="button"
                      aria-label="回答需改进"
                      className={`feedback-btn ${msg.feedback === 'down' ? 'active' : ''}`}
                      onClick={() =>
                        feedbackMutation.mutate({
                          mid: msg.id,
                          feedback: msg.feedback === 'down' ? null : 'down',
                        })
                      }
                    >
                      <ThumbsDown size={13} /> 需改进
                    </button>
                    <CopyButton text={msg.content} />
                    <button type="button" className="feedback-btn" onClick={() => { setMemorySource(msg); setMemoryOpen(true) }}><Brain size={13} />保存为研究记忆</button>
                  </div>
                </>
              )}
            </article>
          ))}

          {activeRun && activeRun.status === 'streaming' && (
            <article className="chat-bubble assistant streaming-bubble" data-testid="streaming-bubble">
              <div className="chat-bubble-meta">
                <span className="bubble-avatar" aria-hidden="true"><Bot size={13} /></span>
                <strong>PaperMind 正在回答（{activeRun.mode}）</strong>
                {activeRun.generationCount > 1 && (
                  <span className="regenerate-badge" data-testid="regenerate-badge">
                    第 {activeRun.generationCount} 次生成（已替换前序草稿）
                  </span>
                )}
              </div>
              <AnswerContent
                content={activeRun.finalText ?? activeRun.draftText}
                citations={activeRun.citations}
                isDraft={activeRun.finalText === null}
                onSelectCitation={handleSelectCitation}
              />
            </article>
          )}

          {!messages.data?.length && !activeRun && (
            <div className="chat-empty-welcome">
              <span className="welcome-badge"><Sparkles size={22} /></span>
              <h3>开始基于文献证据的多轮问答</h3>
              <p className="muted">
                每一句事实结论均附带 [n] 引用徽章；悬停可预览原文快照，点击即可在右侧打开原 PDF 跳页并高亮坐标区域。
              </p>
              {user.email_verified_at && (
                <div className="prompt-suggestions">
                  {samplePrompts.map((prompt) => (
                    <button
                      key={prompt}
                      type="button"
                      className="prompt-pill"
                      onClick={() => setQuestion(prompt)}
                    >
                      {prompt}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        <form className="chat-input-form" onSubmit={(e) => void handleSubmitQuestion(e)}>
          <textarea
            aria-label="提问内容"
            placeholder={
              user.email_verified_at
                ? '输入你的研究问题，支持多轮追问（Enter 发送，Shift + Enter 换行）'
                : '请先验证邮箱后再发起问答'
            }
            disabled={!user.email_verified_at || activeRun?.status === 'streaming'}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                if (question.trim()) void handleSubmitQuestion(e)
              }
            }}
          />
          <Button
            disabled={
              !user.email_verified_at || !question.trim() || activeRun?.status === 'streaming'
            }
          >
            <Send size={16} />
            {activeRun?.status === 'streaming' ? '回答中…' : '发送提问'}
          </Button>
        </form>
      </div>

      <aside className="chat-inspector-sidebar">
        {pdfCitation ? (
          <div className="chat-pdf-drawer" data-testid="chat-pdf-drawer">
            <div className="chat-pdf-drawer-header">
              <div>
                <span className="eyebrow">CITATION SOURCE [{pdfCitation.number}]</span>
                <h4>{pdfCitation.title}</h4>
              </div>
              <button
                type="button"
                className="icon-button"
                aria-label="关闭 PDF 预览"
                onClick={() => {
                  setPdfCitation(null)
                  setPdfHighlight(null)
                }}
              >
                <X size={16} />
              </button>
            </div>
            <PdfPreview
              paperId={pdfCitation.paper_id}
              highlight={pdfHighlight}
              allLocations={pdfCitation.locations}
              onSelectPageIdx={(pageIdx) => {
                setPdfHighlight(
                  locationsToHighlight(
                    pdfCitation.paper_id,
                    pdfCitation.locations,
                    pageIdx,
                    `引用 [${pdfCitation.number}]`,
                  ),
                )
              }}
            />
          </div>
        ) : (
          <div className="progress-panel" data-testid="progress-panel">
            <div className="progress-panel-header">
              <Activity size={17} />
              <h3>检索与校验进度面板</h3>
            </div>
            {activeRun ? (
              <div className="progress-details">
                <div className="progress-stat-grid">
                  <div>
                    <span>模式</span>
                    <strong>{activeRun.mode === 'agentic' ? 'Agentic' : 'Fast'}</strong>
                  </div>
                  <div>
                    <span>检索次数</span>
                    <strong data-testid="progress-retrieval-count">{activeRun.retrievalCount} / 3</strong>
                  </div>
                  <div>
                    <span>生成版本</span>
                    <strong data-testid="progress-generation-count">{activeRun.generationCount || 1} / 2</strong>
                  </div>
                </div>

                {activeRun.rewrites.length > 0 && (
                  <div className="progress-section">
                    <h4>改写检索词</h4>
                    {activeRun.rewrites.map((rw, i) => (
                      <p key={i} className="rewrite-item" data-testid="progress-rewrite-query">
                        {rw}
                      </p>
                    ))}
                  </div>
                )}

                <div className="progress-section">
                  <h4>节点执行轨迹</h4>
                  <ul className="step-timeline" data-testid="progress-steps">
                    {activeRun.steps.map((node, i) => (
                      <li key={`${node}-${i}`}>
                        <CheckCircle2 size={13} />
                        <span>{nodeLabels[node] || node}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {activeRun.verify && (
                  <div className="progress-section" data-testid="progress-verify">
                    <h4>校验结论</h4>
                    <p className="verify-summary">
                      引用规则：{activeRun.verify.citations_valid ? '通过' : '未通过（触发重生成或回退）'}
                    </p>
                    <p className="verify-summary">
                      事实依据：
                      {activeRun.verify.grounding === 'passed'
                        ? 'LLM 核验通过'
                        : activeRun.verify.grounding === 'not_run'
                          ? 'Fast 模式未执行 LLM 核验'
                          : activeRun.verify.grounding === 'failed'
                            ? '未通过依据检查'
                            : activeRun.verify.grounding || '检查中'}
                    </p>
                  </div>
                )}

                {activeRun.sources.length > 0 && (
                  <div className="progress-section">
                    <h4>本轮候选来源（{activeRun.sources.length}）</h4>
                    <div className="candidate-sources-list">
                      {activeRun.sources.map((s) => (
                        <div key={s.chunk_id} className="candidate-source-card">
                          <div className="candidate-source-head">
                            <FileText size={13} />
                            <strong>
                              [{s.number}] {s.title}
                            </strong>
                          </div>
                          <p>{cleanSnippet(s.text, 100)}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <p className="muted progress-empty">
                发起提问后，这里将实时展示 LangGraph 节点进度、问题改写、检索次数、候选来源及引用/依据核验结果。点击回答中的引用徽章将在此处展开原 PDF 并高亮对应坐标框。
              </p>
            )}
          </div>
        )}
      </aside>
    </section>
  )
}

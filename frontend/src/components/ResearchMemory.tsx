import { useEffect, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Brain,
  Check,
  ExternalLink,
  History,
  Pin,
  RotateCcw,
  Search,
  Sparkles,
  Trash2,
} from 'lucide-react'
import {
  api,
  type ChatMessage,
  type ConversationMemory,
  type CoreMemoryResponse,
  type MemoryEntry,
  type MemoryEventItem,
  type MemorySettings,
  type Paper,
  type ResearchProject,
  type SessionSearchResult,
  type User,
} from '../lib/api'
import { Modal } from './ui/modal'
import { Button } from './ui/button'
import { Meter, Notice, PageHeader } from './ui/primitives'

const kinds: Record<string, string> = {
  goal: '研究目标',
  constraint: '研究约束',
  focus: '阅读焦点',
  discussed: '已讨论事项',
  open_question: '待解决问题',
  correction: '用户纠正',
  next_step: '下一步',
  hypothesis: '研究假设',
  conclusion: '研究判断',
  preference: '个人偏好',
  note: '笔记',
}
const levels = {
  user: '个人（跨项目）',
  project: '当前项目',
  paper: '指定论文',
  conversation: '当前会话',
}
const statuses = {
  candidate: '待确认',
  active: '已确认',
  superseded: '已被修正',
}
const evidenceLabels = {
  current: '来源版本有效',
  outdated: '来源已更新，需重新核对',
  unavailable: '来源不可用',
  unlinked: '未关联论文证据',
}

function stripCitationMarkers(text: string): string {
  return text.replace(/\[\d+\]/g, '').replace(/\s+/g, ' ').trim().slice(0, 2000)
}

type Props = {
  userId: string
  conversationId: string | null
  projectId: string | null
  sourceMessage?: ChatMessage | null
  onClose: () => void
  onProjectSelected: (id: string | null) => Promise<void>
}

export function ResearchMemory({
  userId,
  conversationId,
  projectId,
  sourceMessage,
  onClose,
  onProjectSelected,
}: Props) {
  const client = useQueryClient()
  const [content, setContent] = useState(
    sourceMessage ? stripCitationMarkers(sourceMessage.content) : '',
  )
  const [level, setLevel] = useState<MemoryEntry['level']>(
    projectId ? 'project' : conversationId ? 'conversation' : 'user',
  )
  const [kind, setKind] = useState(sourceMessage ? 'conclusion' : 'goal')
  const [epistemic, setEpistemic] = useState<MemoryEntry['epistemic']>(
    sourceMessage?.citations.length ? 'evidence_linked' : 'user_statement',
  )
  const [citationIds, setCitationIds] = useState(
    sourceMessage?.citations.filter((c) => !c.deleted).map((c) => c.id) ?? [],
  )
  const [paperId, setPaperId] = useState('')
  const [projectName, setProjectName] = useState('')
  const [projectDescription, setProjectDescription] = useState('')
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [sourceId, setSourceId] = useState<string | null>(null)
  const [editing, setEditing] = useState<MemoryEntry | null>(null)
  const [editText, setEditText] = useState('')
  const [summaryEdit, setSummaryEdit] = useState<
    Pick<ConversationMemory, 'summary' | 'research_state'> | null
  >(null)
  const [sourceLinked, setSourceLinked] = useState(Boolean(sourceMessage))
  const [searchQuery, setSearchQuery] = useState('')
  const [filterKind, setFilterKind] = useState('')
  const [filterStatus, setFilterStatus] = useState('')
  const [selectedIds, setSelectedIds] = useState<string[]>([])
  const [confirmDeleteProject, setConfirmDeleteProject] = useState(false)
  const [confirmClearSession, setConfirmClearSession] = useState(false)
  const [profileDraft, setProfileDraft] = useState('')
  const [projectCoreDraft, setProjectCoreDraft] = useState('')
  const [showCoreEditor, setShowCoreEditor] = useState(false)

  const projects = useQuery({
    queryKey: ['projects', userId],
    queryFn: () => api<ResearchProject[]>('/projects'),
  })
  const papers = useQuery({
    queryKey: ['all-papers', userId],
    queryFn: () => api<Paper[]>('/papers'),
  })
  const memories = useQuery({
    queryKey: ['memories', userId, conversationId, projectId],
    queryFn: () =>
      api<MemoryEntry[]>(
        `/memories${
          conversationId
            ? `?conversation_id=${conversationId}`
            : projectId
              ? `?project_id=${projectId}`
              : ''
        }`,
      ),
  })
  const coreMemory = useQuery({
    queryKey: ['memory-core', userId, projectId],
    queryFn: () =>
      api<CoreMemoryResponse>(
        `/memory/core${projectId ? `?project_id=${projectId}` : ''}`,
      ),
  })
  const source = useQuery({
    queryKey: ['memory-source', sourceId],
    queryFn: () =>
      api<{ role: string; content: string; seq: number }>(`/memories/${sourceId}/source`),
    enabled: !!sourceId,
  })
  const state = useQuery({
    queryKey: ['conversation-memory', conversationId],
    queryFn: () => api<ConversationMemory>(`/conversations/${conversationId}/memory`),
    enabled: !!conversationId,
  })

  useEffect(() => {
    if (coreMemory.data) {
      setProfileDraft(coreMemory.data.profile.content)
      setProjectCoreDraft(coreMemory.data.project?.content ?? '')
    }
  }, [coreMemory.data])

  function refresh() {
    void client.invalidateQueries({ queryKey: ['memories', userId] })
    void client.invalidateQueries({ queryKey: ['conversation-memory', conversationId] })
    void client.invalidateQueries({ queryKey: ['memory-core', userId] })
    void client.invalidateQueries({ queryKey: ['memory-events', userId] })
  }

  const change = useMutation({
    mutationFn: async ({
      path,
      method = 'PATCH',
      body,
    }: {
      path: string
      method?: string
      body?: unknown
    }) => api(path, method, body),
    onSuccess: () => {
      refresh()
      setError('')
      setEditing(null)
      setSummaryEdit(null)
    },
    onError: (e) => setError(e.message),
  })

  const bulkAction = useMutation({
    mutationFn: (action: 'confirm' | 'supersede' | 'delete' | 'pin' | 'unpin') =>
      api('/memories/bulk', 'POST', { ids: selectedIds, action }),
    onSuccess: () => {
      setSelectedIds([])
      setNotice('已完成批量操作')
      refresh()
    },
    onError: (e) => setError(e.message),
  })

  const cleanupDiscussed = useMutation({
    mutationFn: () =>
      api<{ cleaned: number }>(
        `/memories/cleanup-discussed${projectId ? `?project_id=${projectId}` : ''}`,
        'POST',
      ),
    onSuccess: (res) => {
      setNotice(`已清理 ${res.cleaned} 条未确认的提问记录`)
      refresh()
    },
    onError: (e) => setError(e.message),
  })

  const saveCore = useMutation({
    mutationFn: ({ targetLevel, text }: { targetLevel: 'user' | 'project'; text: string }) =>
      api(
        `/memory/core?level=${targetLevel}${
          targetLevel === 'project' && projectId ? `&project_id=${projectId}` : ''
        }`,
        'PUT',
        { content: text },
      ),
    onSuccess: () => {
      setNotice('核心记忆（冻结快照）已更新，将在下一轮问答生效')
      refresh()
    },
    onError: (e) => setError(e.message),
  })

  const save = useMutation({
    mutationFn: () =>
      api('/memories', 'POST', {
        level,
        project_id: level === 'user' ? null : projectId,
        conversation_id: level === 'conversation' ? conversationId : null,
        paper_id: level === 'paper' ? paperId : null,
        kind,
        content,
        epistemic,
        status: 'active',
        source_message_id: sourceLinked ? sourceMessage?.id : null,
        citation_ids: sourceLinked ? citationIds : [],
      }),
    onSuccess: () => {
      setContent('')
      setError('')
      setNotice('研究记忆已保存')
      refresh()
    },
    onError: (e) => setError(e.message),
  })

  const createProject = useMutation({
    mutationFn: () =>
      api<ResearchProject>('/projects', 'POST', {
        name: projectName,
        description: projectDescription,
      }),
    onSuccess: async (p) => {
      await client.invalidateQueries({ queryKey: ['projects', userId] })
      setProjectName('')
      setProjectDescription('')
      setLevel('project')
      try {
        await onProjectSelected(p.id)
      } catch (e) {
        setError((e as Error).message)
      }
    },
    onError: (e) => setError(e.message),
  })

  const currentProject = projects.data?.find((p) => p.id === projectId)
  const filteredMemories = (memories.data ?? []).filter((item) => {
    if (filterKind && item.kind !== filterKind) return false
    if (filterStatus && item.status !== filterStatus) return false
    if (
      searchQuery.trim() &&
      !item.content.toLowerCase().includes(searchQuery.trim().toLowerCase())
    ) {
      return false
    }
    return true
  })
  const discussedCandidatesCount = (memories.data ?? []).filter(
    (m) => m.kind === 'discussed' && m.status === 'candidate',
  ).length

  return (
    <Modal
      title="研究项目与记忆"
      description="管理核心备忘（Hermes 风格冻结快照）、待验证判断和阅读进度。记忆用于延续研究，论文事实仍需重新读取原文。"
      onClose={onClose}
      size="lg"
      labelId="research-memory-title"
    >
      <div className="research-memory-panel">
        <div className="memory-modal-Banner">
          <span className="muted">
            核心记忆在会话开始时以冻结快照注入；支持按需检索历史会话与一键撤销。
          </span>
          <a
            href="#/memory"
            className="text-button"
            onClick={onClose}
            data-testid="open-memory-center-link"
          >
            <ExternalLink size={13} /> 打开完整记忆中心
          </a>
        </div>

        {error && (
          <p className="error-text" role="alert">
            {error}
          </p>
        )}
        {notice && <Notice tone="success">{notice}</Notice>}
        {(projects.error || memories.error || state.error || papers.error) && (
          <p role="alert">记忆加载失败，请关闭后重试。</p>
        )}

        <section className="memory-section-card">
          <div className="memory-section-header">
            <h3>研究项目与核心备忘（MEMORY.md / USER.md）</h3>
            <button
              type="button"
              className="text-button"
              onClick={() => setShowCoreEditor((v) => !v)}
            >
              {showCoreEditor ? '收起核心备忘编辑器' : '编辑核心画像与备忘'}
            </button>
          </div>
          <label className="field">
            当前项目
            <select
              aria-label="当前研究项目"
              value={projectId ?? ''}
              onChange={(e) => {
                void onProjectSelected(e.target.value || null).catch((err) =>
                  setError(err.message),
                )
              }}
            >
              <option value="">独立会话</option>
              {projects.data?.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
          </label>
          <p className="muted">
            同一项目的新会话会加载相关记忆。切换项目将重置本会话的上下文，聊天记录仍可查看。
          </p>

          {showCoreEditor && coreMemory.data && (
            <div className="core-memory-editors" data-testid="core-memory-editors">
              <div className="core-block-card">
                <div className="core-block-head">
                  <strong>研究者画像（USER.md · 跨项目常驻）</strong>
                  <span className="muted">
                    {profileDraft.length} / {coreMemory.data.profile.char_limit} 字
                  </span>
                </div>
                <textarea
                  aria-label="研究者画像内容"
                  rows={3}
                  maxLength={coreMemory.data.profile.char_limit}
                  placeholder="- 研究方向：例如多模态大模型与参数高效微调&#10;- 回答偏好：先给核心结论，再列出关键实验对比表"
                  value={profileDraft}
                  onChange={(e) => setProfileDraft(e.target.value)}
                />
                <Button
                  variant="outline"
                  size="sm"
                  type="button"
                  disabled={saveCore.isPending}
                  onClick={() =>
                    saveCore.mutate({ targetLevel: 'user', text: profileDraft })
                  }
                >
                  保存研究者画像
                </Button>
              </div>

              {projectId && coreMemory.data.project && (
                <div className="core-block-card">
                  <div className="core-block-head">
                    <strong>
                      项目核心备忘（MEMORY.md · {coreMemory.data.project.project_name}）
                    </strong>
                    <span className="muted">
                      {projectCoreDraft.length} / {coreMemory.data.project.char_limit} 字
                    </span>
                  </div>
                  <textarea
                    aria-label="项目核心备忘内容"
                    rows={4}
                    maxLength={coreMemory.data.project.char_limit}
                    placeholder="- 核心研究问题：…&#10;- 已排除方向：…&#10;- 关键实验约束：…"
                    value={projectCoreDraft}
                    onChange={(e) => setProjectCoreDraft(e.target.value)}
                  />
                  <Button
                    variant="outline"
                    size="sm"
                    type="button"
                    disabled={saveCore.isPending}
                    onClick={() =>
                      saveCore.mutate({ targetLevel: 'project', text: projectCoreDraft })
                    }
                  >
                    保存项目核心备忘
                  </Button>
                </div>
              )}
            </div>
          )}

          {currentProject && (
            <div className="memory-project-actions">
              <p>{currentProject.description}</p>
              <Button
                variant="outline"
                onClick={() => {
                  setProjectName(currentProject.name)
                  setProjectDescription(currentProject.description)
                }}
              >
                载入项目信息
              </Button>
              <Button
                variant="outline"
                disabled={!projectName.trim() || change.isPending}
                onClick={() =>
                  change.mutate(
                    {
                      path: `/projects/${projectId}`,
                      body: { name: projectName, description: projectDescription },
                    },
                    {
                      onSuccess: () => {
                        void client.invalidateQueries({ queryKey: ['projects', userId] })
                      },
                    },
                  )
                }
              >
                更新当前项目
              </Button>
              {!confirmDeleteProject ? (
                <Button
                  variant="outline"
                  onClick={() => setConfirmDeleteProject(true)}
                >
                  删除项目
                </Button>
              ) : (
                <span className="inline-confirm-group">
                  <Button
                    variant="outline"
                    onClick={() => {
                      setConfirmDeleteProject(false)
                      change.mutate(
                        { path: `/projects/${projectId}`, method: 'DELETE' },
                        {
                          onSuccess: () => {
                            void client.invalidateQueries({
                              queryKey: ['projects', userId],
                            })
                            void onProjectSelected(null).catch((err) =>
                              setError(err.message),
                            )
                          },
                        },
                      )
                    }}
                  >
                    确认删除项目
                  </Button>
                  <button
                    type="button"
                    className="text-button"
                    onClick={() => setConfirmDeleteProject(false)}
                  >
                    取消
                  </button>
                </span>
              )}
            </div>
          )}
          <form
            className="memory-form"
            onSubmit={(e) => {
              e.preventDefault()
              createProject.mutate()
            }}
          >
            <label className="field">
              项目名称
              <input
                aria-label="项目名称"
                value={projectName}
                maxLength={100}
                required
                onChange={(e) => setProjectName(e.target.value)}
              />
            </label>
            <label className="field">
              项目描述
              <textarea
                aria-label="项目描述"
                value={projectDescription}
                maxLength={2000}
                onChange={(e) => setProjectDescription(e.target.value)}
              />
            </label>
            <Button type="submit" disabled={createProject.isPending}>
              新建并使用项目
            </Button>
          </form>
        </section>

        {state.data && (
          <section className="memory-section-card">
            <h3>当前会话的研究状态</h3>
            <p>
              {state.data.summary || '尚未生成会话摘要；近期对话会直接用于追问。'}
            </p>
            {Object.entries(state.data.research_state).map(
              ([key, items]) =>
                items.length > 0 && (
                  <div key={key} className="memory-state-group">
                    <strong>{kinds[key] ?? key}</strong>
                    <ul>
                      {items.map((item, i) => (
                        <li key={`${item.source_seq}-${i}`}>
                          {item.content} <small>来源：消息 {item.source_seq}</small>
                        </li>
                      ))}
                    </ul>
                  </div>
                ),
            )}
            <div className="memory-actions">
              <Button
                variant="outline"
                onClick={() =>
                  setSummaryEdit(
                    structuredClone({
                      summary: state.data.summary,
                      research_state: state.data.research_state,
                    }),
                  )
                }
              >
                编辑会话状态
              </Button>
              {!confirmClearSession ? (
                <Button
                  variant="outline"
                  onClick={() => setConfirmClearSession(true)}
                >
                  清除会话记忆
                </Button>
              ) : (
                <span className="inline-confirm-group">
                  <Button
                    variant="outline"
                    onClick={() => {
                      setConfirmClearSession(false)
                      change.mutate({
                        path: `/conversations/${conversationId}/memory`,
                        method: 'DELETE',
                      })
                    }}
                  >
                    确认清除会话记忆
                  </Button>
                  <button
                    type="button"
                    className="text-button"
                    onClick={() => setConfirmClearSession(false)}
                  >
                    取消
                  </button>
                </span>
              )}
            </div>
            {summaryEdit !== null && (
              <form
                onSubmit={(e) => {
                  e.preventDefault()
                  change.mutate({
                    path: `/conversations/${conversationId}/memory`,
                    body: summaryEdit,
                  })
                }}
              >
                <label className="field">
                  会话摘要
                  <textarea
                    aria-label="会话摘要"
                    rows={4}
                    maxLength={2400}
                    value={summaryEdit.summary}
                    onChange={(e) =>
                      setSummaryEdit({ ...summaryEdit, summary: e.target.value })
                    }
                  />
                </label>
                {Object.entries(summaryEdit.research_state).map(([key, items]) => (
                  <div key={key}>
                    {items.map((item, index) => (
                      <div key={`${key}-${index}`} className="field">
                        <label>
                          {kinds[key] ?? key}
                          <input
                            aria-label={`编辑${kinds[key] ?? key}${index + 1}`}
                            value={item.content}
                            maxLength={300}
                            required
                            onChange={(e) =>
                              setSummaryEdit({
                                ...summaryEdit,
                                research_state: {
                                  ...summaryEdit.research_state,
                                  [key]: items.map((v, i) =>
                                    i === index ? { ...v, content: e.target.value } : v,
                                  ),
                                },
                              })
                            }
                          />
                        </label>
                        <button
                          type="button"
                          className="text-button"
                          onClick={() =>
                            setSummaryEdit({
                              ...summaryEdit,
                              research_state: {
                                ...summaryEdit.research_state,
                                [key]: items.filter((_, i) => i !== index),
                              },
                            })
                          }
                        >
                          移除此项
                        </button>
                      </div>
                    ))}
                  </div>
                ))}
                <Button type="submit" disabled={change.isPending}>
                  保存会话状态
                </Button>
              </form>
            )}
          </section>
        )}

        <section className="memory-section-card">
          <h3>保存研究记忆</h3>
          <form
            className="memory-form"
            onSubmit={(e) => {
              e.preventDefault()
              save.mutate()
            }}
          >
            <div className="memory-form-row">
              <label className="field">
                归属
                <select
                  aria-label="记忆归属"
                  value={level}
                  onChange={(e) => setLevel(e.target.value as MemoryEntry['level'])}
                >
                  {Object.entries(levels).map(([key, label]) => (
                    <option
                      key={key}
                      value={key}
                      disabled={
                        (key === 'project' && !projectId) ||
                        (key === 'conversation' && !conversationId)
                      }
                    >
                      {label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                类型
                <select
                  aria-label="记忆类型"
                  value={kind}
                  onChange={(e) => setKind(e.target.value)}
                >
                  {Object.entries(kinds).map(([key, label]) => (
                    <option key={key} value={key}>
                      {label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                判断性质
                <select
                  aria-label="判断性质"
                  value={epistemic}
                  onChange={(e) =>
                    setEpistemic(e.target.value as MemoryEntry['epistemic'])
                  }
                >
                  <option value="user_statement">我的记录或判断</option>
                  <option value="hypothesis">待验证假设</option>
                  <option
                    value="evidence_linked"
                    disabled={!sourceLinked || !citationIds.length}
                  >
                    关联论文证据
                  </option>
                </select>
              </label>
            </div>
            {level === 'paper' && (
              <label className="field">
                论文
                <select
                  aria-label="记忆论文"
                  value={paperId}
                  required
                  onChange={(e) => setPaperId(e.target.value)}
                >
                  <option value="">请选择论文</option>
                  {papers.data?.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.title}
                    </option>
                  ))}
                </select>
              </label>
            )}
            {sourceMessage && (
              <div>
                <label>
                  <input
                    type="checkbox"
                    checked={sourceLinked}
                    onChange={(e) => {
                      setSourceLinked(e.target.checked)
                      if (!e.target.checked) setEpistemic('user_statement')
                    }}
                  />
                  关联原回答及引用
                </label>
                {sourceLinked &&
                  sourceMessage.citations
                    .filter((c) => !c.deleted)
                    .map((c) => (
                      <label key={c.id} className="memory-citation-choice">
                        <input
                          type="checkbox"
                          checked={citationIds.includes(c.id)}
                          onChange={(e) =>
                            setCitationIds((ids) =>
                              e.target.checked
                                ? [...ids, c.id]
                                : ids.filter((id) => id !== c.id),
                            )
                          }
                        />
                        [{c.number}] {c.title}
                      </label>
                    ))}
                <p className="muted">
                  已自动去除旧回答中的 [n] 编号；请核对改写后的独立判断文字。
                </p>
              </div>
            )}
            <label className="field">
              内容
              <textarea
                aria-label="记忆内容"
                value={content}
                required
                maxLength={2000}
                rows={4}
                onChange={(e) => setContent(e.target.value)}
              />
            </label>
            <Button
              type="submit"
              disabled={save.isPending || (level === 'project' && !projectId)}
            >
              保存记忆
            </Button>
          </form>
        </section>

        <section className="memory-section-card">
          <div className="memory-section-header">
            <h3>已保存与待确认记忆（{filteredMemories.length}）</h3>
            {discussedCandidatesCount > 0 && (
              <Button
                variant="outline"
                size="sm"
                type="button"
                disabled={cleanupDiscussed.isPending}
                onClick={() => cleanupDiscussed.mutate()}
              >
                清理 {discussedCandidatesCount} 条未确认提问记录
              </Button>
            )}
          </div>

          <div className="memory-filter-bar">
            <input
              type="search"
              aria-label="搜索记忆内容"
              placeholder="搜索记忆关键词…"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
            <select
              aria-label="按类型筛选"
              value={filterKind}
              onChange={(e) => setFilterKind(e.target.value)}
            >
              <option value="">全部类型</option>
              {Object.entries(kinds).map(([k, label]) => (
                <option key={k} value={k}>
                  {label}
                </option>
              ))}
            </select>
            <select
              aria-label="按状态筛选"
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value)}
            >
              <option value="">全部状态</option>
              {Object.entries(statuses).map(([k, label]) => (
                <option key={k} value={k}>
                  {label}
                </option>
              ))}
            </select>
          </div>

          {selectedIds.length > 0 && (
            <div className="memory-bulk-bar" data-testid="memory-bulk-bar">
              <span>已选 {selectedIds.length} 项</span>
              <Button
                size="sm"
                variant="outline"
                disabled={bulkAction.isPending}
                onClick={() => bulkAction.mutate('confirm')}
              >
                批量确认
              </Button>
              <Button
                size="sm"
                variant="outline"
                disabled={bulkAction.isPending}
                onClick={() => bulkAction.mutate('pin')}
              >
                批量固定
              </Button>
              <Button
                size="sm"
                variant="outline"
                disabled={bulkAction.isPending}
                onClick={() => bulkAction.mutate('delete')}
              >
                批量删除
              </Button>
              <button
                type="button"
                className="text-button"
                onClick={() => setSelectedIds([])}
              >
                取消选择
              </button>
            </div>
          )}

          {memories.isLoading && <p>正在加载…</p>}
          {filteredMemories.length === 0 && (
            <p className="muted">暂无匹配的记忆。</p>
          )}
          {filteredMemories.map((item) => (
            <article
              key={item.id}
              className={`memory-entry ${
                item.status === 'superseded' ? 'is-superseded' : ''
              }`}
            >
              <div className="memory-entry-top">
                <label className="memory-select-label">
                  <input
                    type="checkbox"
                    aria-label={`选择记忆 ${item.id}`}
                    checked={selectedIds.includes(item.id)}
                    onChange={(e) =>
                      setSelectedIds((prev) =>
                        e.target.checked
                          ? [...prev, item.id]
                          : prev.filter((id) => id !== item.id),
                      )
                    }
                  />
                  <span className="muted">
                    {levels[item.level]} · {kinds[item.kind] ?? item.kind} ·{' '}
                    {statuses[item.status]}
                    {item.pinned ? ' · 已固定' : ''}
                  </span>
                </label>
              </div>
              {editing?.id === item.id ? (
                <form
                  onSubmit={(e) => {
                    e.preventDefault()
                    change.mutate({
                      path: `/memories/${item.id}`,
                      body: { content: editText },
                    })
                  }}
                >
                  <textarea
                    aria-label="编辑记忆内容"
                    maxLength={2000}
                    required
                    value={editText}
                    onChange={(e) => setEditText(e.target.value)}
                  />
                  <Button type="submit">保存修改</Button>
                  <Button variant="outline" onClick={() => setEditing(null)}>
                    取消
                  </Button>
                </form>
              ) : (
                <p>{item.content}</p>
              )}
              <p className="muted">
                {item.epistemic === 'hypothesis'
                  ? '待验证假设'
                  : item.epistemic === 'evidence_linked'
                    ? '已关联证据（仍需核对）'
                    : '用户记录'}{' '}
                · {evidenceLabels[item.evidence_status]}
              </p>
              {item.source_message_id && (
                <button
                  type="button"
                  className="text-button"
                  onClick={() =>
                    setSourceId(sourceId === item.id ? null : item.id)
                  }
                >
                  查看来源消息
                </button>
              )}
              {sourceId === item.id && (
                <div className="memory-source-text">
                  {source.isLoading ? (
                    '正在加载来源…'
                  ) : source.error ? (
                    '来源消息不可用'
                  ) : (
                    source.data && (
                      <>
                        <small>
                          {source.data.role === 'user' ? '用户提问' : '助手回答'}{' '}
                          · 消息 {source.data.seq}
                        </small>
                        <p>{source.data.content}</p>
                      </>
                    )
                  )}
                </div>
              )}
              {item.evidence.map((e) => {
                const paper = papers.data?.find((p) => p.id === e.paper_id)
                return (
                  <p key={e.id} className="muted">
                    证据：
                    {paper?.collection_id && e.status === 'current' ? (
                      <a
                        href={`#/library/${paper.collection_id}/paper/${paper.id}`}
                        onClick={onClose}
                      >
                        {e.title}
                      </a>
                    ) : (
                      e.title ?? '来源不可用'
                    )}{' '}
                    ·{' '}
                    {evidenceLabels[
                      e.status as keyof typeof evidenceLabels
                    ] ?? '来源不可用'}
                  </p>
                )
              })}
              <div className="memory-actions">
                {item.status === 'candidate' && (
                  <Button
                    disabled={change.isPending}
                    onClick={() =>
                      change.mutate({
                        path: `/memories/${item.id}`,
                        body: { status: 'active' },
                      })
                    }
                  >
                    确认记忆
                  </Button>
                )}
                <Button
                  variant="outline"
                  disabled={change.isPending}
                  onClick={() =>
                    change.mutate({
                      path: `/memories/${item.id}`,
                      body: { pinned: !item.pinned },
                    })
                  }
                >
                  {item.pinned ? '取消固定' : '固定'}
                </Button>
                <Button
                  variant="outline"
                  onClick={() => {
                    setEditing(item)
                    setEditText(item.content)
                  }}
                >
                  编辑
                </Button>
                <Button
                  variant="outline"
                  disabled={change.isPending}
                  onClick={() =>
                    change.mutate({
                      path: `/memories/${item.id}`,
                      body: { epistemic: 'hypothesis' },
                    })
                  }
                >
                  标为假设
                </Button>
                <Button
                  variant="outline"
                  disabled={change.isPending || item.status === 'superseded'}
                  onClick={() =>
                    change.mutate({
                      path: `/memories/${item.id}`,
                      body: { status: 'superseded' },
                    })
                  }
                >
                  标为已修正
                </Button>
                <Button
                  variant="outline"
                  disabled={change.isPending}
                  onClick={() =>
                    change.mutate({
                      path: `/memories/${item.id}`,
                      method: 'DELETE',
                    })
                  }
                >
                  删除记忆
                </Button>
              </div>
            </article>
          ))}
        </section>
      </div>
    </Modal>
  )
}

type CenterScope = 'core' | 'entries' | 'candidates' | 'sessions' | 'events'

export function MemoryCenter({ user }: { user: User }) {
  const client = useQueryClient()
  const [activeScope, setActiveScope] = useState<CenterScope>('core')
  const [selectedProjectId, setSelectedProjectId] = useState<string>('')
  const [searchQuery, setSearchQuery] = useState('')
  const [filterKind, setFilterKind] = useState('')
  const [filterStatus, setFilterStatus] = useState('')
  const [filterEvidence, setFilterEvidence] = useState('')
  const [selectedIds, setSelectedIds] = useState<string[]>([])
  const [profileText, setProfileText] = useState('')
  const [projectText, setProjectText] = useState('')
  const [sessionQuery, setSessionQuery] = useState('')
  const [submittedSessionQuery, setSubmittedSessionQuery] = useState('')
  const [notice, setNotice] = useState('')
  const [error, setError] = useState('')

  const projects = useQuery({
    queryKey: ['projects', user.id],
    queryFn: () => api<ResearchProject[]>('/projects'),
  })

  useEffect(() => {
    if (!selectedProjectId && projects.data && projects.data.length > 0) {
      setSelectedProjectId(projects.data[0].id)
    }
  }, [projects.data, selectedProjectId])

  const memorySettings = useQuery({
    queryKey: ['memory-settings', user.id],
    queryFn: () => api<MemorySettings>('/memory/settings'),
  })

  const coreMemory = useQuery({
    queryKey: ['memory-core', user.id, selectedProjectId],
    queryFn: () =>
      api<CoreMemoryResponse>(
        `/memory/core${selectedProjectId ? `?project_id=${selectedProjectId}` : ''}`,
      ),
  })

  useEffect(() => {
    if (coreMemory.data) {
      setProfileText(coreMemory.data.profile.content)
      setProjectText(coreMemory.data.project?.content ?? '')
    }
  }, [coreMemory.data])

  const memories = useQuery({
    queryKey: [
      'memories-center',
      user.id,
      selectedProjectId,
      filterKind,
      filterStatus,
      filterEvidence,
      searchQuery,
      activeScope,
    ],
    queryFn: () => {
      const params = new URLSearchParams()
      if (selectedProjectId) params.set('project_id', selectedProjectId)
      if (filterKind) params.set('kind', filterKind)
      if (activeScope === 'candidates') params.set('status', 'candidate')
      else if (filterStatus) params.set('status', filterStatus)
      if (filterEvidence) params.set('evidence_status', filterEvidence)
      if (searchQuery.trim()) params.set('q', searchQuery.trim())
      const qs = params.toString()
      return api<MemoryEntry[]>(`/memories${qs ? `?${qs}` : ''}`)
    },
  })

  const events = useQuery({
    queryKey: ['memory-events', user.id, selectedProjectId],
    queryFn: () =>
      api<MemoryEventItem[]>(
        `/memory/events${selectedProjectId ? `?project_id=${selectedProjectId}` : ''}`,
      ),
  })

  const sessionResults = useQuery({
    queryKey: ['session-search', user.id, selectedProjectId, submittedSessionQuery],
    queryFn: () =>
      api<SessionSearchResult[]>(
        `/memory/sessions/search?q=${encodeURIComponent(submittedSessionQuery)}${
          selectedProjectId ? `&project_id=${selectedProjectId}` : ''
        }`,
      ),
    enabled: submittedSessionQuery.trim().length > 0,
  })

  function invalidateAll() {
    void client.invalidateQueries({ queryKey: ['memories-center', user.id] })
    void client.invalidateQueries({ queryKey: ['memories', user.id] })
    void client.invalidateQueries({ queryKey: ['memory-core', user.id] })
    void client.invalidateQueries({ queryKey: ['memory-events', user.id] })
    void client.invalidateQueries({ queryKey: ['memory-settings', user.id] })
  }

  const updateWriteMode = useMutation({
    mutationFn: (mode: 'auto' | 'review') =>
      api<MemorySettings>('/memory/settings', 'PUT', { memory_write_mode: mode }),
    onSuccess: (res) => {
      setNotice(
        res.memory_write_mode === 'auto'
          ? '已开启自动记忆模式（回答下方可一键撤销）'
          : '已切换为待确认审核模式',
      )
      invalidateAll()
    },
    onError: (e) => setError(e.message),
  })

  const saveCoreBlock = useMutation({
    mutationFn: ({ level, text }: { level: 'user' | 'project'; text: string }) =>
      api(
        `/memory/core?level=${level}${
          level === 'project' && selectedProjectId ? `&project_id=${selectedProjectId}` : ''
        }`,
        'PUT',
        { content: text },
      ),
    onSuccess: () => {
      setNotice('核心记忆已保存，新问答将加载最新冻结快照')
      setError('')
      invalidateAll()
    },
    onError: (e) => setError(e.message),
  })

  const bulkMutate = useMutation({
    mutationFn: (action: 'confirm' | 'supersede' | 'delete' | 'pin' | 'unpin') =>
      api('/memories/bulk', 'POST', { ids: selectedIds, action }),
    onSuccess: () => {
      setSelectedIds([])
      setNotice('批量操作已完成')
      invalidateAll()
    },
    onError: (e) => setError(e.message),
  })

  const cleanupMutate = useMutation({
    mutationFn: () =>
      api<{ cleaned: number }>(
        `/memories/cleanup-discussed${
          selectedProjectId ? `?project_id=${selectedProjectId}` : ''
        }`,
        'POST',
      ),
    onSuccess: (res) => {
      setNotice(`已清理 ${res.cleaned} 条未确认的提问记录`)
      invalidateAll()
    },
    onError: (e) => setError(e.message),
  })

  const undoMutate = useMutation({
    mutationFn: (eid: string) => api(`/memory/events/${eid}/undo`, 'POST'),
    onSuccess: () => {
      setNotice('已撤销该项记忆变更')
      invalidateAll()
    },
    onError: (e) => setError(e.message),
  })

  const writeMode = memorySettings.data?.memory_write_mode ?? 'auto'

  return (
    <section className="page memory-center-page" data-testid="memory-center-page">
      <PageHeader
        eyebrow="HERMES-STYLE MEMORY HARNESS"
        title="研究记忆中心"
        description="三层记忆架构：常驻核心记忆（USER.md / MEMORY.md 冻结快照）、带证据校验的档案记忆、以及跨会话原文检索（Session Search）。"
        actions={
          <div className="memory-header-controls">
            <label className="memory-project-filter">
              <span>课题范围：</span>
              <select
                aria-label="记忆中心课题选择"
                value={selectedProjectId}
                onChange={(e) => {
                  setSelectedProjectId(e.target.value)
                  setSelectedIds([])
                }}
              >
                <option value="">全部项目与个人</option>
                {projects.data?.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name}
                  </option>
                ))}
              </select>
            </label>
            <div className="mode-toggle" role="group" aria-label="记忆写入模式">
              <button
                type="button"
                className={`mode-btn ${writeMode === 'auto' ? 'active' : ''}`}
                onClick={() => updateWriteMode.mutate('auto')}
              >
                <Sparkles size={14} /> 自动记住（可撤销）
              </button>
              <button
                type="button"
                className={`mode-btn ${writeMode === 'review' ? 'active' : ''}`}
                onClick={() => updateWriteMode.mutate('review')}
              >
                <Check size={14} /> 待确认审核
              </button>
            </div>
          </div>
        }
      />

      {notice && <Notice tone="success">{notice}</Notice>}
      {error && <Notice tone="error">{error}</Notice>}

      <div className="memory-center-layout">
        <aside className="panel memory-center-nav" aria-label="记忆分层导航">
          <button
            type="button"
            className={`memory-nav-btn ${activeScope === 'core' ? 'active' : ''}`}
            onClick={() => setActiveScope('core')}
          >
            <Brain size={16} />
            <span>核心记忆（USER / MEMORY.md）</span>
          </button>
          <button
            type="button"
            className={`memory-nav-btn ${activeScope === 'entries' ? 'active' : ''}`}
            onClick={() => setActiveScope('entries')}
          >
            <Pin size={16} />
            <span>档案记忆与研究判断</span>
          </button>
          <button
            type="button"
            className={`memory-nav-btn ${activeScope === 'candidates' ? 'active' : ''}`}
            onClick={() => setActiveScope('candidates')}
          >
            <Check size={16} />
            <span>待确认队列</span>
          </button>
          <button
            type="button"
            className={`memory-nav-btn ${activeScope === 'sessions' ? 'active' : ''}`}
            onClick={() => setActiveScope('sessions')}
          >
            <Search size={16} />
            <span>跨会话原文检索</span>
          </button>
          <button
            type="button"
            className={`memory-nav-btn ${activeScope === 'events' ? 'active' : ''}`}
            onClick={() => setActiveScope('events')}
          >
            <History size={16} />
            <span>记忆活动日志与撤销</span>
          </button>
        </aside>

        <div className="memory-center-content">
          {activeScope === 'core' && coreMemory.data && (
            <div className="memory-core-grid">
              <section className="panel memory-core-panel">
                <div className="panel-heading">
                  <Brain size={18} />
                  <div>
                    <h2>研究者画像（USER.md）</h2>
                    <p className="muted">
                      跨课题常驻上下文，记录你的研究领域、学术背景、术语习惯与回答偏好。
                    </p>
                  </div>
                </div>
                <Meter
                  label="画像容量占用（上限 1500 字符）"
                  used={profileText.length}
                  limit={coreMemory.data.profile.char_limit}
                  display={`${profileText.length} / ${coreMemory.data.profile.char_limit} 字符`}
                />
                <textarea
                  aria-label="记忆中心研究者画像"
                  className="core-markdown-textarea"
                  rows={8}
                  maxLength={coreMemory.data.profile.char_limit}
                  placeholder="- 研究方向：大模型高效微调与推理优化&#10;- 关注指标：显存占用、吞吐量与下游任务消融&#10;- 回答风格：优先给出公式推导与实验表格对比"
                  value={profileText}
                  onChange={(e) => setProfileText(e.target.value)}
                />
                <div className="memory-actions">
                  <Button
                    disabled={saveCoreBlock.isPending}
                    onClick={() =>
                      saveCoreBlock.mutate({ level: 'user', text: profileText })
                    }
                  >
                    保存研究者画像
                  </Button>
                </div>
              </section>

              <section className="panel memory-core-panel">
                <div className="panel-heading">
                  <Sparkles size={18} />
                  <div>
                    <h2>
                      项目核心备忘（MEMORY.md
                      {coreMemory.data.project
                        ? ` · ${coreMemory.data.project.project_name}`
                        : ''}
                      ）
                    </h2>
                    <p className="muted">
                      记录当前课题的核心目标、实验约束、已排除假设与阶段结论，每次会话启动自动冻结注入。
                    </p>
                  </div>
                </div>
                {coreMemory.data.project ? (
                  <>
                    <Meter
                      label="项目备忘容量占用（上限 3000 字符）"
                      used={projectText.length}
                      limit={coreMemory.data.project.char_limit}
                      display={`${projectText.length} / ${coreMemory.data.project.char_limit} 字符`}
                    />
                    <textarea
                      aria-label="记忆中心项目备忘"
                      className="core-markdown-textarea"
                      rows={8}
                      maxLength={coreMemory.data.project.char_limit}
                      placeholder="- 核心目标：对比 LoRA 与 DoRA 在低比特量化下的稳定性&#10;- 约束条件：仅比较相同 rank 与相同微调步数的实验&#10;- 已排除假设：单纯增加 rank 并不能解决长上下文退化"
                      value={projectText}
                      onChange={(e) => setProjectText(e.target.value)}
                    />
                    <div className="memory-actions">
                      <Button
                        disabled={saveCoreBlock.isPending}
                        onClick={() =>
                          saveCoreBlock.mutate({ level: 'project', text: projectText })
                        }
                      >
                        保存项目核心备忘
                      </Button>
                    </div>
                  </>
                ) : (
                  <p className="muted">请在右上角选择一个具体研究项目以编辑其核心备忘。</p>
                )}
              </section>
            </div>
          )}

          {(activeScope === 'entries' || activeScope === 'candidates') && (
            <section className="panel">
              <div className="memory-section-header">
                <h2>
                  {activeScope === 'candidates' ? '待确认记忆队列' : '档案记忆列表'}（
                  {memories.data?.length ?? 0}）
                </h2>
                <Button
                  variant="outline"
                  size="sm"
                  disabled={cleanupMutate.isPending}
                  onClick={() => cleanupMutate.mutate()}
                >
                  <Trash2 size={14} /> 清理未确认的提问记录
                </Button>
              </div>

              <div className="memory-filter-bar">
                <input
                  type="search"
                  aria-label="检索档案记忆"
                  placeholder="输入关键词检索记忆内容…"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
                <select
                  aria-label="档案记忆类型筛选"
                  value={filterKind}
                  onChange={(e) => setFilterKind(e.target.value)}
                >
                  <option value="">全部类型</option>
                  {Object.entries(kinds).map(([k, label]) => (
                    <option key={k} value={k}>
                      {label}
                    </option>
                  ))}
                </select>
                {activeScope !== 'candidates' && (
                  <select
                    aria-label="档案记忆状态筛选"
                    value={filterStatus}
                    onChange={(e) => setFilterStatus(e.target.value)}
                  >
                    <option value="">全部状态</option>
                    {Object.entries(statuses).map(([k, label]) => (
                      <option key={k} value={k}>
                        {label}
                      </option>
                    ))}
                  </select>
                )}
                <select
                  aria-label="档案记忆证据筛选"
                  value={filterEvidence}
                  onChange={(e) => setFilterEvidence(e.target.value)}
                >
                  <option value="">全部证据状态</option>
                  {Object.entries(evidenceLabels).map(([k, label]) => (
                    <option key={k} value={k}>
                      {label}
                    </option>
                  ))}
                </select>
              </div>

              {selectedIds.length > 0 && (
                <div className="memory-bulk-bar">
                  <span>已选中 {selectedIds.length} 条记忆</span>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => bulkMutate.mutate('confirm')}
                  >
                    批量确认
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => bulkMutate.mutate('pin')}
                  >
                    批量固定
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => bulkMutate.mutate('supersede')}
                  >
                    批量标为已修正
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => bulkMutate.mutate('delete')}
                  >
                    批量删除
                  </Button>
                </div>
              )}

              <div className="memory-entries-list">
                {memories.data?.map((item) => (
                  <article
                    key={item.id}
                    className={`memory-entry ${
                      item.status === 'superseded' ? 'is-superseded' : ''
                    }`}
                  >
                    <label className="memory-select-label">
                      <input
                        type="checkbox"
                        checked={selectedIds.includes(item.id)}
                        onChange={(e) =>
                          setSelectedIds((prev) =>
                            e.target.checked
                              ? [...prev, item.id]
                              : prev.filter((id) => id !== item.id),
                          )
                        }
                      />
                      <span className="muted">
                        {levels[item.level]} · {kinds[item.kind] ?? item.kind} ·{' '}
                        {statuses[item.status]}
                        {item.pinned ? ' · 已固定' : ''} ·{' '}
                        {evidenceLabels[item.evidence_status]}
                      </span>
                    </label>
                    <p>{item.content}</p>
                  </article>
                ))}
                {memories.data?.length === 0 && (
                  <p className="muted">当前筛选条件下暂无记忆条目。</p>
                )}
              </div>
            </section>
          )}

          {activeScope === 'sessions' && (
            <section className="panel">
              <h2>跨会话原文检索（Session Search）</h2>
              <p className="muted">
                无需把每次提问都存为候选记忆；直接检索历史会话的原始问答片段。
              </p>
              <form
                className="memory-filter-bar"
                onSubmit={(e) => {
                  e.preventDefault()
                  setSubmittedSessionQuery(sessionQuery.trim())
                }}
              >
                <input
                  type="search"
                  aria-label="跨会话检索关键词"
                  placeholder="输入曾讨论过的概念、实验指标或方法名…"
                  value={sessionQuery}
                  onChange={(e) => setSessionQuery(e.target.value)}
                />
                <Button type="submit" disabled={!sessionQuery.trim()}>
                  <Search size={14} /> 检索历史会话
                </Button>
              </form>
              <div className="memory-entries-list">
                {sessionResults.data?.map((hit) => (
                  <article key={hit.message_id} className="memory-entry">
                    <p className="muted">
                      会话：{hit.conversation_title} ·{' '}
                      {hit.role === 'user' ? '用户提问' : '助手回答'}（消息 #{hit.seq}）
                    </p>
                    <p>{hit.snippet}</p>
                  </article>
                ))}
                {submittedSessionQuery && sessionResults.data?.length === 0 && (
                  <p className="muted">未找到包含该关键词的历史会话片段。</p>
                )}
              </div>
            </section>
          )}

          {activeScope === 'events' && (
            <section className="panel">
              <h2>记忆活动日志与一键撤销</h2>
              <p className="muted">
                记录自动提取与手动修改的每一次记忆变更，可随时回滚。
              </p>
              <div className="memory-entries-list">
                {events.data?.map((ev) => (
                  <article key={ev.id} className="memory-entry memory-event-row">
                    <div>
                      <p className="muted">
                        {ev.actor === 'agent' ? '自动记住' : '手动操作'} · {ev.action} ·{' '}
                        {ev.undone_at ? '已撤销' : '生效中'}
                      </p>
                      <p>{ev.summary}</p>
                    </div>
                    {!ev.undone_at && (
                      <Button
                        variant="outline"
                        size="sm"
                        disabled={undoMutate.isPending}
                        onClick={() => undoMutate.mutate(ev.id)}
                      >
                        <RotateCcw size={13} /> 撤销
                      </Button>
                    )}
                  </article>
                ))}
                {events.data?.length === 0 && (
                  <p className="muted">暂无记忆变更记录。</p>
                )}
              </div>
            </section>
          )}
        </div>
      </div>
    </section>
  )
}

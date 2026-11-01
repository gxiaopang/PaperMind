import { useEffect, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  ArrowLeft,
  Columns2,
  ExternalLink,
  FileText,
  FileUp,
  MessageSquare,
  Pencil,
  RefreshCw,
  ScanText,
  Search,
  Trash2,
  Upload,
  X,
} from 'lucide-react'
import { api, type Collection, type Paper, type UsageData, type User } from '../lib/api'
import { formatBytes, formatNumber, paperStatusLabels, paperStatusProgress } from '../lib/format'
import { PaperDetail } from './PaperDetail'
import { Button } from './ui/button'
import { Modal } from './ui/modal'
import { EmptyState, Meter, Notice, Skeleton, Spinner } from './ui/primitives'

function statusTone(paper: Paper): string {
  if (paper.error || paper.status === 'failed') return 'danger'
  if (paper.status === 'ready' && !paper.pending_version_id) return 'success'
  return 'progress'
}

export function Papers({
  collection,
  user,
  paperId,
  onBack,
  onOpenPaper,
  onAskPaper,
}: {
  collection: Collection
  user: User
  paperId?: string
  onBack: () => void
  onOpenPaper: (paperId: string | null) => void
  onAskPaper?: (paper: Paper) => void
}) {
  const client = useQueryClient()
  const fileInput = useRef<HTMLInputElement>(null)
  const [consent, setConsent] = useState(false)
  const [file, setFile] = useState<File | null>(null)
  const [editing, setEditing] = useState<Paper | null>(null)
  const [title, setTitle] = useState('')
  const [authors, setAuthors] = useState('')
  const [abstract, setAbstract] = useState('')
  const [venue, setVenue] = useState('')
  const [year, setYear] = useState('')
  const [doi, setDoi] = useState('')
  const [arxivId, setArxivId] = useState('')
  const [dragging, setDragging] = useState(false)
  const [fileError, setFileError] = useState('')
  const [search, setSearch] = useState('')

  const papers = useQuery({
    queryKey: ['papers', collection.id],
    queryFn: () => api<Paper[]>(`/collections/${collection.id}/papers`),
    refetchInterval: (query) => (query.state.data?.some((p) => p.pending_version_id) ? 2000 : false),
  })
  const usage = useQuery({
    queryKey: ['usage', user.id],
    queryFn: () => api<UsageData>('/usage'),
    refetchInterval: papers.data?.some((p) => p.pending_version_id) ? 3000 : false,
  })

  function refresh() {
    void client.invalidateQueries({ queryKey: ['papers', collection.id] })
    void client.invalidateQueries({ queryKey: ['all-papers', user.id] })
    void client.invalidateQueries({ queryKey: ['usage', user.id] })
  }

  const upload = useMutation({
    mutationFn: () => api<Paper>(`/collections/${collection.id}/papers?cloud_consent=true`, 'POST', file),
    onSuccess: () => {
      setFile(null)
      if (fileInput.current) fileInput.current.value = ''
      refresh()
    },
  })

  const action = useMutation({
    mutationFn: ({ id, operation }: { id: string; operation: string }) =>
      api(`/papers/${id}${operation === 'delete' ? '' : '/' + operation}`, operation === 'delete' ? 'DELETE' : 'POST'),
    onSuccess: refresh,
  })

  const edit = useMutation({
    mutationFn: () =>
      api(`/papers/${editing?.id}`, 'PATCH', {
        title,
        authors: authors
          .split('；')
          .map((a) => a.trim())
          .filter(Boolean),
        abstract: abstract || null,
        year: year.trim() ? Number.parseInt(year.trim(), 10) : null,
        venue: venue.trim() || null,
        doi: doi.trim() || null,
        arxiv_id: arxivId.trim() || null,
      }),
    onSuccess: () => {
      setEditing(null)
      refresh()
    },
  })

  const missingPaper = !!paperId && !!papers.data && !papers.data.some((p) => p.id === paperId)
  useEffect(() => {
    if (missingPaper) onOpenPaper(null)
  }, [missingPaper, onOpenPaper])

  function choose(selectedFile?: File) {
    setFileError('')
    if (!selectedFile) return
    if (!selectedFile.name.toLowerCase().endsWith('.pdf')) {
      setFileError('请选择 PDF 文件')
      return
    }
    setFile(selectedFile)
  }

  function openEdit(paper: Paper) {
    setEditing(paper)
    setTitle(paper.title)
    setAuthors(paper.authors.join('；'))
    setAbstract(paper.abstract || '')
    setVenue(paper.venue || '')
    setYear(paper.year ? String(paper.year) : '')
    setDoi(paper.doi || '')
    setArxivId(paper.arxiv_id || '')
  }

  if (paperId) {
    const livePaper = papers.data?.find((p) => p.id === paperId)
    if (!livePaper) return <div className="page-loading"><Spinner label="正在打开论文…" /></div>
    return <PaperDetail paper={livePaper} onBack={() => onOpenPaper(null)} onAskPaper={onAskPaper} />
  }

  const verified = !!user.email_verified_at
  const q = search.trim().toLowerCase()
  const visible = (papers.data || []).filter(
    (p) => !q || p.title.toLowerCase().includes(q) || p.authors.some((a) => a.toLowerCase().includes(q)),
  )
  const processing = papers.data?.filter((p) => p.pending_version_id).length ?? 0

  return (
    <section className="page papers-page">
      <button className="text-button back-link" onClick={onBack}>
        <ArrowLeft size={15} />
        返回知识库
      </button>
      <div className="page-header">
        <div className="page-header-text">
          <span className="eyebrow">RESEARCH LIBRARY</span>
          <h1>{collection.name}</h1>
          <p className="muted">{collection.description || '把论文积累成可追溯的研究资料。'}</p>
        </div>
      </div>

      <div className="papers-layout">
        <div className="papers-main">
          <div className="toolbar">
            <div className="search-box">
              <Search size={16} />
              <input
                type="search"
                aria-label="搜索论文"
                placeholder="按标题或作者搜索…"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
              />
            </div>
            <span className="toolbar-meta">
              共 {papers.data?.length ?? 0} 篇{processing > 0 ? ` · ${processing} 篇处理中` : ''}
            </span>
          </div>

          {(papers.error || action.error) && (
            <Notice tone="error">{papers.error?.message || action.error?.message}</Notice>
          )}

          <div className="paper-list" aria-live="polite">
            {papers.isPending && [0, 1].map((i) => <div key={i} className="paper-card"><Skeleton lines={3} /></div>)}
            {visible.map((paper) => {
              const tone = statusTone(paper)
              const progress = paperStatusProgress[paper.status] ?? 10
              return (
                <article className={`paper-card tone-${tone}`} key={paper.id}>
                  <div className="paper-card-head">
                    <span className="paper-icon"><FileText size={20} /></span>
                    <div className="paper-card-body">
                      <h3 className="paper-title">
                        {paper.active_version_id ? (
                          <button type="button" className="paper-title-link" onClick={() => onOpenPaper(paper.id)}>
                            {paper.title}
                          </button>
                        ) : (
                          paper.title
                        )}
                      </h3>
                      <p className="paper-meta">
                        {paper.authors.join(' · ') || '作者待补充'}
                        {paper.year ? ` · ${paper.year}` : ''}
                        {paper.venue ? ` · ${paper.venue}` : ''} · {paper.page_count} 页 · {formatBytes(paper.size_bytes)}
                      </p>
                    </div>
                    <span className={`paper-status status-${tone}`}>{paperStatusLabels[paper.status] || paper.status}</span>
                  </div>

                  {paper.pending_version_id && !paper.error && (
                    <div className="paper-progress">
                      <div className="meter-track small" aria-hidden="true">
                        <span className="meter-fill animated" style={{ width: `${progress}%` }} />
                      </div>
                      {paper.active_version_id && <span className="muted">正在构建新版本，当前版本仍可使用。</span>}
                    </div>
                  )}
                  {paper.error && (
                    <Notice tone="error">
                      {paper.error}
                      {paper.active_version_id ? '；当前版本仍可使用。' : ''}
                    </Notice>
                  )}

                  <div className="paper-actions">
                    <div className="paper-actions-primary">
                      {paper.active_version_id && (
                        <Button size="sm" aria-label="双栏阅读与原文预览" onClick={() => onOpenPaper(paper.id)}>
                          <Columns2 size={15} /> 双栏阅读
                        </Button>
                      )}
                      {paper.active_version_id && onAskPaper && (
                        <Button size="sm" variant="soft" onClick={() => onAskPaper(paper)}>
                          <MessageSquare size={15} /> 提问
                        </Button>
                      )}
                      <a className="button button-outline button-sm" href={`/api/v1/papers/${paper.id}/pdf`} target="_blank" rel="noreferrer">
                        <ExternalLink size={14} />
                        查看 PDF
                      </a>
                    </div>
                    <div className="paper-actions-secondary">
                      {paper.active_version_id && (
                        <a className="button button-ghost button-sm" href={`/api/v1/papers/${paper.id}/markdown`} target="_blank" rel="noreferrer">
                          <ScanText size={14} />
                          解析原文
                        </a>
                      )}
                      <Button size="sm" variant="ghost" onClick={() => openEdit(paper)}>
                        <Pencil size={14} />
                        编辑信息
                      </Button>
                      {paper.error && (
                        <Button
                          size="sm"
                          variant="ghost"
                          disabled={!!paper.pending_version_id || action.isPending}
                          onClick={() => action.mutate({ id: paper.id, operation: 'retry' })}
                        >
                          <RefreshCw size={14} />
                          重试
                        </Button>
                      )}
                      <Button
                        size="sm"
                        variant="ghost"
                        title="使用当前解析结果重新切分检索分块"
                        disabled={!paper.active_version_id || !!paper.pending_version_id || action.isPending}
                        onClick={() => action.mutate({ id: paper.id, operation: 'rechunk' })}
                      >
                        重新分块
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        title="重新发送 PDF 至 MinerU 解析（消耗解析额度）"
                        disabled={!!paper.pending_version_id || action.isPending}
                        onClick={() => {
                          if (confirm('重新解析会再次将 PDF 发送至 MinerU，并消耗解析额度。是否继续？')) {
                            action.mutate({ id: paper.id, operation: 'reparse' })
                          }
                        }}
                      >
                        重新解析
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        className="danger-text"
                        disabled={action.isPending}
                        onClick={() => {
                          if (confirm('确定删除论文及其解析文件？历史引用将标记为来源已删除。')) {
                            action.mutate({ id: paper.id, operation: 'delete' })
                          }
                        }}
                      >
                        <Trash2 size={14} />
                        删除论文
                      </Button>
                    </div>
                  </div>
                </article>
              )
            })}
          </div>
          {papers.data?.length === 0 && (
            <EmptyState compact icon={<FileUp size={32} strokeWidth={1.5} />} title="知识库还是空的">
              还没有论文，上传第一篇 PDF 开始积累。
            </EmptyState>
          )}
          {papers.data && papers.data.length > 0 && visible.length === 0 && (
            <EmptyState compact icon={<Search size={28} />} title="没有匹配的论文">换个关键词试试。</EmptyState>
          )}
        </div>

        <aside className="papers-side">
          <form
            className={`panel upload-panel ${dragging ? 'dragging' : ''}`}
            onDragOver={(e) => {
              e.preventDefault()
              setDragging(true)
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault()
              setDragging(false)
              if (verified) choose(e.dataTransfer.files[0])
            }}
            onSubmit={(e) => {
              e.preventDefault()
              upload.mutate()
            }}
          >
            <div className="panel-heading">
              <Upload size={18} />
              <h2>上传论文</h2>
            </div>
            <input
              ref={fileInput}
              id="paper-file"
              className="sr-only"
              type="file"
              aria-label="PDF 文件"
              accept=".pdf,application/pdf"
              disabled={!verified || upload.isPending}
              onChange={(e) => choose(e.target.files?.[0])}
            />
            {file ? (
              <div className="file-chip">
                <FileText size={18} />
                <div>
                  <strong title={file.name}>{file.name}</strong>
                  <span>{formatBytes(file.size)}</span>
                </div>
                <button
                  type="button"
                  className="icon-button"
                  aria-label="移除文件"
                  onClick={() => {
                    setFile(null)
                    if (fileInput.current) fileInput.current.value = ''
                  }}
                >
                  <X size={16} />
                </button>
              </div>
            ) : (
              <label htmlFor="paper-file" className={`dropzone ${verified ? '' : 'disabled'}`}>
                <FileUp size={26} strokeWidth={1.6} />
                <strong>拖入 PDF，或点击选择文件</strong>
                <span>解析后自动分块入库，同一知识库内的重复文件会复用已有论文。</span>
              </label>
            )}
            <label className="consent">
              <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} />
              <span>我同意将此 PDF 发送至 MinerU 云服务进行解析。</span>
            </label>
            {(upload.error || fileError) && <Notice tone="error">{fileError || upload.error?.message}</Notice>}
            <Button className="full" disabled={!verified || !file || !consent || upload.isPending}>
              {upload.isPending ? '上传中…' : '上传并入库'}
            </Button>
            {!verified && <p className="muted small">请先完成邮箱验证。</p>}
          </form>

          {usage.data && (
            <div className="panel quota-panel">
              <div className="panel-heading">
                <h2>配额概览</h2>
              </div>
              <Meter
                label="论文数量"
                used={usage.data.papers}
                limit={usage.data.limits.papers}
                display={`${usage.data.papers} / ${usage.data.limits.papers}`}
              />
              <Meter
                label="今日解析页数"
                used={usage.data.parse_pages}
                reserved={usage.data.parse_pages_reserved}
                limit={usage.data.limits.daily_parse_pages}
                display={`${usage.data.parse_pages} / ${usage.data.limits.daily_parse_pages}`}
                hint={usage.data.parse_pages_reserved ? `另有 ${usage.data.parse_pages_reserved} 页预占` : undefined}
              />
              <Meter
                label={`模型 Token${usage.data.model_tokens_estimated ? '（含估算）' : ''}`}
                used={usage.data.model_tokens}
                reserved={usage.data.model_tokens_reserved}
                limit={usage.data.limits.daily_model_tokens}
                display={`${formatNumber(usage.data.model_tokens)} / ${formatNumber(usage.data.limits.daily_model_tokens)}`}
              />
            </div>
          )}
        </aside>
      </div>

      {editing && (
        <Modal labelId="edit-paper-title" title="编辑论文信息" size="lg" onClose={() => setEditing(null)}>
          <form
            onSubmit={(e) => {
              e.preventDefault()
              edit.mutate()
            }}
          >
            <div className="field">
              <label htmlFor="paper-title">论文标题</label>
              <input
                id="paper-title"
                required
                maxLength={2000}
                value={title}
                onChange={(e) => setTitle(e.target.value)}
              />
            </div>
            <div className="field">
              <label htmlFor="paper-authors">作者（用中文分号分隔）</label>
              <input id="paper-authors" value={authors} onChange={(e) => setAuthors(e.target.value)} />
            </div>
            <div className="form-grid-2">
              <div className="field">
                <label htmlFor="paper-year">年份</label>
                <input
                  id="paper-year"
                  type="number"
                  min={1900}
                  max={2100}
                  value={year}
                  onChange={(e) => setYear(e.target.value)}
                />
              </div>
              <div className="field">
                <label htmlFor="paper-venue">期刊 / 会议</label>
                <input id="paper-venue" value={venue} onChange={(e) => setVenue(e.target.value)} />
              </div>
              <div className="field">
                <label htmlFor="paper-doi">DOI</label>
                <input id="paper-doi" value={doi} onChange={(e) => setDoi(e.target.value)} />
              </div>
              <div className="field">
                <label htmlFor="paper-arxiv">arXiv ID</label>
                <input id="paper-arxiv" value={arxivId} onChange={(e) => setArxivId(e.target.value)} />
              </div>
            </div>
            <div className="field">
              <label htmlFor="paper-abstract">摘要</label>
              <textarea id="paper-abstract" rows={5} value={abstract} onChange={(e) => setAbstract(e.target.value)} />
            </div>
            {edit.error && <Notice tone="error">{edit.error.message}</Notice>}
            <div className="modal-actions">
              <Button variant="ghost" type="button" onClick={() => setEditing(null)}>
                取消
              </Button>
              <Button disabled={edit.isPending}>保存</Button>
            </div>
          </form>
        </Modal>
      )}
    </section>
  )
}

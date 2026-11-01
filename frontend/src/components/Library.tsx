import { useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { FileText, FolderOpen, Gauge, Layers, Plus, Search, Trash2 } from 'lucide-react'
import { api, type Collection, type Paper, type UsageData, type User } from '../lib/api'
import { formatCompact, formatDate, hashIndex } from '../lib/format'
import { Button } from './ui/button'
import { Modal } from './ui/modal'
import { EmptyState, Notice, PageHeader, Skeleton } from './ui/primitives'

export function Library({ user, onOpen }: { user: User; onOpen: (collection: Collection) => void }) {
  const client = useQueryClient()
  const [creating, setCreating] = useState(false)
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [search, setSearch] = useState('')

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

  const create = useMutation({
    mutationFn: () => api<Collection>('/collections', 'POST', { name, description }),
    onSuccess: () => {
      setCreating(false)
      setName('')
      setDescription('')
      void client.invalidateQueries({ queryKey: ['collections', user.id] })
    },
  })

  const remove = useMutation({
    mutationFn: (id: string) => api(`/collections/${id}`, 'DELETE'),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ['collections', user.id] })
      void client.invalidateQueries({ queryKey: ['all-papers', user.id] })
    },
  })

  const counts = useMemo(() => {
    const map = new Map<string, { total: number; ready: number }>()
    for (const p of allPapers.data || []) {
      if (!p.collection_id) continue
      const entry = map.get(p.collection_id) || { total: 0, ready: 0 }
      entry.total += 1
      if (p.active_version_id) entry.ready += 1
      map.set(p.collection_id, entry)
    }
    return map
  }, [allPapers.data])

  const filtered = (collections.data || []).filter((c) => {
    const q = search.trim().toLowerCase()
    return !q || c.name.toLowerCase().includes(q) || c.description.toLowerCase().includes(q)
  })
  const totalPapers = allPapers.data?.length ?? 0
  const readyPapers = allPapers.data?.filter((p) => p.active_version_id).length ?? 0

  return (
    <section className="page library-page">
      <PageHeader
        eyebrow="YOUR RESEARCH, CONNECTED"
        title="我的知识库"
        description="按研究方向组织文献，每个知识库都是一处可检索、可追问的专属空间。"
        actions={
          <Button onClick={() => setCreating(true)}>
            <Plus size={17} />
            新建知识库
          </Button>
        }
      />

      <div className="stat-strip">
        <div className="stat-card">
          <span className="stat-icon tone-1"><FolderOpen size={18} /></span>
          <div>
            <strong>{collections.data?.length ?? '—'}</strong>
            <span>知识库</span>
          </div>
        </div>
        <div className="stat-card">
          <span className="stat-icon tone-2"><FileText size={18} /></span>
          <div>
            <strong>{allPapers.data ? totalPapers : '—'}</strong>
            <span>论文{allPapers.data ? ` · ${readyPapers} 篇可问答` : ''}</span>
          </div>
        </div>
        <div className="stat-card">
          <span className="stat-icon tone-3"><Layers size={18} /></span>
          <div>
            <strong>{usage.data ? `${usage.data.parse_pages}` : '—'}</strong>
            <span>今日解析页{usage.data ? ` / ${usage.data.limits.daily_parse_pages}` : ''}</span>
          </div>
        </div>
        <div className="stat-card">
          <span className="stat-icon tone-4"><Gauge size={18} /></span>
          <div>
            <strong>{usage.data ? formatCompact(usage.data.model_tokens) : '—'}</strong>
            <span>今日 Token{usage.data ? ` / ${formatCompact(usage.data.limits.daily_model_tokens)}` : ''}</span>
          </div>
        </div>
      </div>

      {remove.error && <Notice tone="error">{remove.error.message}</Notice>}

      {collections.isPending ? (
        <div className="collection-grid">
          {[0, 1, 2].map((i) => (
            <div key={i} className="card-tile skeleton-card"><Skeleton lines={4} /></div>
          ))}
        </div>
      ) : collections.error ? (
        <Notice
          tone="error"
          action={<Button variant="ghost" size="sm" onClick={() => void collections.refetch()}>重试</Button>}
        >
          {collections.error.message}
        </Notice>
      ) : collections.data?.length ? (
        <>
          {collections.data.length > 3 && (
            <div className="toolbar">
              <div className="search-box">
                <Search size={16} />
                <input
                  type="search"
                  aria-label="搜索知识库"
                  placeholder="搜索知识库名称或描述…"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                />
              </div>
            </div>
          )}
          <div className="collection-grid">
            {filtered.map((item) => {
              const count = counts.get(item.id)
              return (
                <article className={`collection-card accent-${hashIndex(item.id, 5)}`} key={item.id}>
                  <div className="collection-top">
                    <div className="folder-icon">
                      <FolderOpen size={22} />
                    </div>
                    <button
                      className="icon-button collection-delete"
                      aria-label={`删除 ${item.name}`}
                      title="删除知识库"
                      disabled={remove.isPending}
                      onClick={() => {
                        if (confirm(`确定删除知识库“${item.name}”？`)) remove.mutate(item.id)
                      }}
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>
                  <h3>
                    <button className="collection-link" onClick={() => onOpen(item)}>
                      {item.name}
                    </button>
                  </h3>
                  <p className="collection-desc">{item.description || '为你的下一个研究问题积累线索。'}</p>
                  <div className="collection-footer">
                    <span className="collection-count">
                      <FileText size={13} />
                      {count ? `${count.total} 篇论文` : '暂无论文'}
                    </span>
                    <span>创建于 {formatDate(item.created_at)}</span>
                  </div>
                </article>
              )
            })}
            {!search && (
              <button type="button" className="card-tile collection-new" onClick={() => setCreating(true)}>
                <Plus size={22} />
                <span>新的研究方向</span>
              </button>
            )}
          </div>
          {filtered.length === 0 && (
            <EmptyState compact icon={<Search size={30} />} title="没有匹配的知识库">
              换个关键词试试。
            </EmptyState>
          )}
        </>
      ) : (
        <EmptyState
          icon={<FolderOpen size={40} strokeWidth={1.4} />}
          title="从一个研究方向开始"
          action={
            <Button variant="outline" onClick={() => setCreating(true)}>
              <Plus size={16} />
              创建知识库
            </Button>
          }
        >
          创建你的第一个知识库，上传 PDF 论文，
          <br />
          然后在问答工作台中基于原文提出问题。
        </EmptyState>
      )}

      {creating && (
        <Modal labelId="create-title" title="新建知识库" description="按研究方向组织你的文献。" onClose={() => setCreating(false)}>
          <form
            onSubmit={(e) => {
              e.preventDefault()
              create.mutate()
            }}
          >
            <div className="field">
              <label htmlFor="collection-name">名称</label>
              <input
                autoFocus
                id="collection-name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                required
                maxLength={100}
                placeholder="例如：检索增强生成"
              />
            </div>
            <div className="field">
              <label htmlFor="collection-description">描述（可选）</label>
              <textarea
                id="collection-description"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                maxLength={2000}
                placeholder="记录这个方向的研究主题…"
              />
            </div>
            {create.error && <Notice tone="error">{create.error.message}</Notice>}
            <div className="modal-actions">
              <Button type="button" variant="ghost" onClick={() => setCreating(false)}>
                取消
              </Button>
              <Button disabled={create.isPending}>{create.isPending ? '创建中…' : '创建'}</Button>
            </div>
          </form>
        </Modal>
      )}
    </section>
  )
}

import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ArrowLeft, BookOpen, Layers, LocateFixed, MessageSquare, Search } from 'lucide-react'
import { api, apiText, type ChunkInfo, type Paper, type ParsedDocumentInfo } from '../lib/api'
import { locationsToHighlight, type PdfHighlight } from '../lib/workbenchHelpers'
import { PdfPreview } from './PdfPreview'
import { SafeMarkdown } from './SafeMarkdown'
import { Button } from './ui/button'
import { EmptyState, Notice, Skeleton } from './ui/primitives'

const kindLabels: Record<string, string> = {
  abstract: '摘要',
  document_meta: '文档概览',
  figure: '图文联合',
  text: '正文',
  table: '表格',
  formula: '公式',
  caption: '图注/表注',
}

export function PaperDetail({
  paper,
  onBack,
  onAskPaper,
}: {
  paper: Paper
  onBack: () => void
  onAskPaper?: (paper: Paper) => void
}) {
  const [leftTab, setLeftTab] = useState<'markdown' | 'chunks'>('markdown')
  const [highlight, setHighlight] = useState<PdfHighlight | null>(null)
  const [selectedLocations, setSelectedLocations] = useState<ChunkInfo['locations']>([])
  const [selectedChunk, setSelectedChunk] = useState<string | null>(null)
  const [kindFilter, setKindFilter] = useState<string>('all')
  const [chunkSearch, setChunkSearch] = useState('')

  const markdownQuery = useQuery({
    queryKey: ['paper-markdown', paper.id, paper.active_version_id],
    queryFn: () => apiText(`/papers/${paper.id}/markdown`),
    enabled: !!paper.active_version_id,
  })

  const docQuery = useQuery({
    queryKey: ['paper-document', paper.id, paper.active_version_id],
    queryFn: () => api<ParsedDocumentInfo>(`/papers/${paper.id}/document`),
    enabled: !!paper.active_version_id,
  })

  const chunksQuery = useQuery({
    queryKey: ['paper-chunks', paper.id, paper.active_version_id],
    queryFn: () => api<ChunkInfo[]>(`/papers/${paper.id}/chunks`),
    enabled: !!paper.active_version_id,
  })

  const kinds = useMemo(() => {
    const counts = new Map<string, number>()
    for (const c of chunksQuery.data || []) counts.set(c.kind, (counts.get(c.kind) || 0) + 1)
    return Array.from(counts.entries())
  }, [chunksQuery.data])

  const visibleChunks = (chunksQuery.data || [])
    .map((chunk, idx) => ({ chunk, idx }))
    .filter(({ chunk }) => kindFilter === 'all' || chunk.kind === kindFilter)
    .filter(({ chunk }) => {
      const q = chunkSearch.trim().toLowerCase()
      return !q || chunk.text.toLowerCase().includes(q) || chunk.section_path.join(' ').toLowerCase().includes(q)
    })

  function locateChunk(chunk: ChunkInfo, pageIdx?: number) {
    setSelectedChunk(chunk.id)
    setSelectedLocations(chunk.locations)
    setHighlight(
      locationsToHighlight(
        paper.id,
        chunk.locations,
        pageIdx,
        `${kindLabels[chunk.kind] || chunk.kind} · ${chunk.section_path.join(' / ') || '正文'}`,
      ),
    )
  }

  const meta = [
    paper.authors.join(' · ') || '作者待补充',
    paper.year ? String(paper.year) : null,
    paper.venue,
    paper.doi ? `DOI: ${paper.doi}` : null,
    paper.arxiv_id ? `arXiv: ${paper.arxiv_id}` : null,
    `${paper.page_count} 页`,
  ].filter(Boolean)

  return (
    <section className="paper-detail-page" data-testid="paper-detail-page">
      <div className="detail-header">
        <div className="detail-header-main">
          <button className="icon-button detail-back" aria-label="返回论文列表" title="返回论文列表" onClick={onBack}>
            <ArrowLeft size={18} />
          </button>
          <div className="detail-heading">
            <h1 title={paper.title}>{paper.title}</h1>
            <p className="detail-meta">{meta.join(' · ')}</p>
          </div>
        </div>
        <div className="detail-header-side">
          {docQuery.data && (
            <div className="detail-stats">
              <span><strong>{docQuery.data.sections.length}</strong> 章节</span>
              <span><strong>{docQuery.data.blocks.length}</strong> 结构块</span>
              <span><strong>{chunksQuery.data?.length ?? '—'}</strong> 检索分块</span>
              <span><strong>{docQuery.data.assets.length}</strong> 插图</span>
            </div>
          )}
          {onAskPaper && paper.active_version_id && (
            <Button onClick={() => onAskPaper(paper)}>
              <MessageSquare size={16} />
              针对此论文提问
            </Button>
          )}
        </div>
      </div>

      <div className="detail-split">
        <div className="detail-pane detail-left">
          <div className="pane-tabs" role="tablist">
            <button
              type="button"
              role="tab"
              aria-selected={leftTab === 'markdown'}
              className={`pane-tab ${leftTab === 'markdown' ? 'active' : ''}`}
              onClick={() => setLeftTab('markdown')}
            >
              <BookOpen size={15} /> 解析全文
              <span className="sr-only">（Markdown / 公式 / 表格）</span>
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={leftTab === 'chunks'}
              className={`pane-tab ${leftTab === 'chunks' ? 'active' : ''}`}
              onClick={() => setLeftTab('chunks')}
            >
              <Layers size={15} /> <span className="sr-only">结构化分块与 PDF 定位</span>
              <span aria-hidden="true">分块定位</span>
              <span className="tab-count">{chunksQuery.data?.length ?? 0}</span>
            </button>
          </div>

          <div className="pane-scroll">
            {!paper.active_version_id ? (
              <EmptyState compact icon={<BookOpen size={28} />} title="尚未完成入库">
                论文尚未完成入库，暂时无法查看解析全文。
              </EmptyState>
            ) : leftTab === 'markdown' ? (
              <>
                {paper.abstract && (
                  <details className="abstract-box">
                    <summary>摘要</summary>
                    <p>{paper.abstract}</p>
                  </details>
                )}
                {markdownQuery.isPending ? (
                  <Skeleton lines={8} />
                ) : markdownQuery.error ? (
                  <Notice tone="error">{markdownQuery.error.message}</Notice>
                ) : (
                  <SafeMarkdown content={markdownQuery.data || ''} paperId={paper.id} />
                )}
              </>
            ) : chunksQuery.isPending ? (
              <Skeleton lines={8} />
            ) : chunksQuery.error ? (
              <Notice tone="error">{chunksQuery.error.message}</Notice>
            ) : (
              <>
                <div className="chunk-toolbar">
                  <div className="search-box small">
                    <Search size={14} />
                    <input
                      type="search"
                      aria-label="搜索分块"
                      placeholder="搜索分块内容或章节…"
                      value={chunkSearch}
                      onChange={(e) => setChunkSearch(e.target.value)}
                    />
                  </div>
                  <div className="chip-group" role="group" aria-label="按类型筛选">
                    <button
                      type="button"
                      className={`chip ${kindFilter === 'all' ? 'active' : ''}`}
                      onClick={() => setKindFilter('all')}
                    >
                      全部
                    </button>
                    {kinds.map(([kind, count]) => (
                      <button
                        key={kind}
                        type="button"
                        className={`chip ${kindFilter === kind ? 'active' : ''}`}
                        onClick={() => setKindFilter(kind)}
                      >
                        {kindLabels[kind] || kind} <span>{count}</span>
                      </button>
                    ))}
                  </div>
                </div>
                <div className="chunk-list">
                  {visibleChunks.map(({ chunk, idx }) => {
                    const firstPage = chunk.locations[0]?.page_idx
                    const hasBbox = chunk.locations.some((l) => l.bbox && l.bbox.length === 4)
                    const locateLabel =
                      (firstPage !== undefined ? `定位第 ${firstPage + 1} 页` : '无 PDF 定位') +
                      (firstPage !== undefined ? (hasBbox ? '（含高亮框）' : '（仅跳页）') : '')
                    return (
                      <article
                        key={chunk.id}
                        className={`chunk-card ${selectedChunk === chunk.id ? 'selected' : ''} ${chunk.indexable ? '' : 'unindexed'}`}
                      >
                        <div className="chunk-card-header">
                          <span className="chunk-badge">#{idx + 1}</span>
                          <span className={`chunk-kind kind-${chunk.kind}`}>{kindLabels[chunk.kind] || chunk.kind}</span>
                          <span className="chunk-section" title={chunk.section_path.join(' › ')}>
                            {chunk.section_path.join(' › ') || '未归属章节'}
                          </span>
                          {chunk.chunk_metadata?.section_role && (
                            <span className="chunk-role">{chunk.chunk_metadata.section_role}</span>
                          )}
                          <Button
                            variant="outline"
                            size="sm"
                            type="button"
                            aria-label={locateLabel}
                            title={locateLabel}
                            disabled={!chunk.locations.length}
                            onClick={() => locateChunk(chunk)}
                          >
                            <LocateFixed size={14} />
                            {firstPage !== undefined ? `P${firstPage + 1}` : '—'}
                          </Button>
                        </div>
                        {!chunk.indexable && (
                          <span className="chunk-unindexed">
                            未索引（{chunk.unindexed_reason || '超长或非检索章节'}）
                          </span>
                        )}
                        <SafeMarkdown content={chunk.text} paperId={paper.id} className="markdown-body chunk-markdown" />
                      </article>
                    )
                  })}
                  {visibleChunks.length === 0 && <p className="muted center">没有符合条件的分块。</p>}
                </div>
              </>
            )}
          </div>
        </div>

        <div className="detail-pane detail-right">
          <PdfPreview
            paperId={paper.id}
            title={paper.title}
            highlight={highlight}
            allLocations={selectedLocations}
            onSelectPageIdx={(pageIdx) => {
              if (selectedLocations.length > 0) {
                setHighlight(locationsToHighlight(paper.id, selectedLocations, pageIdx, highlight?.label))
              }
            }}
          />
        </div>
      </div>
    </section>
  )
}

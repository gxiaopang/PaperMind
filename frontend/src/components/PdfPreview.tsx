import { useEffect, useRef, useState } from 'react'
import { Document, Page, pdfjs } from 'react-pdf'
import { ChevronLeft, ChevronRight, ExternalLink, ZoomIn, ZoomOut } from 'lucide-react'
import { Button } from './ui/button'
import { Spinner } from './ui/primitives'
import type { Location } from '../lib/api'
import type { PdfHighlight } from '../lib/workbenchHelpers'
import 'react-pdf/dist/Page/AnnotationLayer.css'
import 'react-pdf/dist/Page/TextLayer.css'

const pdfWorkerUrl = new URL(
  'pdfjs-dist/build/pdf.worker.min.mjs',
  import.meta.url,
)
// Replace responses cached before the worker's JavaScript MIME type was fixed.
pdfWorkerUrl.searchParams.set('v', `${pdfjs.version}-module`)
pdfjs.GlobalWorkerOptions.workerSrc = pdfWorkerUrl.toString()

export function PdfPreview({
  paperId,
  title,
  initialPage = 1,
  highlight,
  allLocations,
  onSelectPageIdx,
  onPageChange,
}: {
  paperId: string
  title?: string
  initialPage?: number
  highlight?: PdfHighlight | null
  allLocations?: Location[]
  onSelectPageIdx?: (pageIdx: number) => void
  onPageChange?: (pageIdx: number) => void
}) {
  const [numPages, setNumPages] = useState<number>(0)
  const [pageNumber, setPageNumber] = useState<number>(Math.max(1, initialPage))
  const [pageInput, setPageInput] = useState<string>(String(Math.max(1, initialPage)))
  const [scale, setScale] = useState<number>(1.0)
  const [loadError, setLoadError] = useState<string>('')
  const containerRef = useRef<HTMLDivElement>(null)
  const bboxLayerRef = useRef<HTMLDivElement>(null)
  const [containerWidth, setContainerWidth] = useState<number>(520)

  useEffect(() => {
    if (highlight && highlight.paperId === paperId && highlight.pageNumber >= 1) {
      setPageNumber(highlight.pageNumber)
    }
  }, [highlight, paperId])

  useEffect(() => {
    setPageInput(String(pageNumber))
  }, [pageNumber])

  useEffect(() => {
    if (!containerRef.current) return
    const updateWidth = () => {
      if (containerRef.current) {
        const w = containerRef.current.clientWidth - 40
        if (w > 220) setContainerWidth(Math.min(w, 900))
      }
    }
    updateWidth()
    const observer = new ResizeObserver(updateWidth)
    observer.observe(containerRef.current)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    if (numPages > 0 && pageNumber <= numPages) onPageChange?.(pageNumber - 1)
  }, [numPages, pageNumber, onPageChange])

  const activeBboxes =
    highlight && highlight.paperId === paperId && highlight.pageNumber === pageNumber
      ? highlight.bboxes
      : []

  function scrollToHighlight() {
    const box = bboxLayerRef.current?.querySelector<HTMLElement>('.pdf-bbox-highlight')
    box?.scrollIntoView({ block: 'center', behavior: 'smooth' })
  }

  const distinctPageIdxs = allLocations
    ? Array.from(new Set(allLocations.map((l) => l.page_idx))).sort((a, b) => a - b)
    : []

  const pdfUrl = `/api/v1/papers/${paperId}/pdf`

  function commitPageInput() {
    const val = Number.parseInt(pageInput, 10)
    if (Number.isFinite(val) && val >= 1 && (!numPages || val <= numPages)) setPageNumber(val)
    else setPageInput(String(pageNumber))
  }

  return (
    <div className="pdf-preview" data-testid="pdf-preview">
      <div className="pdf-toolbar">
        <div className="pdf-toolbar-left">
          {title && <strong className="pdf-doc-title" title={title}>{title}</strong>}
          <span className="pdf-page-indicator" data-testid="pdf-page-indicator">
            第 {pageNumber} / {numPages || '—'} 页
          </span>
        </div>
        <div className="pdf-toolbar-controls">
          <Button
            variant="ghost"
            size="icon"
            type="button"
            aria-label="上一页"
            disabled={pageNumber <= 1}
            onClick={() => setPageNumber((p) => Math.max(1, p - 1))}
          >
            <ChevronLeft size={16} />
          </Button>
          <input
            type="text"
            inputMode="numeric"
            aria-label="跳转页码"
            className="pdf-page-input"
            value={pageInput}
            onChange={(e) => setPageInput(e.target.value.replace(/[^0-9]/g, ''))}
            onBlur={commitPageInput}
            onKeyDown={(e) => {
              if (e.key === 'Enter') commitPageInput()
            }}
          />
          <Button
            variant="ghost"
            size="icon"
            type="button"
            aria-label="下一页"
            disabled={numPages > 0 && pageNumber >= numPages}
            onClick={() => setPageNumber((p) => (numPages ? Math.min(numPages, p + 1) : p + 1))}
          >
            <ChevronRight size={16} />
          </Button>
          <span className="toolbar-divider" />
          <Button
            variant="ghost"
            size="icon"
            type="button"
            aria-label="缩小"
            disabled={scale <= 0.6}
            onClick={() => setScale((s) => Math.max(0.6, Number((s - 0.15).toFixed(2))))}
          >
            <ZoomOut size={15} />
          </Button>
          <button type="button" className="pdf-zoom-label" title="重置缩放" onClick={() => setScale(1)}>
            {Math.round(scale * 100)}%
          </button>
          <Button
            variant="ghost"
            size="icon"
            type="button"
            aria-label="放大"
            disabled={scale >= 2}
            onClick={() => setScale((s) => Math.min(2.0, Number((s + 0.15).toFixed(2))))}
          >
            <ZoomIn size={15} />
          </Button>
          <a
            className="button button-ghost button-icon"
            href={pdfUrl}
            target="_blank"
            rel="noreferrer"
            aria-label="新窗口打开"
            title="新窗口打开"
          >
            <ExternalLink size={15} />
          </a>
        </div>
      </div>

      {(highlight && highlight.paperId === paperId && highlight.pageNumber === pageNumber) || distinctPageIdxs.length > 1 ? (
        <div className="pdf-location-bar">
          {highlight && highlight.paperId === paperId && highlight.pageNumber === pageNumber && (
            <span
              className={`pdf-highlight-status ${activeBboxes.length > 0 ? 'has-bbox' : 'page-only'}`}
              data-testid="pdf-highlight-status"
            >
              {activeBboxes.length > 0
                ? `${highlight.label ? highlight.label + ' · ' : ''}已高亮 ${activeBboxes.length} 处区域`
                : `${highlight.label ? highlight.label + ' · ' : ''}页级定位（无坐标框）`}
            </span>
          )}
          {distinctPageIdxs.length > 1 && (
            <div className="pdf-location-pills">
              <span>跨页：</span>
              {distinctPageIdxs.map((idx) => (
                <button
                  key={idx}
                  type="button"
                  className={`location-pill ${pageNumber === idx + 1 ? 'active' : ''}`}
                  onClick={() => {
                    setPageNumber(idx + 1)
                    onSelectPageIdx?.(idx)
                  }}
                >
                  第 {idx + 1} 页
                </button>
              ))}
            </div>
          )}
        </div>
      ) : null}

      <div className="pdf-canvas-area" ref={containerRef}>
        {loadError ? (
          <div className="notice error" role="alert">
            PDF 预览加载失败：{loadError}
          </div>
        ) : (
          <Document
            file={pdfUrl}
            onLoadSuccess={({ numPages: total }) => {
              setNumPages(total)
              setLoadError('')
              if (pageNumber > total && total > 0) setPageNumber(total)
            }}
            onLoadError={(err) => setLoadError(err.message || '无法读取 PDF 文件')}
            loading={<div className="pdf-status"><Spinner label="正在加载 PDF…" /></div>}
          >
            <div className="pdf-page-wrapper" data-testid="pdf-page-wrapper">
              <Page
                pageNumber={pageNumber}
                width={Math.round(containerWidth * scale)}
                renderTextLayer={true}
                renderAnnotationLayer={false}
                loading={<div className="pdf-status"><Spinner /></div>}
                onRenderSuccess={scrollToHighlight}
              />
              {activeBboxes.length > 0 && (
                <div className="pdf-bbox-layer" data-testid="pdf-bbox-layer" ref={bboxLayerRef}>
                  {activeBboxes.map(([x0, y0, x1, y1], index) => (
                    <div
                      key={`${index}-${x0}-${y0}-${x1}-${y1}`}
                      className="pdf-bbox-highlight"
                      data-testid="pdf-bbox-box"
                      style={{
                        left: `${(x0 * 100).toFixed(2)}%`,
                        top: `${(y0 * 100).toFixed(2)}%`,
                        width: `${((x1 - x0) * 100).toFixed(2)}%`,
                        height: `${((y1 - y0) * 100).toFixed(2)}%`,
                      }}
                    />
                  ))}
                </div>
              )}
            </div>
          </Document>
        )}
      </div>
    </div>
  )
}

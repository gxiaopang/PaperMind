import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeRaw from 'rehype-raw'
import rehypeSanitize, { defaultSchema } from 'rehype-sanitize'
import rehypeKatex from 'rehype-katex'
import { isSafeExternalHref, resolvePaperAssetSrc } from '../lib/workbenchHelpers'
import 'katex/dist/katex.min.css'

const sanitizeSchema = {
  ...defaultSchema,
  tagNames: Array.from(
    new Set([
      ...(defaultSchema.tagNames || []),
      'table',
      'thead',
      'tbody',
      'tr',
      'th',
      'td',
      'caption',
      'sup',
      'sub',
      'span',
      'div',
      'img',
    ]),
  ),
  attributes: {
    ...defaultSchema.attributes,
    td: [...(defaultSchema.attributes?.td || []), 'colSpan', 'rowSpan', 'colspan', 'rowspan', 'align'],
    th: [...(defaultSchema.attributes?.th || []), 'colSpan', 'rowSpan', 'colspan', 'rowspan', 'align'],
    span: [...(defaultSchema.attributes?.span || []), ['className', 'math', 'math-inline', 'math-display']],
    div: [...(defaultSchema.attributes?.div || []), ['className', 'math', 'math-display']],
    code: [...(defaultSchema.attributes?.code || []), ['className', /^language-/, 'math-inline', 'math-display']],
    img: ['src', 'alt', 'title'],
    a: ['href', 'title'],
  },
  protocols: {
    ...defaultSchema.protocols,
    href: ['http', 'https', 'mailto'],
    src: ['http', 'https'],
  },
}

export function SafeMarkdown({
  content,
  paperId,
  className = 'markdown-body',
}: {
  content: string
  paperId?: string
  className?: string
}) {
  return (
    <div className={className}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeRaw, [rehypeSanitize, sanitizeSchema], rehypeKatex]}
        components={{
          a: ({ href, children }) => {
            if (!isSafeExternalHref(href)) {
              return <span className="unsafe-link-disabled">{children}</span>
            }
            return (
              <a href={href} target="_blank" rel="noopener noreferrer">
                {children}
              </a>
            )
          },
          img: ({ src, alt, title }) => {
            const resolved = resolvePaperAssetSrc(src, paperId)
            if (!resolved) {
              return <span className="muted">[图片资源不可用：{alt || '图片'}]</span>
            }
            return <img src={resolved} alt={alt || '论文插图'} title={title} loading="lazy" className="paper-markdown-img" />
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}

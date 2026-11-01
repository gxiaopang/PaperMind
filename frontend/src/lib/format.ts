export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`
}

export function formatNumber(value: number): string {
  return value.toLocaleString('zh-CN')
}

export function formatCompact(value: number): string {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(value >= 10_000_000 ? 0 : 1)}M`
  if (value >= 10_000) return `${(value / 1000).toFixed(0)}K`
  return value.toLocaleString('zh-CN')
}

export function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString('zh-CN')
}

export function relativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime()
  const minutes = Math.round(diff / 60000)
  if (minutes < 1) return '刚刚'
  if (minutes < 60) return `${minutes} 分钟前`
  const hours = Math.round(minutes / 60)
  if (hours < 24) return `${hours} 小时前`
  const days = Math.round(hours / 24)
  if (days < 7) return `${days} 天前`
  return formatDate(iso)
}

export function percent(used: number, limit: number): number {
  if (!limit || limit <= 0) return 0
  return Math.min(100, Math.max(0, (used / limit) * 100))
}

export const paperStatusLabels: Record<string, string> = {
  pending: '等待入库',
  uploading: '上传至云端',
  parsing: '云端解析中',
  chunking: '分块中',
  embedding: '生成向量中',
  ready: '已入库',
  failed: '入库失败',
}

export const paperStatusProgress: Record<string, number> = {
  pending: 8,
  uploading: 22,
  parsing: 45,
  chunking: 70,
  embedding: 88,
  ready: 100,
}

/** Stable accent index for a string, used to tint collection cards. */
export function hashIndex(value: string, modulo: number): number {
  let hash = 0
  for (let i = 0; i < value.length; i++) hash = (hash * 31 + value.charCodeAt(i)) | 0
  return Math.abs(hash) % modulo
}


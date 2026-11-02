import { test, expect } from '@playwright/test'
import {
  applySseEvent,
  isSafeExternalHref,
  locationsToHighlight,
  memoryKindLabels,
  resolvePaperAssetSrc,
  splitAnswerWithCitations,
  validBbox,
  type RunProgressState,
} from '../src/lib/workbenchHelpers'
import { parseRoute, routeToHash } from '../src/lib/router'

test('单元校验：相对坐标 bbox 归一化、安全链接与资源路径、引用切分与 SSE 重生成草稿状态机', async () => {
  expect(validBbox([0.1, 0.2, 0.8, 0.9])).toBe(true)
  expect(validBbox([0.8, 0.2, 0.1, 0.9])).toBe(false)
  expect(validBbox([-0.1, 0.2, 0.5, 0.9])).toBe(false)
  expect(validBbox(null)).toBe(false)

  const withBbox = locationsToHighlight('p1', [
    { page_idx: 0, bbox: [0.1, 0.15, 0.85, 0.45] },
    { page_idx: 1, bbox: null },
  ])
  expect(withBbox.pageNumber).toBe(1)
  expect(withBbox.bboxes).toEqual([[0.1, 0.15, 0.85, 0.45]])

  const withoutBbox = locationsToHighlight(
    'p1',
    [
      { page_idx: 0, bbox: [0.1, 0.15, 0.85, 0.45] },
      { page_idx: 1, bbox: null },
    ],
    1,
  )
  expect(withoutBbox.pageNumber).toBe(2)
  expect(withoutBbox.bboxes).toEqual([])

  expect(resolvePaperAssetSrc('images/fig1.png', 'paper-123')).toBe(
    '/api/v1/papers/paper-123/assets/images/fig1.png',
  )
  expect(resolvePaperAssetSrc('../secret.png', 'paper-123')).toBeNull()
  expect(resolvePaperAssetSrc('javascript:alert(1)', 'paper-123')).toBeNull()
  expect(isSafeExternalHref('https://arxiv.org/abs/1706.03762')).toBe(true)
  expect(isSafeExternalHref('javascript:alert(1)')).toBe(false)

  expect(splitAnswerWithCitations('缩放点积注意力[1]结合实验表格[2]。')).toEqual([
    { type: 'text', value: '缩放点积注意力' },
    { type: 'cite', number: 1 },
    { type: 'text', value: '结合实验表格' },
    { type: 'cite', number: 2 },
    { type: 'text', value: '。' },
  ])

  let state: RunProgressState = {
    runId: 'r1',
    question: 'q',
    mode: 'agentic',
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
  }

  state = applySseEvent(state, 'regenerate', { id: 1, attempt: 1, draft: true })
  state = applySseEvent(state, 'token', { id: 2, attempt: 1, text: '第一版未带引用的草稿文本。' })
  // Duplicate event ID 2 must be ignored
  state = applySseEvent(state, 'token', { id: 2, attempt: 1, text: '重复片段' })
  expect(state.draftText).toBe('第一版未带引用的草稿文本。')
  expect(state.generationCount).toBe(1)

  // Second attempt regenerate clears draft
  state = applySseEvent(state, 'regenerate', { id: 3, attempt: 2, draft: true })
  expect(state.draftText).toBe('')
  expect(state.generationCount).toBe(2)

  state = applySseEvent(state, 'token', { id: 4, attempt: 2, text: '修正后的第二版草稿[1]。' })
  state = applySseEvent(state, 'answer', { id: 5, attempt: 2, text: '修正后的第二版草稿[1]。', status: 'answered' })
  state = applySseEvent(state, 'done', {
    id: 6,
    attempt: 2,
    status: 'answered',
    usage: { tokens: 256, estimated: false },
    used_memories: [{ id: 'm1', kind: 'goal', level: 'project', epistemic: 'user_statement', content: '研究目标' }],
    memory_events: [{ id: 'ev1', project_id: 'p1', conversation_id: 'c1', run_id: 'r1', memory_id: 'm1', target: 'entry', action: 'add', actor: 'agent', summary: '已记住：研究目标', before_state: {}, after_state: {}, undone_at: null, created_at: '2026-11-02T00:00:00Z' }],
  })
  expect(state.finalText).toBe('修正后的第二版草稿[1]。')
  expect(state.status).toBe('completed')
  expect(state.usageTokens).toBe(256)
  expect(state.usedMemories).toHaveLength(1)
  expect(state.memoryEvents).toHaveLength(1)
  expect(memoryKindLabels.goal).toBe('研究目标')
  expect(parseRoute('#/memory')).toEqual({ tab: 'memory' })
  expect(routeToHash({ tab: 'memory' })).toBe('#/memory')
})


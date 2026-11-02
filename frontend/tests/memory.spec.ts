import { test, expect } from '@playwright/test'
import type { Conversation, MemoryEntry, ResearchProject } from '../src/lib/api'

test('研究记忆：确认候选、修正假设、保存回答证据，并在新会话延续项目', async ({ page }) => {
  const now = '2026-11-02T00:00:00Z'
  const projects: ResearchProject[] = [{ id: 'p1', name: '低秩适配研究', description: '核对消融证据', created_at: now }]
  const conversations: Conversation[] = [{ id: 'c1', project_id: 'p1', title: '方法比较', scope: {}, active_paper_ids: [], created_at: now }]
  let entries: MemoryEntry[] = [{ id: 'm1', level: 'project', project_id: 'p1', conversation_id: null, paper_id: null, kind: 'goal', content: '核对低秩适配的实验边界', status: 'candidate', epistemic: 'user_statement', pinned: false, citation_ids: [], evidence: [], source_message_id: null, evidence_status: 'unlinked', created_at: now, updated_at: now }]
  const savedBodies: Array<Record<string, unknown>> = []
  await page.route('**/api/v1/**', async route => {
    const request = route.request()
    const url = new URL(request.url())
    const path = url.pathname.replace('/api/v1', '')
    const method = request.method()
    const body = request.postDataJSON() as Record<string, unknown> | null
    let result: unknown = []
    let status = 200
    if (path === '/auth/me') result = { id: 'u1', email: 'memory@example.org', role: 'user', status: 'active', email_verified_at: now }
    else if (path === '/auth/csrf') result = { csrf_token: 'test-csrf' }
    else if (path === '/usage') result = { papers: 1, storage_bytes: 0, model_tokens: 0, parse_pages: 0, limits: { papers: 100, storage_bytes: 10000, daily_model_tokens: 10000, daily_parse_pages: 100 } }
    else if (path === '/memory/settings') result = { memory_write_mode: 'auto' }
    else if (path === '/memory/core') result = { profile: { level: 'user', project_id: null, project_name: null, content: '', char_limit: 1500, char_count: 0, revision: 0, updated_at: null }, project: { level: 'project', project_id: 'p1', project_name: '低秩适配研究', content: '核对消融证据', char_limit: 3000, char_count: 6, revision: 0, updated_at: now } }
    else if (path === '/memory/events') result = []
    else if (path === '/projects') {
      if (method === 'POST') { const item = { id: 'p2', name: String(body?.name), description: String(body?.description), created_at: now }; projects.push(item); result = item; status = 201 }
      else result = projects
    } else if (path === '/conversations') {
      if (method === 'POST') { const item: Conversation = { id: 'c2', project_id: body?.project_id as string, title: '新会话', scope: {}, active_paper_ids: [], created_at: now }; conversations.unshift(item); result = item; status = 201 }
      else result = conversations
    } else if (path.startsWith('/conversations/') && method === 'PATCH' && !path.endsWith('/memory')) {
      const item = conversations.find(c => c.id === path.split('/')[2])!; Object.assign(item, body); result = item
    } else if (path.endsWith('/messages')) result = path.includes('/c1/') ? [{ id: 'answer1', seq: 2, run_id: 'r1', role: 'assistant', content: '此实验采用低秩更新[1]。', rewritten_query: null, verify_result: {}, feedback: null, citations: [{ id: 'citation1', number: 1, paper_id: 'paper1', version_id: 'v1', chunk_id: 'chunk1', title: 'LoRA', text: '低秩更新', locations: [{ page_idx: 0 }], deleted: false }] }] : []
    else if (path.endsWith('/memory')) result = { summary: '用户正在核对实验。', research_state: { open_question: [{ content: '需要核对消融', source_seq: 1 }] }, summarized_seq: 2, active_paper_ids: [], project_id: 'p1' }
    else if (path === '/memories' && method === 'POST') {
      savedBodies.push(body!)
      const item: MemoryEntry = { ...entries[0], ...body, id: 'm2', pinned: false, evidence: [], evidence_status: 'current', created_at: now, updated_at: now }
      entries.push(item); result = item; status = 201
    } else if (path === '/memories') {
      const cid = url.searchParams.get('conversation_id')
      const pid = cid ? conversations.find(c => c.id === cid)?.project_id : url.searchParams.get('project_id')
      result = entries.filter(m => m.level === 'user' || m.project_id === pid)
    } else if (path.startsWith('/memories/')) {
      const id = path.split('/').at(-1)
      if (method === 'DELETE') { entries = entries.filter(m => m.id !== id); status = 204 }
      else { const item = entries.find(m => m.id === id)!; Object.assign(item, body); result = item }
    }
    await route.fulfill({ status, contentType: 'application/json', body: status === 204 ? undefined : JSON.stringify(result) })
  })
  await page.goto('/#/chat')
  await expect(page.getByLabel('研究项目', { exact: true })).toHaveValue('p1')
  await page.getByRole('button', { name: '研究记忆', exact: true }).click()
  const dialog = page.getByRole('dialog')
  await expect(dialog.getByText('核对低秩适配的实验边界', { exact: true })).toBeVisible()
  await dialog.getByRole('button', { name: '确认记忆', exact: true }).click()
  await expect(dialog.getByText(/当前项目 · 研究目标 · 已确认/)).toBeVisible()
  await dialog.getByRole('button', { name: '固定', exact: true }).click()
  await expect(dialog.getByRole('button', { name: '取消固定', exact: true })).toBeVisible()
  await dialog.getByRole('button', { name: '编辑', exact: true }).click()
  await dialog.getByLabel('编辑记忆内容').fill('需重新验证，尚不能下结论')
  await dialog.getByRole('button', { name: '保存修改', exact: true }).click()
  await dialog.getByRole('button', { name: '标为假设', exact: true }).click()
  await expect(dialog.getByText(/待验证假设 · 未关联论文证据/)).toBeVisible()
  await dialog.getByRole('button', { name: '关闭对话框' }).click()
  await page.getByRole('button', { name: '保存为研究记忆' }).click()
  await dialog.getByLabel('记忆内容', { exact: true }).fill('低秩更新的实验设置需要继续比较')
  await dialog.getByRole('button', { name: '保存记忆', exact: true }).click()
  await expect(dialog.getByText('低秩更新的实验设置需要继续比较', { exact: true })).toBeVisible()
  await expect.poll(() => savedBodies.length).toBe(1)
  expect(savedBodies[0]).toMatchObject({ project_id: 'p1', source_message_id: 'answer1', citation_ids: ['citation1'], epistemic: 'evidence_linked' })
  await dialog.getByRole('button', { name: '关闭对话框' }).click()
  await page.getByRole('button', { name: '新建会话', exact: true }).click()
  await page.getByRole('button', { name: '研究记忆', exact: true }).click()
  await expect(dialog.getByText('需重新验证，尚不能下结论', { exact: true })).toBeVisible()
  await dialog.getByLabel('当前研究项目').selectOption('')
  await dialog.getByLabel('项目名称').fill('另一个研究项目')
  await dialog.getByLabel('项目描述').fill('独立的研究目标')
  await dialog.getByRole('button', { name: '新建并使用项目' }).click()
  await expect(dialog.getByLabel('当前研究项目')).toHaveValue('p2')
  await expect(dialog.getByText('需重新验证，尚不能下结论', { exact: true })).toHaveCount(0)
})

function twoPagePdf(): Buffer {
  const bodies = [
    '<< /Type /Catalog /Pages 2 0 R >>',
    '<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>',
    '<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] /Resources << >> >>',
    '<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] /Resources << >> >>',
  ]
  let pdf = '%PDF-1.4\n'
  const offsets = [0]
  bodies.forEach((body, i) => { offsets.push(Buffer.byteLength(pdf)); pdf += `${i + 1} 0 obj\n${body}\nendobj\n` })
  const xref = Buffer.byteLength(pdf)
  pdf += `xref\n0 5\n0000000000 65535 f \n${offsets.slice(1).map(offset => `${String(offset).padStart(10, '0')} 00000 n \n`).join('')}trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF\n`
  return Buffer.from(pdf)
}

test('论文阅读位置：记录第二页并从第一页返回', async ({ page }) => {
  let position: { version_id: string; page_idx: number } | null = null
  const paper = { id: 'paper1', collection_id: 'library1', title: '阅读位置测试论文', authors: [], year: null, page_count: 2, size_bytes: 500, status: 'ready', active_version_id: 'v1', pending_version_id: null, error: null, abstract: null, venue: null, doi: null, arxiv_id: null }
  await page.route('**/api/v1/**', async route => {
    const path = new URL(route.request().url()).pathname.replace('/api/v1', '')
    if (path.endsWith('/pdf')) { await route.fulfill({ status: 200, contentType: 'application/pdf', body: twoPagePdf() }); return }
    if (path.endsWith('/markdown')) { await route.fulfill({ status: 200, contentType: 'text/plain', body: '# 阅读位置测试' }); return }
    let data: unknown = []
    if (path === '/auth/me') data = { id: 'u1', email: 'memory@example.org', role: 'user', status: 'active', email_verified_at: '2026-11-02' }
    else if (path === '/auth/csrf') data = { csrf_token: 'test-csrf' }
    else if (path === '/collections') data = [{ id: 'library1', name: '阅读测试库', description: '', created_at: '2026-11-02' }]
    else if (path === '/usage') data = { model_tokens: 0, papers: 1, limits: { daily_model_tokens: 10000, papers: 100 } }
    else if (path === '/collections/library1/papers' || path === '/papers') data = [paper]
    else if (path.endsWith('/document')) data = { page_count: 2, sections: [], blocks: [], assets: [], warnings: [] }
    else if (path.endsWith('/reading-position')) {
      if (route.request().method() === 'PUT') position = route.request().postDataJSON()
      data = { position, current: Boolean(position) }
    }
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(data) })
  })
  await page.goto('/#/library/library1/paper/paper1')
  await expect(page.getByTestId('pdf-page-indicator')).toContainText('第 1 / 2 页')
  await page.getByRole('button', { name: '下一页', exact: true }).click()
  await expect(page.getByTestId('pdf-page-indicator')).toContainText('第 2 / 2 页')
  await page.getByRole('button', { name: '记住阅读位置', exact: true }).click()
  await expect.poll(() => position?.page_idx).toBe(1)
  await expect(page.getByRole('button', { name: '回到第 2 页' })).toBeVisible()
  await page.getByRole('button', { name: '上一页', exact: true }).click()
  await expect(page.getByTestId('pdf-page-indicator')).toContainText('第 1 / 2 页')
  await page.getByRole('button', { name: '回到第 2 页' }).click()
  await expect(page.getByTestId('pdf-page-indicator')).toContainText('第 2 / 2 页')
})

test('研究记忆中心与对话透明化：核心记忆编辑、批量治理、跨会话检索与一键撤销', async ({ page }) => {
  const now = '2026-11-02T00:00:00Z'
  const projects: ResearchProject[] = [{ id: 'p1', name: '低秩适配研究', description: '核对消融证据', created_at: now }]
  const conversations: Conversation[] = [{ id: 'c1', project_id: 'p1', title: '方法比较', scope: {}, active_paper_ids: [], created_at: now }]
  const entries: MemoryEntry[] = [
    { id: 'm1', level: 'project', project_id: 'p1', conversation_id: null, paper_id: null, kind: 'goal', content: '对比 LoRA 与 DoRA 的收敛稳定性', status: 'candidate', epistemic: 'user_statement', pinned: false, citation_ids: [], evidence: [], source_message_id: null, evidence_status: 'unlinked', created_at: now, updated_at: now },
  ]
  let profileContent = '- 偏好严谨数学推导'
  let eventUndone = false
  await page.route('**/api/v1/**', async route => {
    const request = route.request()
    const url = new URL(request.url())
    const path = url.pathname.replace('/api/v1', '')
    const method = request.method()
    const body = request.postDataJSON() as Record<string, unknown> | null
    let result: unknown = []
    if (path === '/auth/me') result = { id: 'u1', email: 'memory@example.org', role: 'user', status: 'active', email_verified_at: now }
    else if (path === '/auth/csrf') result = { csrf_token: 'test-csrf' }
    else if (path === '/usage') result = { papers: 1, storage_bytes: 0, model_tokens: 0, parse_pages: 0, limits: { papers: 100, storage_bytes: 10000, daily_model_tokens: 10000, daily_parse_pages: 100 } }
    else if (path === '/projects') result = projects
    else if (path === '/conversations') result = conversations
    else if (path.endsWith('/messages')) {
      result = [{
        id: 'answer1',
        seq: 2,
        run_id: 'r1',
        role: 'assistant',
        content: '此实验采用低秩更新[1]。',
        rewritten_query: null,
        verify_result: {
          used_memories: [{ id: 'm1', kind: 'goal', level: 'project', epistemic: 'user_statement', content: '对比 LoRA 与 DoRA 的收敛稳定性' }],
          memory_events: [{ id: 'ev1', project_id: 'p1', conversation_id: 'c1', run_id: 'r1', memory_id: 'm1', target: 'entry', action: 'add', actor: 'agent', summary: '已记住：对比 LoRA 与 DoRA 的收敛稳定性', before_state: {}, after_state: {}, undone_at: eventUndone ? now : null, created_at: now }],
        },
        feedback: null,
        citations: [{ id: 'citation1', number: 1, paper_id: 'paper1', version_id: 'v1', chunk_id: 'chunk1', title: 'LoRA', text: '低秩更新', locations: [{ page_idx: 0 }], deleted: false }],
      }]
    } else if (path === '/memory/settings') {
      result = { memory_write_mode: body?.memory_write_mode || 'auto' }
    } else if (path === '/memory/core') {
      if (method === 'PUT') {
        profileContent = String(body?.content || '')
        result = { level: 'user', project_id: null, project_name: null, content: profileContent, char_limit: 1500, char_count: profileContent.length, revision: 1, updated_at: now }
      } else {
        result = {
          profile: { level: 'user', project_id: null, project_name: null, content: profileContent, char_limit: 1500, char_count: profileContent.length, revision: 1, updated_at: now },
          project: { level: 'project', project_id: 'p1', project_name: '低秩适配研究', content: '- 核对消融证据', char_limit: 3000, char_count: 8, revision: 1, updated_at: now },
        }
      }
    } else if (path === '/memories') {
      result = entries
    } else if (path === '/memories/bulk') {
      entries.forEach(item => { if ((body?.ids as string[])?.includes(item.id) && body?.action === 'confirm') item.status = 'active' })
      result = { updated: 1 }
    } else if (path === '/memory/sessions/search') {
      result = [{ message_id: 'msg1', conversation_id: 'c1', conversation_title: '方法比较', project_id: 'p1', seq: 1, role: 'user', snippet: '历史讨论：DoRA 在低秩下的梯度分解细节', created_at: now }]
    } else if (path === '/memory/events') {
      result = [{ id: 'ev1', project_id: 'p1', conversation_id: 'c1', run_id: 'r1', memory_id: 'm1', target: 'entry', action: 'add', actor: 'agent', summary: '已记住：对比 LoRA 与 DoRA 的收敛稳定性', before_state: {}, after_state: {}, undone_at: eventUndone ? now : null, created_at: now }]
    } else if (path === '/memory/events/ev1/undo') {
      eventUndone = true
      result = { id: 'ev1', undone_at: now }
    }
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(result) })
  })

  await page.goto('/#/chat')
  await expect(page.getByText('已记住：对比 LoRA 与 DoRA 的收敛稳定性')).toBeVisible()
  await page.getByRole('button', { name: /参考了 1 条研究记忆/ }).click()
  await expect(page.getByText('对比 LoRA 与 DoRA 的收敛稳定性').first()).toBeVisible()

  await page.getByRole('button', { name: '研究记忆中心', exact: true }).click()
  const center = page.getByTestId('memory-center-page')
  await expect(center.getByRole('heading', { name: '研究记忆中心', exact: true })).toBeVisible()
  await center.getByLabel('记忆中心研究者画像', { exact: true }).fill('- 偏好严谨数学推导与消融对比')
  await center.getByRole('button', { name: '保存研究者画像', exact: true }).click()
  await expect.poll(() => profileContent).toBe('- 偏好严谨数学推导与消融对比')

  await center.getByRole('button', { name: '档案记忆与研究判断', exact: true }).click()
  await center.getByRole('checkbox', { name: /当前项目 · 研究目标 · 待确认/ }).check()
  await center.getByRole('button', { name: '批量确认', exact: true }).click()
  await expect(center.getByText(/当前项目 · 研究目标 · 已确认/)).toBeVisible()

  await center.getByRole('button', { name: '跨会话原文检索', exact: true }).click()
  await center.getByLabel('跨会话检索关键词', { exact: true }).fill('DoRA')
  await center.getByRole('button', { name: '检索历史会话', exact: true }).click()
  await expect(center.getByText('历史讨论：DoRA 在低秩下的梯度分解细节')).toBeVisible()

  await center.getByRole('button', { name: '记忆活动日志与撤销', exact: true }).click()
  await center.getByRole('button', { name: '撤销', exact: true }).click()
  await expect.poll(() => eventUndone).toBe(true)
  await expect(center.getByText('自动记住 · add · 已撤销', { exact: true })).toBeVisible()
  await expect(center.getByRole('button', { name: '撤销', exact: true })).toHaveCount(0)
})

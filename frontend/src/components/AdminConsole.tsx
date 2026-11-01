import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Activity, Award, Server, Users } from 'lucide-react'
import { api, type User } from '../lib/api'
import { formatNumber } from '../lib/format'
import { Button } from './ui/button'
import { Modal } from './ui/modal'
import { Meter, Notice, PageHeader } from './ui/primitives'

type AdminOverview = {
  system: {
    app_env: string
    tracing_enabled: boolean
    llm_model: string
    llm_fast_model: string
    embedding_model: string
    rerank_model: string
    provider_concurrency: number
  }
  users: {
    total: number
    active: number
    disabled: number
  }
  global_usage: {
    papers: number
    storage_bytes: number
    parse_pages: number
    parse_pages_reserved: number
    daily_parse_pages_limit: number
    model_tokens: number
    model_tokens_reserved: number
    model_tokens_estimated: boolean
    daily_model_tokens_limit: number
    quota_timezone: string
  }
  queues: {
    outbox: {
      ingest: number
      cleanup: number
      chat: number
      summarize: number
      mail: number
    }
    active_tasks: {
      ingest: number
      chat: number
    }
    redis_queued: Record<string, number>
  }
  evaluations: {
    retrieval: {
      status?: string
      dataset?: string
      cases?: number
      variants?: Record<
        string,
        {
          hit_at_8: number
          mrr_at_8: number
          precision_at_8: number
          recall_at_8: number
        }
      >
    } | null
    chat: {
      status?: string
      dataset?: string
      single_final_run_all_passed: boolean
      scenarios: Array<{
        case: string
        passed: boolean
        elapsed_s: number
      }>
    } | null
  }
}

type AdminUserRow = {
  id: string
  email: string
  role: string
  status: 'active' | 'disabled'
  email_verified_at: string | null
  created_at: string
  usage: {
    papers: number
    storage_bytes: number
    parse_pages: number
    parse_pages_reserved: number
    model_tokens: number
    model_tokens_reserved: number
  }
  quota_overrides: {
    max_papers: number | null
    storage_mb: number | null
    daily_parse_pages: number | null
    daily_model_tokens: number | null
    upload_concurrency: number | null
  }
  effective_limits: {
    max_papers: number
    storage_mb: number
    daily_parse_pages: number
    daily_model_tokens: number
    upload_concurrency: number
  }
}

export function AdminConsole({ currentUser }: { currentUser: User }) {
  const client = useQueryClient()
  const [editingUser, setEditingUser] = useState<AdminUserRow | null>(null)
  const [maxPapers, setMaxPapers] = useState('')
  const [storageMb, setStorageMb] = useState('')
  const [dailyPages, setDailyPages] = useState('')
  const [dailyTokens, setDailyTokens] = useState('')
  const [uploadConcurrency, setUploadConcurrency] = useState('')

  const overviewQuery = useQuery({
    queryKey: ['admin-overview'],
    queryFn: () => api<AdminOverview>('/admin/overview'),
  })

  const usersQuery = useQuery({
    queryKey: ['admin-users'],
    queryFn: () => api<AdminUserRow[]>('/admin/users'),
  })

  const statusMutation = useMutation({
    mutationFn: ({ id, status }: { id: string; status: 'active' | 'disabled' }) =>
      api(`/admin/users/${id}`, 'PATCH', { status }),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ['admin-users'] })
      void client.invalidateQueries({ queryKey: ['admin-overview'] })
    },
  })

  const quotaMutation = useMutation({
    mutationFn: (id: string) =>
      api(`/admin/users/${id}/quota`, 'PUT', {
        max_papers: maxPapers.trim() ? Number.parseInt(maxPapers.trim(), 10) : null,
        storage_mb: storageMb.trim() ? Number.parseInt(storageMb.trim(), 10) : null,
        daily_parse_pages: dailyPages.trim() ? Number.parseInt(dailyPages.trim(), 10) : null,
        daily_model_tokens: dailyTokens.trim() ? Number.parseInt(dailyTokens.trim(), 10) : null,
        upload_concurrency: uploadConcurrency.trim() ? Number.parseInt(uploadConcurrency.trim(), 10) : null,
      }),
    onSuccess: () => {
      setEditingUser(null)
      void client.invalidateQueries({ queryKey: ['admin-users'] })
    },
  })

  function openQuotaModal(u: AdminUserRow) {
    setEditingUser(u)
    setMaxPapers(u.quota_overrides.max_papers ? String(u.quota_overrides.max_papers) : '')
    setStorageMb(u.quota_overrides.storage_mb ? String(u.quota_overrides.storage_mb) : '')
    setDailyPages(u.quota_overrides.daily_parse_pages ? String(u.quota_overrides.daily_parse_pages) : '')
    setDailyTokens(u.quota_overrides.daily_model_tokens ? String(u.quota_overrides.daily_model_tokens) : '')
    setUploadConcurrency(u.quota_overrides.upload_concurrency ? String(u.quota_overrides.upload_concurrency) : '')
  }

  const ov = overviewQuery.data

  return (
    <section className="page admin-console-page" data-testid="admin-console-page">
      <PageHeader
        eyebrow="OPERATIONS & GOVERNANCE"
        title="系统管理控制台"
        description="管理员仅可管理全局配额、账户状态、队列运行指标与离线评测报告；不授予访问他人私有论文或会话正文的权限。"
      />

      {ov && (
        <div className="admin-cards-grid">
          <section className="panel settings-card">
            <div className="panel-heading">
              <Server size={19} />
              <div>
                <h2>全局用量与系统状态</h2>
                <p className="muted">
                  主模型：{ov.system.llm_model} · 快速模型：{ov.system.llm_fast_model} · Langfuse：
                  {ov.system.tracing_enabled ? '已开启' : '关闭'}
                </p>
              </div>
            </div>
            <div className="admin-metrics-list">
              <div>
                <span>注册账户数</span>
                <strong>
                  {ov.users.total}（正常 {ov.users.active} / 停用 {ov.users.disabled}）
                </strong>
              </div>
              <Meter
                label="全局今日解析页数"
                used={ov.global_usage.parse_pages}
                reserved={ov.global_usage.parse_pages_reserved}
                limit={ov.global_usage.daily_parse_pages_limit}
                display={`${ov.global_usage.parse_pages}（预占 ${ov.global_usage.parse_pages_reserved}）/ ${ov.global_usage.daily_parse_pages_limit} 页`}
              />
              <Meter
                label="全局今日模型 Token"
                used={ov.global_usage.model_tokens}
                reserved={ov.global_usage.model_tokens_reserved}
                limit={ov.global_usage.daily_model_tokens_limit}
                display={`${formatNumber(ov.global_usage.model_tokens)}（预占 ${formatNumber(ov.global_usage.model_tokens_reserved)}）/ ${formatNumber(ov.global_usage.daily_model_tokens_limit)}`}
              />
            </div>
          </section>

          <section className="panel settings-card">
            <div className="panel-heading">
              <Activity size={19} />
              <div>
                <h2>ARQ 队列与任务积压状态</h2>
                <p className="muted">邮件、入库、问答三类独立队列与 Outbox 待投递统计。</p>
              </div>
            </div>
            <div className="admin-metrics-list" data-testid="admin-queue-status">
              <div>
                <span>正在执行任务</span>
                <strong>
                  入库 {ov.queues.active_tasks.ingest} · 问答 {ov.queues.active_tasks.chat}
                </strong>
              </div>
              <div>
                <span>Outbox 待完成</span>
                <strong>
                  入库 {ov.queues.outbox.ingest} · 问答 {ov.queues.outbox.chat} · 摘要 {ov.queues.outbox.summarize} · 邮件 {ov.queues.outbox.mail}
                </strong>
              </div>
              <div>
                <span>Redis 队列长度</span>
                <strong>
                  ingest: {ov.queues.redis_queued['papermind:ingest'] ?? 0} · chat: {ov.queues.redis_queued['papermind:chat'] ?? 0}
                </strong>
              </div>
            </div>
          </section>
        </div>
      )}

      {ov?.evaluations && (
        <section className="panel settings-card admin-eval-section" data-testid="admin-eval-section">
          <div className="panel-heading">
            <Award size={19} />
            <div>
              <h2>离线评测与真实验收报告摘要</h2>
              <p className="muted">展示当前冻结集检索评测与真实供应商问答功能验证；回答内容质量另行验收。</p>
            </div>
          </div>

          <div className="eval-reports-grid">
            {ov.evaluations.retrieval?.variants && (
              <div>
                <h3>检索消融评测（{ov.evaluations.retrieval.dataset} · {ov.evaluations.retrieval.cases} 题）</h3>
                <div className="table-wrap">
                  <table className="usage-table">
                    <thead>
                      <tr>
                        <th>检索变体</th>
                        <th>Hit@8</th>
                        <th>MRR@8</th>
                        <th>Precision@8</th>
                        <th>Recall@8</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(ov.evaluations.retrieval.variants).map(([variant, m]) => (
                        <tr key={variant}>
                          <td><code>{variant}</code></td>
                          <td>{(m.hit_at_8 * 100).toFixed(1)}%</td>
                          <td>{m.mrr_at_8.toFixed(4)}</td>
                          <td>{m.precision_at_8.toFixed(4)}</td>
                          <td>{m.recall_at_8.toFixed(4)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {ov.evaluations.chat?.scenarios && (
              <div>
                <h3>问答真实冒烟（{ov.evaluations.chat.dataset}）</h3>
                <p className="muted small">
                  {ov.evaluations.chat.single_final_run_all_passed
                    ? '全部场景在同一次完整运行中通过。'
                    : '各场景分批通过；尚未验证一次完整运行全部通过，回答内容支持率与连续稳定性仍待验收。'}
                </p>
                <div className="table-wrap">
                  <table className="usage-table">
                    <thead>
                      <tr>
                        <th>测试场景</th>
                        <th>状态</th>
                        <th>耗时（秒）</th>
                      </tr>
                    </thead>
                    <tbody>
                      {ov.evaluations.chat.scenarios.map((c) => (
                        <tr key={c.case}>
                          <td>{c.case}</td>
                          <td>
                            <span className={`paper-status ${c.passed ? 'status-success' : 'status-danger'}`}>
                              {c.passed ? '通过' : '未通过'}
                            </span>
                          </td>
                          <td>{c.elapsed_s.toFixed(2)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        </section>
      )}

      <section className="panel settings-card admin-users-section" data-testid="admin-users-section">
        <div className="panel-heading">
          <Users size={19} />
          <div>
            <h2>账户与配额管理</h2>
            <p className="muted">启停用户账户（停用后立即撤销其活跃会话）或设置单用户专属配额上限。</p>
          </div>
        </div>

        {(statusMutation.error || quotaMutation.error) && (
          <Notice tone="error">{statusMutation.error?.message || quotaMutation.error?.message}</Notice>
        )}

        <div className="table-wrap">
          <table className="usage-table admin-users-table">
            <thead>
              <tr>
                <th>邮箱</th>
                <th>角色</th>
                <th>状态</th>
                <th>当前论文 / 页数 / Token 用量</th>
                <th>生效上限（论文 / 页 / Token）</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              {usersQuery.data?.map((u) => (
                <tr key={u.id} data-testid={`admin-user-row-${u.email}`}>
                  <td><strong>{u.email}</strong></td>
                  <td>{u.role === 'admin' ? '管理员' : '普通用户'}</td>
                  <td>
                    <span className={`paper-status ${u.status === 'disabled' ? 'status-danger' : 'status-success'}`}>
                      {u.status === 'active' ? '正常' : '已停用'}
                    </span>
                  </td>
                  <td>
                    {u.usage.papers} 篇 · {u.usage.parse_pages} 页 · {formatNumber(u.usage.model_tokens)}
                  </td>
                  <td>
                    {u.effective_limits.max_papers} 篇 / {u.effective_limits.daily_parse_pages} 页 /{' '}
                    {formatNumber(u.effective_limits.daily_model_tokens)}
                  </td>
                  <td className="admin-row-actions">
                    <Button variant="outline" size="sm" type="button" onClick={() => openQuotaModal(u)}>
                      调整配额
                    </Button>
                    {u.id !== currentUser.id && (
                      <Button
                        variant="ghost"
                        size="sm"
                        type="button"
                        disabled={statusMutation.isPending}
                        onClick={() =>
                          statusMutation.mutate({
                            id: u.id,
                            status: u.status === 'active' ? 'disabled' : 'active',
                          })
                        }
                      >
                        {u.status === 'active' ? '停用账户' : '恢复启用'}
                      </Button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {editingUser && (
        <Modal
          labelId="quota-modal-title"
          title={`调整用户配额：${editingUser.email}`}
          description="留空表示使用系统默认上限。"
          onClose={() => setEditingUser(null)}
        >
          <form
            onSubmit={(e) => {
              e.preventDefault()
              quotaMutation.mutate(editingUser.id)
            }}
          >
            <div className="form-grid-2">
              <div className="field">
                <label htmlFor="quota-max-papers">最大论文数</label>
                <input
                  id="quota-max-papers"
                  type="number"
                  min={1}
                  placeholder={`默认 ${editingUser.effective_limits.max_papers}`}
                  value={maxPapers}
                  onChange={(e) => setMaxPapers(e.target.value)}
                />
              </div>
              <div className="field">
                <label htmlFor="quota-storage-mb">私有存储上限（MB）</label>
                <input
                  id="quota-storage-mb"
                  type="number"
                  min={1}
                  placeholder={`默认 ${editingUser.effective_limits.storage_mb}`}
                  value={storageMb}
                  onChange={(e) => setStorageMb(e.target.value)}
                />
              </div>
              <div className="field">
                <label htmlFor="quota-daily-pages">每日解析页数上限</label>
                <input
                  id="quota-daily-pages"
                  type="number"
                  min={1}
                  placeholder={`默认 ${editingUser.effective_limits.daily_parse_pages}`}
                  value={dailyPages}
                  onChange={(e) => setDailyPages(e.target.value)}
                />
              </div>
              <div className="field">
                <label htmlFor="quota-daily-tokens">每日模型 Token 上限</label>
                <input
                  id="quota-daily-tokens"
                  type="number"
                  min={1}
                  placeholder={`默认 ${editingUser.effective_limits.daily_model_tokens}`}
                  value={dailyTokens}
                  onChange={(e) => setDailyTokens(e.target.value)}
                />
              </div>
            </div>
            <div className="modal-actions">
              <Button variant="ghost" type="button" onClick={() => setEditingUser(null)}>
                取消
              </Button>
              <Button disabled={quotaMutation.isPending}>保存配额</Button>
            </div>
          </form>
        </Modal>
      )}
    </section>
  )
}

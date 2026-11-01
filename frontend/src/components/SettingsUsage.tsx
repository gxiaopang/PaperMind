import { useEffect, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { BarChart3, Monitor, Moon, Sliders, Sun } from 'lucide-react'
import { api, type UsageData, type User, type UserSettings } from '../lib/api'
import { formatBytes, formatNumber } from '../lib/format'
import { useUI, type ThemePreference } from '../lib/store'
import { Button } from './ui/button'
import { Meter, Notice, PageHeader, Skeleton } from './ui/primitives'

const themeOptions: Array<{ value: ThemePreference; label: string; icon: typeof Sun }> = [
  { value: 'light', label: '浅色', icon: Sun },
  { value: 'dark', label: '深色', icon: Moon },
  { value: 'system', label: '跟随系统', icon: Monitor },
]

export function SettingsUsage({ user }: { user: User }) {
  const client = useQueryClient()
  const { theme, setTheme } = useUI()
  const [language, setLanguage] = useState<'zh' | 'en'>('zh')
  const [detailLevel, setDetailLevel] = useState<'concise' | 'standard' | 'detailed'>('standard')
  const [savedNotice, setSavedNotice] = useState('')

  const settingsQuery = useQuery({
    queryKey: ['user-settings', user.id],
    queryFn: () => api<UserSettings>('/settings'),
  })

  const usageQuery = useQuery({
    queryKey: ['usage', user.id],
    queryFn: () => api<UsageData>('/usage'),
  })

  useEffect(() => {
    if (settingsQuery.data) {
      setLanguage(settingsQuery.data.answer_language)
      setDetailLevel(settingsQuery.data.detail_level)
    }
  }, [settingsQuery.data])

  const saveMutation = useMutation({
    mutationFn: () =>
      api<UserSettings>('/settings', 'PUT', {
        answer_language: language,
        detail_level: detailLevel,
      }),
    onSuccess: (data) => {
      setLanguage(data.answer_language)
      setDetailLevel(data.detail_level)
      setSavedNotice('个人回答设置已保存')
      void client.invalidateQueries({ queryKey: ['user-settings', user.id] })
    },
  })

  const usage = usageQuery.data

  return (
    <section className="page settings-usage-page" data-testid="settings-usage-page">
      <PageHeader
        eyebrow="PREFERENCES & QUOTAS"
        title="个人设置与用量"
        description="管理你的问答偏好与界面主题，查看今日解析页数、模型 Token 与存储配额。"
      />

      <div className="settings-grid">
        <section className="panel settings-card">
          <div className="panel-heading">
            <Sliders size={19} />
            <div>
              <h2>问答偏好与外观</h2>
              <p className="muted">每次提交新问答时会自动记录并应用你当前的偏好设定。</p>
            </div>
          </div>

          <form
            onSubmit={(e) => {
              e.preventDefault()
              setSavedNotice('')
              saveMutation.mutate()
            }}
          >
            <div className="field">
              <label htmlFor="pref-language">默认回答语言</label>
              <select
                id="pref-language"
                className="form-select"
                value={language}
                onChange={(e) => setLanguage(e.target.value as 'zh' | 'en')}
              >
                <option value="zh">中文（默认）</option>
                <option value="en">English（英文）</option>
              </select>
            </div>

            <div className="field">
              <label htmlFor="pref-detail">回答详细程度</label>
              <select
                id="pref-detail"
                className="form-select"
                value={detailLevel}
                onChange={(e) => setDetailLevel(e.target.value as 'concise' | 'standard' | 'detailed')}
              >
                <option value="concise">简明扼要（聚焦核心结论与公式）</option>
                <option value="standard">标准适中（平衡可读性与细节）</option>
                <option value="detailed">详尽深入（展开推导与实验对比）</option>
              </select>
            </div>

            <div className="field">
              <span className="field-label">界面主题</span>
              <div className="theme-choices" role="radiogroup" aria-label="界面主题">
                {themeOptions.map(({ value, label, icon: Icon }) => (
                  <button
                    key={value}
                    type="button"
                    role="radio"
                    aria-checked={theme === value}
                    className={`theme-choice ${theme === value ? 'active' : ''}`}
                    onClick={() => setTheme(value)}
                  >
                    <Icon size={15} />
                    {label}
                  </button>
                ))}
              </div>
            </div>

            {savedNotice && <Notice tone="success">{savedNotice}</Notice>}
            {saveMutation.error && <Notice tone="error">{saveMutation.error.message}</Notice>}

            <Button disabled={saveMutation.isPending}>
              {saveMutation.isPending ? '保存中…' : '保存个人设置'}
            </Button>
          </form>
        </section>

        <section className="panel settings-card">
          <div className="panel-heading">
            <BarChart3 size={19} />
            <div>
              <h2>个人配额与实时用量</h2>
              <p className="muted">配额周期时区：{usage?.quota_timezone || 'UTC'}。</p>
            </div>
          </div>

          {usage ? (
            <div className="quota-meters" data-testid="quota-meters">
              <Meter
                label="已入库论文数量"
                used={usage.papers}
                limit={usage.limits.papers}
                display={`${usage.papers} / ${usage.limits.papers} 篇`}
              />

              <Meter
                label="私有存储占用"
                used={usage.storage_bytes}
                limit={usage.limits.storage_bytes}
                display={`${formatBytes(usage.storage_bytes)} / ${formatBytes(usage.limits.storage_bytes)}`}
              />

              <Meter
                label="今日 MinerU 解析页数"
                used={usage.parse_pages}
                reserved={usage.parse_pages_reserved}
                limit={usage.limits.daily_parse_pages}
                display={`已用 ${usage.parse_pages} 页（预占 ${usage.parse_pages_reserved} 页）/ ${usage.limits.daily_parse_pages} 页`}
              />

              <Meter
                label={`今日模型 Token 用量${usage.model_tokens_estimated ? '（含保守估算）' : '（供应商实报）'}`}
                used={usage.model_tokens}
                reserved={usage.model_tokens_reserved}
                limit={usage.limits.daily_model_tokens}
                display={`已用 ${formatNumber(usage.model_tokens)}（预占 ${formatNumber(usage.model_tokens_reserved)}）/ ${formatNumber(usage.limits.daily_model_tokens)}`}
              />

              {usage.by_node && usage.by_node.length > 0 && (
                <div className="node-usage-breakdown">
                  <h3>今日各节点模型调用明细</h3>
                  <div className="table-wrap">
                    <table className="usage-table">
                      <thead>
                        <tr>
                          <th>执行节点</th>
                          <th>模型</th>
                          <th>调用次数</th>
                          <th>Token 合计</th>
                        </tr>
                      </thead>
                      <tbody>
                        {usage.by_node.map((row) => (
                          <tr key={`${row.node}-${row.model}`}>
                            <td><code>{row.node}</code></td>
                            <td>{row.model}</td>
                            <td>{row.calls}</td>
                            <td>
                              {formatNumber(row.tokens)}
                              {row.estimated ? ' *' : ''}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <Skeleton lines={5} />
          )}
        </section>
      </div>
    </section>
  )
}

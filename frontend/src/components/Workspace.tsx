import { useEffect, useState, type ReactNode } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import {
  ChevronRight,
  Library as LibraryIcon,
  LogOut,
  Menu,
  MessageSquare,
  Monitor,
  Moon,
  PanelLeft,
  Server,
  ShieldCheck,
  Sliders,
  Sun,
} from 'lucide-react'
import { api, type Collection, type Paper, type UsageData, type User } from '../lib/api'
import { formatCompact, percent } from '../lib/format'
import { useRoute, type WorkspaceTab } from '../lib/router'
import { useUI } from '../lib/store'
import { AdminConsole } from './AdminConsole'
import { EmailCodeField } from './Auth'
import { Brand } from './Brand'
import { ChatWorkbench } from './ChatWorkbench'
import { Library } from './Library'
import { Papers } from './Papers'
import { SettingsUsage } from './SettingsUsage'
import { Button } from './ui/button'
import { Notice, Spinner } from './ui/primitives'

const tabTitles: Record<WorkspaceTab, string> = {
  library: '我的知识库',
  chat: '问答工作台',
  settings: '设置与用量',
  admin: '系统管理',
}

function NavItem({
  icon,
  label,
  active,
  count,
  onClick,
}: {
  icon: ReactNode
  label: string
  active: boolean
  count?: ReactNode
  onClick: () => void
}) {
  return (
    <button
      type="button"
      className={`nav-item ${active ? 'active' : ''}`}
      aria-current={active ? 'page' : undefined}
      title={label}
      onClick={onClick}
    >
      {icon}
      <span className="nav-text">{label}</span>
      {count !== undefined && <span className="nav-count">{count}</span>}
    </button>
  )
}

function ThemeToggle() {
  const { theme, setTheme } = useUI()
  const next = theme === 'light' ? 'dark' : theme === 'dark' ? 'system' : 'light'
  const label = { light: '浅色', dark: '深色', system: '跟随系统' }[theme]
  const Icon = theme === 'light' ? Sun : theme === 'dark' ? Moon : Monitor
  return (
    <button
      type="button"
      className="icon-button"
      aria-label={`切换主题（当前：${label}）`}
      title={`主题：${label}`}
      onClick={() => setTheme(next)}
    >
      <Icon size={17} />
    </button>
  )
}

export function Workspace({ user }: { user: User }) {
  const client = useQueryClient()
  const { sessionExpired, setSessionExpired, sidebarCollapsed, toggleSidebar, mobileNavOpen, setMobileNavOpen } = useUI()
  const [route, navigate] = useRoute()
  const [initialAskPaperId, setInitialAskPaperId] = useState<string | null>(null)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [verificationCode, setVerificationCode] = useState('')

  const tab: WorkspaceTab = route.tab === 'admin' && user.role !== 'admin' ? 'library' : route.tab

  const collections = useQuery({
    queryKey: ['collections', user.id],
    queryFn: () => api<Collection[]>('/collections'),
  })
  const usage = useQuery({
    queryKey: ['usage', user.id],
    queryFn: () => api<UsageData>('/usage'),
  })
  const selected = route.collectionId ? collections.data?.find((c) => c.id === route.collectionId) : undefined
  const paperQuery = useQuery({
    queryKey: ['papers', route.collectionId],
    queryFn: () => api<Paper[]>(`/collections/${route.collectionId}/papers`),
    enabled: !!selected && !!route.paperId,
  })
  const currentPaper = route.paperId ? paperQuery.data?.find((p) => p.id === route.paperId) : undefined

  useEffect(() => {
    if (route.collectionId && collections.data && !selected) navigate({ tab: 'library' })
  }, [route.collectionId, collections.data, selected, navigate])

  useEffect(() => {
    setMobileNavOpen(false)
  }, [route.tab, route.collectionId, route.paperId, setMobileNavOpen])

  async function logout() {
    try {
      await api('/auth/logout', 'POST')
      client.clear()
      location.assign('/')
    } catch (e) {
      setError((e as Error).message)
    }
  }

  async function verifyEmail() {
    try {
      setError('')
      const result = await api<{ message: string }>('/auth/verify-email-code', 'POST', { code: verificationCode })
      setNotice(result.message)
      await client.invalidateQueries({ queryKey: ['me'] })
    } catch (e) {
      setError((e as Error).message)
    }
  }

  function handleAskPaper(paper: Paper) {
    setInitialAskPaperId(paper.id)
    navigate({ tab: 'chat' })
  }

  const crumbs: Array<{ label: string; onClick?: () => void }> = [{ label: tabTitles[tab] }]
  if (tab === 'library' && selected) {
    crumbs[0].onClick = () => navigate({ tab: 'library' })
    crumbs.push({ label: selected.name })
    if (route.paperId) {
      crumbs[1].onClick = () => navigate({ tab: 'library', collectionId: selected.id })
      crumbs.push({ label: currentPaper?.title || '论文阅读' })
    }
  }

  const fill = tab === 'chat' || (tab === 'library' && !!route.paperId)
  const tokenPct = usage.data ? percent(usage.data.model_tokens, usage.data.limits.daily_model_tokens) : 0

  return (
    <div className={`app-shell ${sidebarCollapsed ? 'collapsed' : ''} ${mobileNavOpen ? 'nav-open' : ''}`}>
      <aside className="sidebar">
        <div className="sidebar-head">
          <Brand compact={sidebarCollapsed} />
          <button
            type="button"
            className="icon-button sidebar-toggle"
            aria-label={sidebarCollapsed ? '展开侧边栏' : '收起侧边栏'}
            onClick={toggleSidebar}
          >
            <PanelLeft size={17} />
          </button>
        </div>
        <div className="nav-label">研究空间</div>
        <nav className="sidebar-nav" aria-label="主导航">
          <NavItem
            icon={<LibraryIcon size={18} />}
            label="我的知识库"
            active={tab === 'library'}
            count={collections.data?.length ?? '—'}
            onClick={() => navigate({ tab: 'library' })}
          />
          <NavItem
            icon={<MessageSquare size={18} />}
            label="问答工作台"
            active={tab === 'chat'}
            onClick={() => navigate({ tab: 'chat' })}
          />
          <NavItem
            icon={<Sliders size={18} />}
            label="设置与用量"
            active={tab === 'settings'}
            onClick={() => navigate({ tab: 'settings' })}
          />
          {user.role === 'admin' && (
            <NavItem
              icon={<Server size={18} />}
              label="系统管理"
              active={tab === 'admin'}
              onClick={() => navigate({ tab: 'admin' })}
            />
          )}
        </nav>

        {usage.data && (
          <div className="sidebar-usage" title="今日模型 Token 用量">
            <div className="sidebar-usage-top">
              <span>今日额度</span>
              <strong>
                {formatCompact(usage.data.model_tokens)} / {formatCompact(usage.data.limits.daily_model_tokens)}
              </strong>
            </div>
            <div className="meter-track small">
              <span className="meter-fill" style={{ width: `${tokenPct}%` }} />
            </div>
          </div>
        )}

        <div className="sidebar-bottom">
          <div className="avatar" aria-hidden="true">{user.email[0].toUpperCase()}</div>
          <div className="account">
            <strong>{user.role === 'admin' ? '管理员' : '我的账户'}</strong>
            <span title={user.email}>{user.email}</span>
          </div>
          <div className="sidebar-bottom-actions">
            <ThemeToggle />
            <button className="icon-button" aria-label="退出登录" title="退出登录" onClick={() => void logout()}>
              <LogOut size={17} />
            </button>
          </div>
        </div>
      </aside>
      <div className="sidebar-scrim" onClick={() => setMobileNavOpen(false)} aria-hidden="true" />

      <main className="workspace-main">
        <header className="topbar">
          <button
            type="button"
            className="icon-button mobile-menu"
            aria-label="打开导航"
            onClick={() => setMobileNavOpen(true)}
          >
            <Menu size={19} />
          </button>
          <nav className="breadcrumbs" aria-label="当前位置">
            {crumbs.map((crumb, i) => (
              <span key={i} className="crumb">
                {i > 0 && <ChevronRight size={14} className="crumb-sep" />}
                {crumb.onClick ? (
                  <button type="button" className="crumb-link" onClick={crumb.onClick}>{crumb.label}</button>
                ) : (
                  <span className="crumb-current" aria-current="page">{crumb.label}</span>
                )}
              </span>
            ))}
          </nav>
          <span className="private-pill">
            <ShieldCheck size={14} /> 私人空间
          </span>
        </header>

        <div className={`workspace-content ${fill ? 'fill' : ''}`}>
          {sessionExpired && (
            <div className="banner banner-error" role="alert" data-testid="session-expired-modal">
              <ShieldCheck size={18} />
              <span>{sessionExpired}</span>
              <Button
                variant="outline"
                size="sm"
                type="button"
                onClick={() => {
                  setSessionExpired('会话已过期，请重新登录以继续操作。')
                  client.setQueryData(['me'], null)
                }}
              >
                重新登录
              </Button>
            </div>
          )}

          {!user.email_verified_at && (
            <div className="banner banner-warn">
              <ShieldCheck size={18} />
              <form onSubmit={e => { e.preventDefault(); void verifyEmail() }}>
                <p>请验证邮箱，以便后续上传论文和使用问答服务。</p>
                <EmailCodeField email={user.email} code={verificationCode} onChange={setVerificationCode} />
                <Button type="submit" size="sm">确认验证码</Button>
              </form>
            </div>
          )}
          {notice && <Notice tone="success">{notice}</Notice>}
          {error && <Notice tone="error">{error}</Notice>}

          {tab === 'chat' ? (
            <ChatWorkbench
              user={user}
              initialPaperId={initialAskPaperId}
              onClearInitialPaper={() => setInitialAskPaperId(null)}
            />
          ) : tab === 'settings' ? (
            <SettingsUsage user={user} />
          ) : tab === 'admin' ? (
            <AdminConsole currentUser={user} />
          ) : route.collectionId ? (
            selected ? (
              <Papers
                key={selected.id}
                collection={selected}
                user={user}
                paperId={route.paperId}
                onBack={() => navigate({ tab: 'library' })}
                onOpenPaper={(paperId) => navigate({ tab: 'library', collectionId: selected.id, paperId: paperId ?? undefined })}
                onAskPaper={handleAskPaper}
              />
            ) : (
              <div className="page-loading"><Spinner label="正在打开知识库…" /></div>
            )
          ) : (
            <Library user={user} onOpen={(c) => navigate({ tab: 'library', collectionId: c.id })} />
          )}
        </div>
      </main>
    </div>
  )
}


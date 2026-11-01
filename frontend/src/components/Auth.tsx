import { useEffect, useState, type FormEvent } from 'react'
import { useMutation } from '@tanstack/react-query'
import { ArrowRight, Eye, EyeOff, Layers, Quote, ShieldCheck, Sparkles } from 'lucide-react'
import { api, type User } from '../lib/api'
import { useUI } from '../lib/store'
import { Brand } from './Brand'
import { Button } from './ui/button'
import { Notice } from './ui/primitives'

export function EmailCodeField({ email, code, onChange }: {
  email: string
  code: string
  onChange: (value: string) => void
}) {
  const [remaining, setRemaining] = useState(0)
  const [sending, setSending] = useState(false)
  const [notice, setNotice] = useState('')
  const [error, setError] = useState('')

  useEffect(() => {
    if (remaining <= 0) return
    const timer = window.setTimeout(() => setRemaining(remaining - 1), 1000)
    return () => window.clearTimeout(timer)
  }, [remaining])

  async function sendCode() {
    setSending(true)
    setError('')
    setNotice('')
    try {
      const result = await api<{ message: string; resend_after: number }>(
        '/auth/registration-code', 'POST', { email },
      )
      setRemaining(result.resend_after)
      setNotice(result.message)
    } catch (err) {
      setError(err instanceof Error ? err.message : '验证码发送失败，请重试')
    } finally {
      setSending(false)
    }
  }

  return (
    <div className="field email-code-field">
      <label htmlFor="email-code">邮箱验证码</label>
      <div className="email-code-row">
        <input id="email-code" type="text" inputMode="numeric" autoComplete="one-time-code"
          placeholder="6 位数字验证码" pattern="[0-9]{6}" minLength={6} maxLength={6}
          value={code} onChange={e => onChange(e.target.value.replace(/[^0-9]/g, '').slice(0, 6))} required />
        <Button type="button" variant="outline" disabled={sending || remaining > 0 || !email}
          onClick={() => void sendCode()}>
          {sending ? '发送中…' : remaining > 0 ? `${remaining} 秒后重发` : '发送验证码'}
        </Button>
      </div>
      {notice && <Notice tone="success">{notice}</Notice>}
      {error && <Notice tone="error">{error}</Notice>}
    </div>
  )
}

function PasswordInput({ id, autoComplete, value, onChange }: {
  id: string
  autoComplete: string
  value: string
  onChange: (value: string) => void
}) {
  const [visible, setVisible] = useState(false)
  return (
    <div className="input-with-action">
      <input
        id={id}
        type={visible ? 'text' : 'password'}
        autoComplete={autoComplete}
        placeholder="至少 8 个字符"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        required
        minLength={8}
        maxLength={128}
      />
      <button
        type="button"
        className="input-action"
        aria-label={visible ? '隐藏密码' : '显示密码'}
        onClick={() => setVisible((v) => !v)}
      >
        {visible ? <EyeOff size={16} /> : <Eye size={16} />}
      </button>
    </div>
  )
}

const features = [
  { icon: Quote, title: '逐句引用溯源', text: '每个结论都附带 [n] 引用，一键跳到 PDF 原文并高亮。' },
  { icon: Layers, title: '结构化双栏阅读', text: '公式、表格、插图解析成 Markdown，与原文并排对照。' },
  { icon: Sparkles, title: 'Agentic 深度核验', text: '改写检索、相关性评估与事实依据检查，减少幻觉。' },
]

export function Auth({ onSuccess }: { onSuccess: () => void }) {
  const { view, setView, sessionExpired, setSessionExpired } = useUI()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [code, setCode] = useState('')
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const [pending, setPending] = useState(false)
  const title = { login: '欢迎回来', register: '开启你的研究空间', forgot: '找回密码' }[view]

  async function submit(event: FormEvent) {
    event.preventDefault()
    setPending(true)
    setError('')
    setMessage('')
    try {
      if (view === 'login') {
        await api<User>('/auth/login', 'POST', { email, password })
        setSessionExpired(null)
        onSuccess()
      } else {
        const result = await api<{ message: string }>(
          view === 'register' ? '/auth/register' : '/auth/forgot-password',
          'POST',
          view === 'register' ? { email, password, code } : { email },
        )
        setMessage(result.message)
        if (view === 'register') {
          setView('login')
          setCode('')
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '请求失败，请重试')
    } finally {
      setPending(false)
    }
  }

  function change(next: 'login' | 'register' | 'forgot') {
    setView(next)
    setError('')
    setMessage('')
  }

  return (
    <div className="auth-layout">
      <aside className="auth-story">
        <Brand />
        <div className="auth-story-main">
          <span className="eyebrow">AI RESEARCH COMPANION</span>
          <h1>
            读懂每一篇论文，
            <br />
            <em>每句回答都有出处。</em>
          </h1>
          <p>上传 PDF，PaperMind 帮你解析结构、建立知识库，并用可核验的引用回答你的研究问题。</p>
          <ul className="auth-features">
            {features.map(({ icon: Icon, title: t, text }) => (
              <li key={t}>
                <span className="auth-feature-icon"><Icon size={18} /></span>
                <div>
                  <strong>{t}</strong>
                  <span>{text}</span>
                </div>
              </li>
            ))}
          </ul>
        </div>
        <div className="auth-story-preview" aria-hidden="true">
          <div className="preview-q">缩放点积注意力为什么要除以 √d<sub>k</sub>？</div>
          <div className="preview-a">
            为避免点积数值过大导致 softmax 梯度极小<span className="preview-cite">1</span>，
            论文将其按 √d<sub>k</sub> 缩放<span className="preview-cite">2</span>。
          </div>
          <div className="preview-source">
            <span>[1] Attention Is All You Need · 第 4 页</span>
          </div>
        </div>
      </aside>

      <main className="auth-main">
        <div className="auth-box">
          <div className="auth-mobile-brand"><Brand /></div>
          <h2>{title}</h2>
          <p className="muted">
            {view === 'login'
              ? '登录账户，继续你的研究。'
              : view === 'register'
                ? '获取邮箱验证码，验证后即可创建账户。'
                : '输入注册邮箱，我们会向你发送重置链接。'}
          </p>
          {sessionExpired && (
            <Notice tone="error" testId="session-expired-banner">{sessionExpired}</Notice>
          )}
          <form onSubmit={submit} className="auth-form">
            <div className="field">
              <label htmlFor="email">邮箱地址</label>
              <input
                id="email"
                type="email"
                autoComplete="email"
                placeholder="you@university.edu"
                value={email}
                onChange={(e) => { setEmail(e.target.value); setCode('') }}
                required
                maxLength={320}
              />
            </div>
            {view === 'register' && <EmailCodeField key={email} email={email} code={code} onChange={setCode} />}
            {view !== 'forgot' && (
              <div className="field">
                <div className="label-row">
                  <label htmlFor="password">密码</label>
                  {view === 'login' && (
                    <button className="text-button" type="button" onClick={() => change('forgot')}>
                      忘记密码？
                    </button>
                  )}
                </div>
                <PasswordInput
                  id="password"
                  autoComplete={view === 'login' ? 'current-password' : 'new-password'}
                  value={password}
                  onChange={setPassword}
                />
              </div>
            )}
            {error && <Notice tone="error">{error}</Notice>}
            {message && <Notice tone="success">{message}</Notice>}
            <Button className="full" disabled={pending}>
              {pending ? '请稍候…' : view === 'login' ? '登录' : view === 'register' ? '创建账户' : '发送重置邮件'}
              <ArrowRight size={17} />
            </Button>
          </form>
          <p className="auth-switch">
            {view === 'login' ? '还没有账户？' : '已有账户？'}{' '}
            <button className="text-button" onClick={() => change(view === 'login' ? 'register' : 'login')}>
              {view === 'login' ? '免费注册' : '返回登录'}
            </button>
          </p>
          <div className="auth-note">
            <ShieldCheck size={15} />
            <span>你的知识库和研究资料仅对你可见</span>
          </div>
        </div>
      </main>
    </div>
  )
}

export function TokenAction({ action, token }: { action: string; token: string }) {
  const [password, setPassword] = useState('')
  const [done, setDone] = useState(false)
  const mutation = useMutation({
    mutationFn: () =>
      api<{ message: string }>(
        action === 'verify' ? '/auth/verify-email' : '/auth/reset-password',
        'POST',
        action === 'verify' ? { token } : { token, password },
      ),
    onSuccess: () => {
      history.replaceState(null, '', '/')
      setDone(true)
    },
  })
  return (
    <div className="center-page">
      <Brand />
      <div className="center-card">
        <span className="center-card-icon"><ShieldCheck size={26} /></span>
        <h2>{done ? '操作成功' : action === 'verify' ? '验证你的邮箱' : '设置新密码'}</h2>
        {done ? (
          <>
            <p className="muted">{mutation.data?.message}</p>
            <Button className="full" onClick={() => location.assign('/')}>返回 PaperMind</Button>
          </>
        ) : (
          <form
            onSubmit={(e) => {
              e.preventDefault()
              mutation.mutate()
            }}
          >
            {action !== 'verify' && (
              <div className="field">
                <label htmlFor="new-password">新密码</label>
                <PasswordInput id="new-password" autoComplete="new-password" value={password} onChange={setPassword} />
              </div>
            )}
            {mutation.error && <Notice tone="error">{mutation.error.message}</Notice>}
            <Button className="full" disabled={mutation.isPending}>
              {mutation.isPending ? '处理中…' : action === 'verify' ? '确认验证' : '重置密码'}
            </Button>
          </form>
        )}
      </div>
    </div>
  )
}


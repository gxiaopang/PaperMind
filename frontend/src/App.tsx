import { useEffect } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { api, ApiError, type User } from './lib/api'
import { useHash } from './lib/router'
import { useUI } from './lib/store'
import { Auth, TokenAction } from './components/Auth'
import { Brand } from './components/Brand'
import { Workspace } from './components/Workspace'
import { Button } from './components/ui/button'
import { Spinner } from './components/ui/primitives'

function useThemeEffect() {
  const theme = useUI((s) => s.theme)
  useEffect(() => {
    const media = window.matchMedia('(prefers-color-scheme: dark)')
    const apply = () => {
      const resolved = theme === 'system' ? (media.matches ? 'dark' : 'light') : theme
      document.documentElement.dataset.theme = resolved
      document.querySelector('meta[name="theme-color"]')?.setAttribute('content', resolved === 'dark' ? '#0f1412' : '#f6f7f5')
    }
    apply()
    media.addEventListener('change', apply)
    return () => media.removeEventListener('change', apply)
  }, [theme])
}

export function App() {
  useThemeEffect()
  const client = useQueryClient()
  const { sessionExpired } = useUI()
  const hash = useHash()
  const user = useQuery({ queryKey: ['me'], queryFn: () => api<User>('/auth/me') })
  const params = new URLSearchParams(hash.slice(1))
  const action = params.get('action'),
    token = params.get('token')
  if ((action === 'verify' || action === 'reset') && token) return <TokenAction action={action} token={token} />
  if (user.isPending)
    return (
      <div className="center-page">
        <Brand />
        <Spinner label="正在打开你的研究空间…" />
      </div>
    )
  if (user.error && !(user.error instanceof ApiError && user.error.status === 401))
    return (
      <div className="center-page">
        <Brand />
        <p role="alert" className="muted">{user.error.message}</p>
        <Button onClick={() => void user.refetch()}>重新连接</Button>
      </div>
    )
  return user.data && !sessionExpired ? (
    <Workspace user={user.data} />
  ) : (
    <Auth
      onSuccess={() => {
        void client.invalidateQueries({ queryKey: ['me'] })
      }}
    />
  )
}

import { useCallback, useSyncExternalStore } from 'react'

export type WorkspaceTab = 'library' | 'chat' | 'memory' | 'settings' | 'admin'

export type Route = {
  tab: WorkspaceTab
  collectionId?: string
  paperId?: string
}

const TABS: WorkspaceTab[] = ['library', 'chat', 'memory', 'settings', 'admin']

/** Parse `#/library/<cid>/paper/<pid>`, `#/chat` … Legacy token hashes fall back to the library. */
export function parseRoute(hash: string): Route {
  if (!hash.startsWith('#/')) return { tab: 'library' }
  const [tab, collectionId, kind, paperId] = hash.slice(2).split('/').map(decodeURIComponent)
  if (!TABS.includes(tab as WorkspaceTab)) return { tab: 'library' }
  if (tab !== 'library' || !collectionId) return { tab: tab as WorkspaceTab }
  return { tab: 'library', collectionId, paperId: kind === 'paper' && paperId ? paperId : undefined }
}

export function routeToHash(route: Route): string {
  let hash = `#/${route.tab}`
  if (route.tab === 'library' && route.collectionId) {
    hash += `/${encodeURIComponent(route.collectionId)}`
    if (route.paperId) hash += `/paper/${encodeURIComponent(route.paperId)}`
  }
  return hash
}

function subscribe(callback: () => void) {
  window.addEventListener('hashchange', callback)
  return () => window.removeEventListener('hashchange', callback)
}

export function useHash(): string {
  return useSyncExternalStore(subscribe, () => location.hash)
}

export function useRoute(): [Route, (route: Route) => void] {
  const hash = useHash()
  const navigate = useCallback((route: Route) => {
    const next = routeToHash(route)
    if (location.hash !== next) location.hash = next
  }, [])
  return [parseRoute(hash), navigate]
}


import { create } from 'zustand'

type View = 'login' | 'register' | 'forgot'
export type ThemePreference = 'light' | 'dark' | 'system'

function readStorage<T extends string>(key: string, allowed: readonly T[], fallback: T): T {
  try {
    const value = localStorage.getItem(key)
    return value && (allowed as readonly string[]).includes(value) ? (value as T) : fallback
  } catch {
    return fallback
  }
}

function writeStorage(key: string, value: string) {
  try {
    localStorage.setItem(key, value)
  } catch {
    // Storage may be unavailable (private mode); preferences then last for the session only.
  }
}

export const useUI = create<{
  view: View
  setView: (view: View) => void
  sessionExpired: string | null
  setSessionExpired: (message: string | null) => void
  theme: ThemePreference
  setTheme: (theme: ThemePreference) => void
  sidebarCollapsed: boolean
  toggleSidebar: () => void
  mobileNavOpen: boolean
  setMobileNavOpen: (open: boolean) => void
}>((set, get) => ({
  view: 'login',
  setView: (view) => set({ view }),
  sessionExpired: null,
  setSessionExpired: (sessionExpired) => set({ sessionExpired }),
  theme: readStorage('pm-theme', ['light', 'dark', 'system'] as const, 'system'),
  setTheme: (theme) => {
    writeStorage('pm-theme', theme)
    set({ theme })
  },
  sidebarCollapsed: readStorage('pm-sidebar', ['collapsed', 'expanded'] as const, 'expanded') === 'collapsed',
  toggleSidebar: () => {
    const next = !get().sidebarCollapsed
    writeStorage('pm-sidebar', next ? 'collapsed' : 'expanded')
    set({ sidebarCollapsed: next })
  },
  mobileNavOpen: false,
  setMobileNavOpen: (mobileNavOpen) => set({ mobileNavOpen }),
}))

import { defineConfig, devices } from '@playwright/test'

export default defineConfig({
  testDir: './tests', fullyParallel: false, workers: 1,
  use: { baseURL: 'http://127.0.0.1:8081', trace: 'retain-on-failure', screenshot: 'only-on-failure' },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: { command: 'npm run dev -- --port 8081', url: 'http://127.0.0.1:8081', reuseExistingServer: false, env: { PAPERMIND_API_TARGET: 'http://127.0.0.1:18001' } },
})

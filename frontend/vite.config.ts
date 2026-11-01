import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { proxy: { '/api': { target: process.env.PAPERMIND_API_TARGET || 'http://127.0.0.1:18000', changeOrigin: false } } },
})

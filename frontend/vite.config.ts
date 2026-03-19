import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/ws/resources': {
        target: 'ws://localhost:8765',
        ws: true,
        rewriteWsOrigin: true,
        rewrite: () => '/',
      },
      '/ws/agents': {
        target: 'ws://localhost:8766',
        ws: true,
        rewriteWsOrigin: true,
        rewrite: () => '/',
      },
    },
  },
})

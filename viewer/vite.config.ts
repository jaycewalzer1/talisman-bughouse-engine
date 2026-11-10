import { resolve } from 'node:path'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { runsApi } from './vite.devApi'

// https://vite.dev/config/
// Three pages: the spectator viewer (index.html), the dev panel (dev.html) and the scheduler network diagram
// (network.html). runsApi serves the repo's runs/ directory to the dev panel while the dev server runs.
export default defineConfig({
  plugins: [react(), runsApi()],
  build: {
    rollupOptions: {
      input: {
        main: resolve(import.meta.dirname, 'index.html'),
        dev: resolve(import.meta.dirname, 'dev.html'),
        network: resolve(import.meta.dirname, 'network.html'),
      },
    },
  },
})

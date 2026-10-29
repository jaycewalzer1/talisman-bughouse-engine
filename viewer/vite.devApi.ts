// Dev-server-only API over the repo's runs/ directory, for the dev panel (dev.html).
//   GET /api/runs                -> [{ name, mtime, games: [file...], files: [file...] }], newest first
//   GET /api/runs/<run>/<file>   -> that .json / .jsonl file
// Only plain names inside runs/<run>/ are served; nothing outside runs/ is reachable.

import fs from 'node:fs'
import path from 'node:path'
import type { Plugin } from 'vite'

const RUNS_DIR = path.resolve(import.meta.dirname, '..', 'runs')
const SAFE = /^[\w.-]+$/
const GAME_FILE = /^game_\d+\.jsonl$/

function listRuns() {
  if (!fs.existsSync(RUNS_DIR)) return []
  return fs.readdirSync(RUNS_DIR, { withFileTypes: true })
    .filter(d => d.isDirectory() && SAFE.test(d.name))
    .map(d => {
      const dir = path.join(RUNS_DIR, d.name)
      const files = fs.readdirSync(dir).filter(f => SAFE.test(f) && /\.jsonl?$/.test(f)).sort()
      const mtime = Math.max(fs.statSync(dir).mtimeMs, ...files.map(f => fs.statSync(path.join(dir, f)).mtimeMs))
      return { name: d.name, mtime, games: files.filter(f => GAME_FILE.test(f)), files: files.filter(f => !GAME_FILE.test(f)) }
    })
    .sort((a, b) => b.mtime - a.mtime)
}

export function runsApi(): Plugin {
  return {
    name: 'runs-api',
    configureServer(server) {
      server.middlewares.use('/api/runs', (req, res) => {
        const parts = decodeURIComponent((req.url ?? '/').split('?')[0]).split('/').filter(Boolean)
        if (parts.length === 0) {
          res.setHeader('Content-Type', 'application/json')
          res.end(JSON.stringify(listRuns()))
          return
        }
        const file = parts.length === 2 && parts.every(p => SAFE.test(p) && p !== '.' && p !== '..')
          ? path.join(RUNS_DIR, parts[0], parts[1]) : null
        if (!file || !/\.jsonl?$/.test(file) || !fs.existsSync(file)) {
          res.statusCode = 404
          res.end('not found')
          return
        }
        res.setHeader('Content-Type', file.endsWith('.jsonl') ? 'application/x-ndjson' : 'application/json')
        fs.createReadStream(file).pipe(res)
      })
    },
  }
}

// Backend URL resolution order:
// 1. VITE_API_URL baked in at build time (docker build --build-arg / .env)
// 2. window.localStorage override (set via the UI, persists per-browser)
// 3. localhost:8000 fallback for local dev
// Backend URL resolution:
// - Production (Docker/Render): same origin, so relative paths like /predict.
//   The UI override is ignored so a stale saved URL can't break the deployed app.
// - Local dev (npm run dev): localStorage override, then VITE_API_URL,
//   then http://localhost:8000.
const BUILD_TIME_URL = import.meta.env.VITE_API_URL

export function getApiBase() {
  if (!import.meta.env.DEV) return BUILD_TIME_URL || ''
  return (
    localStorage.getItem('churnlens_api_url') ||
    BUILD_TIME_URL ||
    'http://localhost:8000'
  )
}

export function setApiBase(url) {
  localStorage.setItem('churnlens_api_url', url)
}

export async function predictChurn(payload) {
  const base = getApiBase().replace(/\/$/, '')
  const res = await fetch(`${base}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`Backend returned ${res.status}: ${text.slice(0, 200)}`)
  }
  return res.json()
}

export async function checkHealth() {
  const base = getApiBase().replace(/\/$/, '')
  const res = await fetch(`${base}/health`)
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`)
  return res.json()
}

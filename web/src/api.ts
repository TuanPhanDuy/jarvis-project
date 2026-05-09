import type { ScheduleItem, AuditEntry } from './types'

const BASE = ''  // proxied via vite dev server

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...init?.headers },
    ...init,
  })
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json()
}

export const api = {
  health: () => json<{ status: string; sessions_active: number }>('/api/health'),

  // Schedules
  listSchedules: () => json<ScheduleItem[]>('/api/schedules'),
  createSchedule: (body: {
    job_type: string
    params: Record<string, string>
    cron: string
    session_id?: string
  }) => json('/api/schedules', { method: 'POST', body: JSON.stringify(body) }),
  deleteSchedule: (jobId: string) =>
    fetch(`${BASE}/api/schedules/${jobId}`, { method: 'DELETE' }),

  // Audit
  listAudit: (sessionId?: string, limit = 100) => {
    const q = new URLSearchParams({ limit: String(limit) })
    if (sessionId) q.set('session_id', sessionId)
    return json<AuditEntry[]>(`/api/audit?${q}`)
  },

  // Feedback
  submitFeedback: (body: {
    session_id: string
    rating: number
    response_snippet?: string
    comment?: string
  }) => json('/api/feedback', { method: 'POST', body: JSON.stringify(body) }),

  getFeedbackStats: (sessionId?: string) => {
    const q = new URLSearchParams()
    if (sessionId) q.set('session_id', sessionId)
    return json<{ total: number; avg_rating: number; recent: { rating: number; comment: string }[] }>(
      `/api/feedback/stats?${q}`
    )
  },

  // Peer coordination
  listPeers: () => json<{ device_id: string; host: string; port: number; last_seen: number }[]>('/api/peer/list'),

  // Self-improvement report
  getImprovementReport: () => json<{ content: string | null }>('/api/improvement-report'),

  // Research reports
  listReports: () => json<{ name: string; size_bytes: number; modified: number }[]>('/api/reports'),
  getReport: (filename: string) => json<{ name: string; content: string }>(`/api/reports/${encodeURIComponent(filename)}`),
}

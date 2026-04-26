import { useEffect, useState } from 'react'
import { Plus, Trash2, Clock, RefreshCw, CalendarClock } from 'lucide-react'
import { api } from '../api'
import type { ScheduleItem } from '../types'

export function SchedulePanel({ sessionId }: { sessionId: string }) {
  const [schedules, setSchedules] = useState<ScheduleItem[]>([])
  const [loading, setLoading] = useState(false)
  const [showForm, setShowForm] = useState(false)
  const [form, setForm] = useState({
    job_type: 'research',
    topic: '',
    cron: '0 9 * * *',
  })

  const load = async () => {
    setLoading(true)
    try {
      setSchedules(await api.listSchedules())
    } catch {
      /* ignore */
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  const handleCreate = async () => {
    if (!form.topic.trim()) return
    try {
      await api.createSchedule({
        job_type: form.job_type,
        params: form.job_type === 'research' ? { topic: form.topic } : { query: form.topic },
        cron: form.cron,
        session_id: sessionId,
      })
      setShowForm(false)
      setForm({ job_type: 'research', topic: '', cron: '0 9 * * *' })
      load()
    } catch (e) {
      alert('Failed to create schedule')
    }
  }

  const handleDelete = async (jobId: string) => {
    await api.deleteSchedule(jobId)
    setSchedules(s => s.filter(x => x.job_id !== jobId))
  }

  return (
    <div className="flex flex-col gap-3 p-3">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Schedules</span>
        <div className="flex gap-1">
          <button onClick={load} className="p-1 text-gray-500 hover:text-gray-300 transition-colors">
            <RefreshCw size={13} className={loading ? 'animate-spin' : ''} />
          </button>
          <button
            onClick={() => setShowForm(f => !f)}
            className="p-1 text-gray-500 hover:text-cyan-400 transition-colors"
          >
            <Plus size={13} />
          </button>
        </div>
      </div>

      {showForm && (
        <div className="bg-jarvis-bg border border-jarvis-border rounded-lg p-3 space-y-2 text-sm">
          <select
            value={form.job_type}
            onChange={e => setForm(f => ({ ...f, job_type: e.target.value }))}
            className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1.5 text-gray-200 text-xs focus:outline-none focus:border-cyan-500"
          >
            <option value="research">Research</option>
            <option value="monitor">Monitor</option>
          </select>
          <input
            value={form.topic}
            onChange={e => setForm(f => ({ ...f, topic: e.target.value }))}
            placeholder={form.job_type === 'research' ? 'Topic…' : 'Query to monitor…'}
            className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1.5 text-gray-200 text-xs placeholder-gray-600 focus:outline-none focus:border-cyan-500"
          />
          <input
            value={form.cron}
            onChange={e => setForm(f => ({ ...f, cron: e.target.value }))}
            placeholder="Cron (e.g. 0 9 * * *)"
            className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1.5 text-gray-200 text-xs placeholder-gray-600 font-mono focus:outline-none focus:border-cyan-500"
          />
          <div className="flex gap-2">
            <button
              onClick={() => setShowForm(false)}
              className="flex-1 py-1 rounded text-xs text-gray-500 hover:text-gray-300 border border-gray-700 hover:border-gray-600 transition-all"
            >
              Cancel
            </button>
            <button
              onClick={handleCreate}
              className="flex-1 py-1 rounded text-xs bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 hover:bg-cyan-500/30 transition-all"
            >
              Create
            </button>
          </div>
        </div>
      )}

      {schedules.length === 0 && !loading && (
        <p className="text-xs text-gray-600 text-center py-4">No schedules yet</p>
      )}

      <div className="space-y-2">
        {schedules.map(s => (
          <div key={s.job_id} className="bg-jarvis-bg border border-jarvis-border rounded-lg p-2.5 group">
            <div className="flex items-start justify-between">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-1.5 mb-1">
                  <CalendarClock size={11} className="text-cyan-400 shrink-0" />
                  <span className="text-xs font-medium text-gray-300 truncate">{s.subject}</span>
                </div>
                <div className="flex items-center gap-2 text-xs text-gray-600">
                  <span className="px-1.5 py-0.5 rounded bg-gray-800 font-mono">{s.cron}</span>
                  {s.next_run && (
                    <span className="flex items-center gap-0.5">
                      <Clock size={9} />
                      {new Date(s.next_run).toLocaleString()}
                    </span>
                  )}
                </div>
              </div>
              <button
                onClick={() => handleDelete(s.job_id)}
                className="opacity-0 group-hover:opacity-100 p-1 text-gray-600 hover:text-red-400 transition-all"
              >
                <Trash2 size={11} />
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

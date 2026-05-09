import { useState, useEffect, useCallback } from 'react'
import { RefreshCw, Activity, Wrench, Radio } from 'lucide-react'
const API_BASE = '/api'

interface MetricsSummary {
  requests_total: { http: number; websocket: number }
  tool_calls: Record<string, number>
  active_ws_connections: number
  active_sessions: number
}

export function MetricsPanel() {
  const [data, setData] = useState<MetricsSummary | null>(null)
  const [loading, setLoading] = useState(false)

  const fetch_ = useCallback(async () => {
    setLoading(true)
    try {
      const res = await fetch(`${API_BASE}/metrics/summary`)
      if (res.ok) setData(await res.json())
    } catch {}
    finally { setLoading(false) }
  }, [])

  useEffect(() => {
    fetch_()
    const id = setInterval(fetch_, 15_000)
    return () => clearInterval(id)
  }, [fetch_])

  if (!data) return (
    <div className="flex items-center justify-center h-full text-gray-600 text-xs">
      {loading ? 'Loading metrics…' : 'No data yet.'}
    </div>
  )

  const toolEntries = Object.entries(data.tool_calls)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 10)

  const maxCalls = toolEntries[0]?.[1] || 1

  const totalRequests = (data.requests_total.http || 0) + (data.requests_total.websocket || 0)

  return (
    <div className="flex flex-col gap-4 p-3 text-xs text-gray-300">
      {/* Refresh */}
      <div className="flex items-center justify-between">
        <span className="text-gray-500 font-medium uppercase tracking-wider">Live Metrics</span>
        <button onClick={fetch_} className="text-gray-500 hover:text-cyan-400 transition-colors" title="Refresh">
          <RefreshCw size={11} className={loading ? 'animate-spin' : ''} />
        </button>
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-2 gap-2">
        <StatCard icon={<Activity size={12} />} label="Total Requests" value={totalRequests} />
        <StatCard icon={<Radio size={12} />} label="WS Connections" value={data.active_ws_connections} />
        <StatCard icon={<Activity size={12} />} label="HTTP" value={data.requests_total.http || 0} />
        <StatCard icon={<Activity size={12} />} label="Active Sessions" value={data.active_sessions} />
      </div>

      {/* Tool usage bar chart */}
      {toolEntries.length > 0 && (
        <div>
          <div className="flex items-center gap-1 mb-2 text-gray-500 font-medium uppercase tracking-wider">
            <Wrench size={10} />
            <span>Top Tools</span>
          </div>
          <div className="space-y-1.5">
            {toolEntries.map(([tool, count]) => (
              <div key={tool}>
                <div className="flex justify-between mb-0.5">
                  <span className="text-gray-400 truncate max-w-[160px]" title={tool}>{tool}</span>
                  <span className="text-cyan-400 font-mono">{count}</span>
                </div>
                <div className="h-1 rounded-full bg-jarvis-border overflow-hidden">
                  <div
                    className="h-full rounded-full bg-cyan-500/60 transition-all"
                    style={{ width: `${(count / maxCalls) * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {toolEntries.length === 0 && (
        <p className="text-gray-600 text-center mt-4">No tool calls recorded yet.</p>
      )}
    </div>
  )
}

function StatCard({ icon, label, value }: { icon: React.ReactNode; label: string; value: number }) {
  return (
    <div className="rounded border border-jarvis-border bg-jarvis-bg/60 p-2.5 flex flex-col gap-1">
      <div className="flex items-center gap-1 text-gray-500">
        {icon}
        <span className="text-[10px] uppercase tracking-wider">{label}</span>
      </div>
      <span className="text-cyan-400 font-mono text-lg font-bold">{value}</span>
    </div>
  )
}

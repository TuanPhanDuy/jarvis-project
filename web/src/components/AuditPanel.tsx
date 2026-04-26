import { useEffect, useState } from 'react'
import { RefreshCw, CheckCircle, XCircle, Clock, Wrench } from 'lucide-react'
import { api } from '../api'
import type { AuditEntry } from '../types'

const RISK_COLORS: Record<string, string> = {
  SAFE: 'text-green-500',
  LOW: 'text-blue-400',
  MEDIUM: 'text-yellow-400',
  HIGH: 'text-orange-400',
  CRITICAL: 'text-red-500',
}

const RISK_BG: Record<string, string> = {
  SAFE: 'border-green-900/40',
  LOW: 'border-jarvis-border',
  MEDIUM: 'border-yellow-900/40',
  HIGH: 'border-orange-900/40',
  CRITICAL: 'border-red-900/60',
}

export function AuditPanel({ sessionId }: { sessionId: string }) {
  const [entries, setEntries] = useState<AuditEntry[]>([])
  const [loading, setLoading] = useState(false)

  const load = async () => {
    setLoading(true)
    try {
      setEntries(await api.listAudit(sessionId, 50))
    } catch {
      /* ignore */
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [sessionId])

  return (
    <div className="flex flex-col gap-3 p-3">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Audit Log</span>
        <button onClick={load} className="p-1 text-gray-500 hover:text-gray-300 transition-colors">
          <RefreshCw size={13} className={loading ? 'animate-spin' : ''} />
        </button>
      </div>

      {entries.length === 0 && !loading && (
        <p className="text-xs text-gray-600 text-center py-4">No tool calls recorded yet</p>
      )}

      <div className="space-y-1.5">
        {entries.map((e, i) => (
          <div key={i} className={`bg-jarvis-bg border rounded-lg px-3 py-2 ${RISK_BG[e.risk_level?.toUpperCase()] || 'border-jarvis-border'}`}>
            <div className="flex items-center justify-between mb-0.5">
              <div className="flex items-center gap-1.5">
                <Wrench size={11} className="text-gray-500" />
                <code className="text-xs text-cyan-300 font-mono">{e.tool_name}</code>
              </div>
              <div className="flex items-center gap-1.5">
                <span className={`text-xs font-medium ${RISK_COLORS[e.risk_level] || 'text-gray-400'}`}>
                  {e.risk_level}
                </span>
                {e.approved === true && <CheckCircle size={11} className="text-green-400" />}
                {e.approved === false && <XCircle size={11} className="text-red-400" />}
                {e.approved === null && <Clock size={11} className="text-gray-500" />}
              </div>
            </div>
            <div className="text-xs text-gray-600">
              {new Date(e.timestamp).toLocaleTimeString()}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

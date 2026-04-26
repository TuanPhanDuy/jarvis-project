import { useEffect, useState } from 'react'
import { ShieldAlert, ShieldCheck, ShieldX, X, Clock } from 'lucide-react'
import type { ApprovalRequest } from '../types'

const RISK_COLORS = {
  LOW: 'text-blue-400 border-blue-500/40 bg-blue-500/10',
  MEDIUM: 'text-yellow-400 border-yellow-500/40 bg-yellow-500/10',
  HIGH: 'text-orange-400 border-orange-500/40 bg-orange-500/10',
  CRITICAL: 'text-red-400 border-red-500/40 bg-red-500/10',
}

interface ApprovalModalProps {
  request: ApprovalRequest
  onApprove: (requestId: string) => void
  onDeny: (requestId: string) => void
}

export function ApprovalModal({ request, onApprove, onDeny }: ApprovalModalProps) {
  const [timeLeft, setTimeLeft] = useState(request.expires_in)

  useEffect(() => {
    if (timeLeft <= 0) {
      onDeny(request.request_id)
      return
    }
    const t = setTimeout(() => setTimeLeft(t => t - 1), 1000)
    return () => clearTimeout(t)
  }, [timeLeft, request.request_id, onDeny])

  const riskStyle = RISK_COLORS[request.risk_level]
  const pct = (timeLeft / request.expires_in) * 100

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm animate-fadeIn">
      <div className="bg-jarvis-card border border-jarvis-border rounded-xl p-6 max-w-md w-full mx-4 shadow-2xl glow-cyan">
        {/* Header */}
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center gap-3">
            <ShieldAlert size={22} className="text-yellow-400 shrink-0" />
            <div>
              <h2 className="text-white font-semibold">Action Approval Required</h2>
              <p className="text-gray-400 text-sm mt-0.5">JARVIS wants to execute a tool</p>
            </div>
          </div>
          <button
            onClick={() => onDeny(request.request_id)}
            className="text-gray-500 hover:text-gray-300 transition-colors"
          >
            <X size={16} />
          </button>
        </div>

        {/* Tool info */}
        <div className="bg-jarvis-bg rounded-lg p-4 mb-4 border border-jarvis-border space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-gray-400 text-xs">Tool</span>
            <code className="text-cyan-300 text-sm font-mono bg-gray-800 px-2 py-0.5 rounded">
              {request.tool_name}
            </code>
          </div>
          <div className="flex items-start justify-between gap-3">
            <span className="text-gray-400 text-xs shrink-0 mt-0.5">Description</span>
            <p className="text-gray-200 text-sm text-right">{request.description}</p>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-gray-400 text-xs">Risk Level</span>
            <span className={`text-xs font-medium px-2 py-0.5 rounded border ${riskStyle}`}>
              {request.risk_level}
            </span>
          </div>
        </div>

        {/* Timer */}
        <div className="mb-5">
          <div className="flex items-center justify-between text-xs text-gray-500 mb-1.5">
            <span className="flex items-center gap-1"><Clock size={11} /> Expires in</span>
            <span className={timeLeft <= 10 ? 'text-red-400 font-medium' : 'text-gray-400'}>
              {timeLeft}s
            </span>
          </div>
          <div className="h-1 bg-gray-800 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-1000 ${
                pct > 50 ? 'bg-cyan-500' : pct > 20 ? 'bg-yellow-500' : 'bg-red-500'
              }`}
              style={{ width: `${pct}%` }}
            />
          </div>
        </div>

        {/* Actions */}
        <div className="flex gap-3">
          <button
            onClick={() => onDeny(request.request_id)}
            className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg border border-red-500/40 text-red-400 hover:bg-red-500/10 transition-all text-sm font-medium"
          >
            <ShieldX size={15} />
            Deny
          </button>
          <button
            onClick={() => onApprove(request.request_id)}
            className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-cyan-500/20 border border-cyan-500/50 text-cyan-300 hover:bg-cyan-500/30 transition-all text-sm font-medium glow-cyan"
          >
            <ShieldCheck size={15} />
            Approve
          </button>
        </div>
      </div>
    </div>
  )
}

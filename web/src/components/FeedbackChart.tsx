import { useEffect, useState } from 'react'
import { RefreshCw, ThumbsUp, ThumbsDown, MessageSquare } from 'lucide-react'
import { api } from '../api'

interface FeedbackStats {
  total: number
  avg_rating: number
  recent: { rating: number; comment: string }[]
}

export function FeedbackChart({ sessionId }: { sessionId?: string }) {
  const [stats, setStats] = useState<FeedbackStats | null>(null)
  const [loading, setLoading] = useState(false)

  const load = async () => {
    setLoading(true)
    try {
      const data = await api.getFeedbackStats(sessionId)
      setStats(data)
    } catch {
      /* ignore */
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [sessionId])

  const positiveCount = stats?.recent.filter(r => r.rating > 0).length ?? 0
  const negativeCount = stats?.recent.filter(r => r.rating < 0).length ?? 0
  const total = stats?.total ?? 0
  // avg_rating is in [-1, 1] range; convert to 0-100 satisfaction score
  const satisfaction = stats ? Math.round(((stats.avg_rating + 1) / 2) * 100) : null

  const positivePct = positiveCount + negativeCount > 0
    ? Math.round((positiveCount / (positiveCount + negativeCount)) * 100)
    : 0

  return (
    <div className="flex flex-col gap-3 p-3">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider flex items-center gap-1.5">
          <ThumbsUp size={12} />
          Feedback
        </span>
        <button onClick={load} className="p-1 text-gray-500 hover:text-gray-300 transition-colors">
          <RefreshCw size={13} className={loading ? 'animate-spin' : ''} />
        </button>
      </div>

      {stats === null && !loading && (
        <p className="text-xs text-gray-600 text-center py-4">No feedback yet — rate responses with 👍 / 👎</p>
      )}

      {stats !== null && (
        <>
          {/* Summary row */}
          <div className="bg-jarvis-bg border border-jarvis-border rounded-lg px-3 py-2.5 flex items-center justify-between">
            <div>
              <div className="text-sm font-semibold text-gray-200">
                {satisfaction !== null ? `${satisfaction}%` : '—'}
                <span className="text-xs font-normal text-gray-500 ml-1">satisfaction</span>
              </div>
              <p className="text-xs text-gray-600 mt-0.5">{total} total ratings</p>
            </div>
            <div className="flex items-center gap-3 text-xs">
              <div className="flex items-center gap-1 text-green-400">
                <ThumbsUp size={12} />
                <span className="font-medium">{positiveCount}</span>
              </div>
              <div className="flex items-center gap-1 text-red-400">
                <ThumbsDown size={12} />
                <span className="font-medium">{negativeCount}</span>
              </div>
            </div>
          </div>

          {/* Split bar */}
          {(positiveCount + negativeCount) > 0 && (
            <div>
              <div className="flex justify-between text-xs text-gray-600 mb-1">
                <span>{positivePct}% positive</span>
                <span>{100 - positivePct}% negative</span>
              </div>
              <div className="h-2 rounded-full overflow-hidden flex gap-0.5">
                <div
                  className="h-full bg-green-500 rounded-l-full transition-all duration-500"
                  style={{ width: `${positivePct}%` }}
                />
                <div
                  className="h-full bg-red-500 rounded-r-full transition-all duration-500"
                  style={{ width: `${100 - positivePct}%` }}
                />
              </div>
            </div>
          )}

          {/* Recent comments */}
          {stats.recent.some(r => r.comment) && (
            <div className="space-y-1.5">
              <p className="text-xs text-gray-500 flex items-center gap-1">
                <MessageSquare size={10} />
                Recent comments
              </p>
              {stats.recent.filter(r => r.comment).slice(0, 3).map((r, i) => (
                <div key={i} className="bg-jarvis-bg border border-jarvis-border rounded-md px-2.5 py-1.5">
                  <div className="flex items-center gap-1 mb-0.5">
                    {r.rating > 0
                      ? <ThumbsUp size={9} className="text-green-400" />
                      : <ThumbsDown size={9} className="text-red-400" />
                    }
                  </div>
                  <p className="text-xs text-gray-400">{r.comment}</p>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  )
}

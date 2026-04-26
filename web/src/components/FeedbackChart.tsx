import { useEffect, useState } from 'react'
import { RefreshCw, Star, TrendingUp, TrendingDown } from 'lucide-react'
import { api } from '../api'

interface FeedbackStats {
  total: number
  avg_rating: number
  recent: { rating: number; comment: string }[]
}

const RATING_COLORS: Record<number, string> = {
  1: 'bg-red-500',
  2: 'bg-orange-500',
  3: 'bg-yellow-500',
  4: 'bg-blue-500',
  5: 'bg-green-500',
}

function RatingBar({ rating, count, max }: { rating: number; count: number; max: number }) {
  const pct = max > 0 ? (count / max) * 100 : 0
  return (
    <div className="flex items-center gap-2">
      <div className="flex items-center gap-0.5 w-12 justify-end">
        <Star size={9} className="text-yellow-400" />
        <span className="text-xs text-gray-500">{rating}</span>
      </div>
      <div className="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full ${RATING_COLORS[rating] || 'bg-gray-500'} transition-all duration-500`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-xs text-gray-600 w-6 text-right">{count}</span>
    </div>
  )
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

  const recentRatings = stats?.recent ?? []
  // Count per rating level from recent entries
  const counts: Record<number, number> = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 }
  recentRatings.forEach(r => {
    const normalized = Math.max(1, Math.min(5, r.rating))
    counts[normalized] = (counts[normalized] || 0) + 1
  })
  const maxCount = Math.max(...Object.values(counts), 1)

  const avg = stats?.avg_rating ?? 0
  const TrendIcon = avg >= 4 ? TrendingUp : avg >= 3 ? TrendingUp : TrendingDown
  const trendColor = avg >= 4 ? 'text-green-400' : avg >= 3 ? 'text-yellow-400' : 'text-red-400'

  return (
    <div className="flex flex-col gap-3 p-3">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider flex items-center gap-1.5">
          <Star size={12} />
          Feedback
        </span>
        <button onClick={load} className="p-1 text-gray-500 hover:text-gray-300 transition-colors">
          <RefreshCw size={13} className={loading ? 'animate-spin' : ''} />
        </button>
      </div>

      {stats === null && !loading && (
        <p className="text-xs text-gray-600 text-center py-4">No feedback data yet</p>
      )}

      {stats !== null && (
        <>
          {/* Summary */}
          <div className="bg-jarvis-bg border border-jarvis-border rounded-lg px-3 py-2.5 flex items-center justify-between">
            <div>
              <div className="flex items-center gap-1.5">
                <TrendIcon size={14} className={trendColor} />
                <span className={`text-sm font-semibold ${trendColor}`}>
                  {avg.toFixed(1)}/5
                </span>
              </div>
              <p className="text-xs text-gray-600 mt-0.5">{stats.total} total ratings</p>
            </div>
            <div className="flex items-center gap-0.5">
              {[1, 2, 3, 4, 5].map(n => (
                <Star
                  key={n}
                  size={12}
                  className={n <= Math.round(avg) ? 'text-yellow-400' : 'text-gray-700'}
                  fill={n <= Math.round(avg) ? 'currentColor' : 'none'}
                />
              ))}
            </div>
          </div>

          {/* Distribution */}
          <div className="space-y-1">
            {[5, 4, 3, 2, 1].map(r => (
              <RatingBar key={r} rating={r} count={counts[r] ?? 0} max={maxCount} />
            ))}
          </div>

          {/* Recent comments */}
          {recentRatings.some(r => r.comment) && (
            <div className="space-y-1.5">
              <p className="text-xs text-gray-500">Recent comments:</p>
              {recentRatings.filter(r => r.comment).slice(0, 3).map((r, i) => (
                <div key={i} className="bg-jarvis-bg border border-jarvis-border rounded-md px-2.5 py-1.5">
                  <div className="flex items-center gap-1 mb-0.5">
                    <Star size={9} className="text-yellow-400" fill="currentColor" />
                    <span className="text-xs text-gray-500">{r.rating}</span>
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

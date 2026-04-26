import { useEffect, useState } from 'react'
import { RefreshCw, Lightbulb, AlertCircle } from 'lucide-react'
import { api } from '../api'

export function ImprovementsPanel() {
  const [content, setContent] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await api.getImprovementReport()
      setContent(data.content ?? null)
    } catch {
      setError('No improvement report found. Run: jarvis-eval --analyze-feedback')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  // Very simple markdown → plain text for display
  const renderLines = (text: string) =>
    text.split('\n').map((line, i) => {
      if (line.startsWith('## ') || line.startsWith('# ')) {
        return (
          <p key={i} className="text-xs font-semibold text-cyan-300 mt-3 mb-1">
            {line.replace(/^#{1,3} /, '')}
          </p>
        )
      }
      if (line.startsWith('- ') || line.startsWith('• ')) {
        return (
          <p key={i} className="text-xs text-gray-400 pl-2">
            {line}
          </p>
        )
      }
      if (line.startsWith('**') && line.endsWith('**')) {
        return (
          <p key={i} className="text-xs font-medium text-gray-300">
            {line.replace(/\*\*/g, '')}
          </p>
        )
      }
      if (line.trim() === '') {
        return <div key={i} className="h-1" />
      }
      return (
        <p key={i} className="text-xs text-gray-400">
          {line}
        </p>
      )
    })

  return (
    <div className="flex flex-col gap-3 p-3">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider flex items-center gap-1.5">
          <Lightbulb size={12} />
          Self-Improvement
        </span>
        <button onClick={load} className="p-1 text-gray-500 hover:text-gray-300 transition-colors">
          <RefreshCw size={13} className={loading ? 'animate-spin' : ''} />
        </button>
      </div>

      {loading && (
        <p className="text-xs text-gray-600 text-center py-4">Loading…</p>
      )}

      {error && !loading && (
        <div className="flex items-start gap-1.5 text-xs text-yellow-500/80 py-2">
          <AlertCircle size={12} className="shrink-0 mt-0.5" />
          <span>{error}</span>
        </div>
      )}

      {content && !loading && (
        <div className="overflow-y-auto max-h-96 space-y-0.5">
          {renderLines(content)}
        </div>
      )}
    </div>
  )
}

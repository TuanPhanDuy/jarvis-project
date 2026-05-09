import { useEffect, useState } from 'react'
import { RefreshCw, Lightbulb, AlertCircle } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
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
        <div className="overflow-y-auto max-h-96 prose-jarvis text-xs">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {content}
          </ReactMarkdown>
        </div>
      )}
    </div>
  )
}

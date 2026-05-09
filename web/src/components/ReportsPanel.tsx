import { useEffect, useState } from 'react'
import { RefreshCw, FileText, ChevronLeft, ArrowDownToLine } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { api } from '../api'

interface ReportMeta {
  name: string
  size_bytes: number
  modified: number
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  return `${(bytes / 1024).toFixed(1)} KB`
}

function formatDate(ts: number): string {
  return new Date(ts * 1000).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })
}

export function ReportsPanel() {
  const [reports, setReports] = useState<ReportMeta[]>([])
  const [loading, setLoading] = useState(false)
  const [selected, setSelected] = useState<string | null>(null)
  const [content, setContent] = useState<string | null>(null)
  const [contentLoading, setContentLoading] = useState(false)

  const load = async () => {
    setLoading(true)
    try {
      setReports(await api.listReports())
    } catch {
      /* ignore */
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  const openReport = async (name: string) => {
    setSelected(name)
    setContent(null)
    setContentLoading(true)
    try {
      const data = await api.getReport(name)
      setContent(data.content)
    } catch {
      setContent('*Failed to load report.*')
    } finally {
      setContentLoading(false)
    }
  }

  const back = () => {
    setSelected(null)
    setContent(null)
  }

  // Report detail view
  if (selected) {
    return (
      <div className="flex flex-col h-full">
        <div className="flex items-center gap-2 px-3 py-2.5 border-b border-jarvis-border shrink-0">
          <button
            onClick={back}
            className="p-1 text-gray-500 hover:text-cyan-400 transition-colors"
            title="Back to list"
          >
            <ChevronLeft size={14} />
          </button>
          <span className="text-xs font-medium text-gray-300 truncate flex-1">{selected}</span>
        </div>
        <div className="flex-1 overflow-y-auto px-3 py-3">
          {contentLoading && (
            <p className="text-xs text-gray-600 text-center py-4">Loading…</p>
          )}
          {content && !contentLoading && (
            <div className="prose-jarvis text-xs">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {content}
              </ReactMarkdown>
            </div>
          )}
        </div>
      </div>
    )
  }

  // Report list view
  return (
    <div className="flex flex-col gap-3 p-3">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider flex items-center gap-1.5">
          <FileText size={12} />
          Reports
        </span>
        <button onClick={load} className="p-1 text-gray-500 hover:text-gray-300 transition-colors">
          <RefreshCw size={13} className={loading ? 'animate-spin' : ''} />
        </button>
      </div>

      {reports.length === 0 && !loading && (
        <div className="text-center py-6 space-y-1">
          <p className="text-xs text-gray-600">No reports yet</p>
          <p className="text-xs text-gray-700">Ask JARVIS to research a topic to generate one</p>
        </div>
      )}

      {loading && (
        <p className="text-xs text-gray-600 text-center py-4">Loading…</p>
      )}

      <div className="space-y-1.5">
        {reports.map(r => (
          <button
            key={r.name}
            onClick={() => openReport(r.name)}
            className="w-full text-left bg-jarvis-bg border border-jarvis-border hover:border-cyan-500/40 hover:bg-cyan-500/5 rounded-lg px-3 py-2.5 transition-all group"
          >
            <div className="flex items-start justify-between gap-2">
              <div className="flex items-start gap-2 min-w-0">
                <ArrowDownToLine size={11} className="text-cyan-500 mt-0.5 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity" />
                <span className="text-xs text-gray-300 group-hover:text-cyan-300 transition-colors truncate leading-snug">
                  {r.name.replace(/\.md$/, '')}
                </span>
              </div>
              <span className="text-xs text-gray-600 shrink-0">{formatSize(r.size_bytes)}</span>
            </div>
            <p className="text-xs text-gray-600 mt-0.5 pl-[19px]">{formatDate(r.modified)}</p>
          </button>
        ))}
      </div>
    </div>
  )
}

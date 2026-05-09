import { useState, useEffect, useCallback } from 'react'
import { Search, RefreshCw, MessageSquare, Share2 } from 'lucide-react'
const API_BASE = '/api'

interface Episode {
  session_id: string
  user_id: string
  role: string
  content: string
  timestamp: number
}

interface GraphEntry {
  subject: string
  predicate: string
  object: string
  updated_at?: number
}

type Tab = 'episodes' | 'graph'

export function MemoryPanel() {
  const [tab, setTab] = useState<Tab>('episodes')
  const [search, setSearch] = useState('')
  const [episodes, setEpisodes] = useState<Episode[]>([])
  const [graph, setGraph] = useState<GraphEntry[]>([])
  const [loading, setLoading] = useState(false)

  const fetchEpisodes = useCallback(async () => {
    setLoading(true)
    try {
      const params = new URLSearchParams({ limit: '30' })
      const res = await fetch(`${API_BASE}/memory/episodes?${params}`)
      if (res.ok) setEpisodes(await res.json())
    } catch {}
    finally { setLoading(false) }
  }, [])

  const fetchGraph = useCallback(async () => {
    setLoading(true)
    try {
      const params = search ? new URLSearchParams({ entity: search, limit: '50' }) : new URLSearchParams({ limit: '50' })
      const res = await fetch(`${API_BASE}/memory/graph?${params}`)
      if (res.ok) setGraph(await res.json())
    } catch {}
    finally { setLoading(false) }
  }, [search])

  useEffect(() => {
    if (tab === 'episodes') fetchEpisodes()
    else fetchGraph()
  }, [tab, fetchEpisodes, fetchGraph])

  const filteredEpisodes = search
    ? episodes.filter(e => e.content.toLowerCase().includes(search.toLowerCase()))
    : episodes

  const formatTime = (ts: number) =>
    new Date(ts * 1000).toLocaleString(undefined, { dateStyle: 'short', timeStyle: 'short' })

  return (
    <div className="flex flex-col h-full text-xs text-gray-300">
      {/* Toolbar */}
      <div className="flex items-center gap-2 p-3 border-b border-jarvis-border">
        <Search size={11} className="text-gray-500 shrink-0" />
        <input
          value={search}
          onChange={e => setSearch(e.target.value)}
          placeholder="Search memory…"
          className="flex-1 bg-transparent outline-none text-gray-300 placeholder-gray-600"
        />
        <button
          onClick={() => tab === 'episodes' ? fetchEpisodes() : fetchGraph()}
          className="text-gray-500 hover:text-cyan-400 transition-colors"
          title="Refresh"
        >
          <RefreshCw size={11} className={loading ? 'animate-spin' : ''} />
        </button>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-jarvis-border shrink-0">
        {(['episodes', 'graph'] as Tab[]).map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`flex items-center gap-1 px-3 py-2 text-xs font-medium transition-all ${
              tab === t ? 'text-cyan-400 border-b border-cyan-400' : 'text-gray-500 hover:text-gray-300'
            }`}
          >
            {t === 'episodes' ? <MessageSquare size={11} /> : <Share2 size={11} />}
            {t === 'episodes' ? 'Episodes' : 'Knowledge'}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-2 space-y-2">
        {tab === 'episodes' && (
          filteredEpisodes.length === 0
            ? <p className="text-gray-600 text-center mt-8">No episodes found.</p>
            : filteredEpisodes.map((ep, i) => (
              <div key={i} className="rounded border border-jarvis-border bg-jarvis-bg/60 p-2">
                <div className="flex justify-between items-center mb-1">
                  <span className={`font-semibold ${ep.role === 'user' ? 'text-cyan-400' : 'text-green-400'}`}>
                    {ep.role}
                  </span>
                  <span className="text-gray-600">{formatTime(ep.timestamp)}</span>
                </div>
                <p className="text-gray-400 line-clamp-3">{ep.content}</p>
              </div>
            ))
        )}

        {tab === 'graph' && (
          graph.length === 0
            ? <p className="text-gray-600 text-center mt-8">No knowledge graph entries yet.</p>
            : graph.map((entry, i) => (
              <div key={i} className="rounded border border-jarvis-border bg-jarvis-bg/60 p-2">
                <span className="text-cyan-300 font-medium">{entry.subject}</span>
                <span className="text-gray-500 mx-1.5">→</span>
                <span className="text-purple-400">{entry.predicate}</span>
                <span className="text-gray-500 mx-1.5">→</span>
                <span className="text-yellow-300">{entry.object}</span>
              </div>
            ))
        )}
      </div>
    </div>
  )
}

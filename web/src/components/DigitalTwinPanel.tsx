import { useEffect, useState } from 'react'
import { RefreshCw, Cpu, HardDrive, Network, GitBranch, AlertCircle } from 'lucide-react'
import { api } from '../api'

interface PeerNode {
  device_id: string
  host: string
  port: number
  last_seen: number
}

export function DigitalTwinPanel() {
  const [peers, setPeers] = useState<PeerNode[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await api.listPeers()
      setPeers(data)
    } catch (e) {
      setError('Could not load peer data (peer coordination may be disabled)')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  const timeSince = (ts: number) => {
    const sec = Math.floor(Date.now() / 1000 - ts)
    if (sec < 60) return `${sec}s ago`
    if (sec < 3600) return `${Math.floor(sec / 60)}m ago`
    return `${Math.floor(sec / 3600)}h ago`
  }

  return (
    <div className="flex flex-col gap-3 p-3">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider flex items-center gap-1.5">
          <Network size={12} />
          Digital Twin
        </span>
        <button onClick={load} className="p-1 text-gray-500 hover:text-gray-300 transition-colors">
          <RefreshCw size={13} className={loading ? 'animate-spin' : ''} />
        </button>
      </div>

      {/* Local system node */}
      <div className="bg-cyan-500/10 border border-cyan-500/30 rounded-lg px-3 py-2.5">
        <div className="flex items-center gap-2 mb-1.5">
          <div className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
          <span className="text-xs font-medium text-cyan-300">This Node (local)</span>
        </div>
        <div className="grid grid-cols-3 gap-1.5">
          <div className="flex items-center gap-1 text-xs text-gray-500">
            <Cpu size={10} />
            <span>CPU</span>
          </div>
          <div className="flex items-center gap-1 text-xs text-gray-500">
            <HardDrive size={10} />
            <span>Disk</span>
          </div>
          <div className="flex items-center gap-1 text-xs text-gray-500">
            <GitBranch size={10} />
            <span>Repos</span>
          </div>
        </div>
        <p className="text-xs text-gray-600 mt-1.5">
          Ask JARVIS: <em className="text-gray-500">"snapshot my system"</em> to populate.
        </p>
      </div>

      {/* Peer nodes */}
      <div>
        <p className="text-xs text-gray-500 mb-1.5">
          {peers.length === 0 ? 'No peers discovered' : `${peers.length} peer${peers.length !== 1 ? 's' : ''} on LAN`}
        </p>
        <div className="space-y-1.5">
          {peers.map(peer => (
            <div key={peer.device_id} className="bg-jarvis-bg border border-jarvis-border rounded-lg px-3 py-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1.5">
                  <div className="w-1.5 h-1.5 rounded-full bg-green-400" />
                  <code className="text-xs text-gray-300 font-mono">{peer.device_id}</code>
                </div>
                <span className="text-xs text-gray-600">{timeSince(peer.last_seen)}</span>
              </div>
              <p className="text-xs text-gray-600 mt-0.5">
                {peer.host}:{peer.port}
              </p>
            </div>
          ))}
        </div>
      </div>

      {error && (
        <div className="flex items-start gap-1.5 text-xs text-yellow-500/80">
          <AlertCircle size={12} className="shrink-0 mt-0.5" />
          <span>{error}</span>
        </div>
      )}

      <p className="text-xs text-gray-700">
        Enable peer discovery: <code className="text-gray-600">JARVIS_PEER_ENABLED=true</code>
      </p>
    </div>
  )
}

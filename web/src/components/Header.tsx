import { Wifi, WifiOff, RefreshCw, Cpu, Search, Users } from 'lucide-react'

export type AgentMode = 'planner' | 'researcher' | 'team'

interface HeaderProps {
  connected: boolean
  sessionId: string
  agentMode: AgentMode
  onAgentModeChange: (mode: AgentMode) => void
  onNewSession: () => void
}

const MODES: { id: AgentMode; label: string; icon: React.ReactNode; desc: string }[] = [
  { id: 'planner', label: 'Planner', icon: <Cpu size={13} />, desc: 'Orchestrates tasks with specialist sub-agents' },
  { id: 'researcher', label: 'Researcher', icon: <Search size={13} />, desc: 'Deep research mode with web search' },
  { id: 'team', label: 'Team', icon: <Users size={13} />, desc: 'Manager + Team Lead + Frontend + Backend agents' },
]

export function Header({ connected, sessionId, agentMode, onAgentModeChange, onNewSession }: HeaderProps) {
  return (
    <header className="flex items-center justify-between px-5 py-3 border-b border-jarvis-border bg-jarvis-card/80 backdrop-blur-sm shrink-0">
      {/* Logo */}
      <div className="flex items-center gap-3">
        <div className="relative">
          <div className="w-8 h-8 rounded-full bg-cyan-500/10 border border-cyan-500/50 flex items-center justify-center glow-cyan">
            <span className="text-cyan-400 font-bold text-sm">J</span>
          </div>
          <span className={`absolute -bottom-0.5 -right-0.5 w-2.5 h-2.5 rounded-full border border-jarvis-card ${connected ? 'bg-cyan-400' : 'bg-red-500'}`} />
        </div>
        <div>
          <h1 className="text-cyan-400 font-bold text-base leading-none glow-cyan-text tracking-widest">
            JARVIS
          </h1>
          <p className="text-gray-500 text-xs mt-0.5">AI Research Agent</p>
        </div>
      </div>

      {/* Center: agent mode toggle */}
      <div className="flex items-center gap-1 bg-jarvis-bg rounded-lg border border-jarvis-border p-0.5">
        {MODES.map(mode => (
          <button
            key={mode.id}
            onClick={() => onAgentModeChange(mode.id)}
            title={mode.desc}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
              agentMode === mode.id
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                : 'text-gray-500 hover:text-gray-300'
            }`}
          >
            {mode.icon}
            {mode.label}
          </button>
        ))}
      </div>

      {/* Right: session info */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1.5 text-xs text-gray-500">
          {connected
            ? <Wifi size={13} className="text-cyan-400" />
            : <WifiOff size={13} className="text-red-400" />
          }
          <span className={connected ? 'text-cyan-400' : 'text-red-400'}>
            {connected ? 'Connected' : 'Reconnecting…'}
          </span>
        </div>
        <div className="text-xs text-gray-600 font-mono truncate max-w-[120px]" title={sessionId}>
          {sessionId.slice(0, 8)}…
        </div>
        <button
          onClick={onNewSession}
          title="New Session"
          className="p-1.5 rounded-md text-gray-500 hover:text-cyan-400 hover:bg-cyan-500/10 transition-all"
        >
          <RefreshCw size={14} />
        </button>
      </div>
    </header>
  )
}

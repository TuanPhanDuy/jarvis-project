import { Users, Search, Code, CheckCircle, BarChart2, Server } from 'lucide-react'

interface TeamMember {
  role: string
  title: string
  icon: React.ReactNode
  color: string
  expertise: string[]
}

const TEAM: TeamMember[] = [
  {
    role: 'researcher',
    title: 'Researcher',
    icon: <Search size={14} />,
    color: 'text-cyan-400 border-cyan-500/30 bg-cyan-500/10',
    expertise: ['Web search', 'Synthesis', 'Summaries'],
  },
  {
    role: 'coder',
    title: 'Coder',
    icon: <Code size={14} />,
    color: 'text-green-400 border-green-500/30 bg-green-500/10',
    expertise: ['Python', 'Scripts', 'Algorithms'],
  },
  {
    role: 'qa',
    title: 'QA',
    icon: <CheckCircle size={14} />,
    color: 'text-yellow-400 border-yellow-500/30 bg-yellow-500/10',
    expertise: ['Code review', 'Testing', 'Security'],
  },
  {
    role: 'analyst',
    title: 'Analyst',
    icon: <BarChart2 size={14} />,
    color: 'text-purple-400 border-purple-500/30 bg-purple-500/10',
    expertise: ['SQL', 'Data analysis', 'Statistics'],
  },
  {
    role: 'devops',
    title: 'DevOps',
    icon: <Server size={14} />,
    color: 'text-orange-400 border-orange-500/30 bg-orange-500/10',
    expertise: ['System info', 'Git', 'Shell'],
  },
]

interface TeamRosterProps {
  activeRoles: Set<string>
}

export function TeamRoster({ activeRoles }: TeamRosterProps) {
  return (
    <div className="border-b border-jarvis-border bg-jarvis-card/40 px-4 py-2.5 shrink-0">
      <div className="flex items-center gap-3 overflow-x-auto pb-0.5">
        <div className="flex items-center gap-1.5 text-xs text-gray-500 shrink-0">
          <Users size={12} />
          <span>Team</span>
        </div>
        {TEAM.map(m => {
          const isActive = activeRoles.has(m.role)
          return (
            <div
              key={m.role}
              title={m.expertise.join(' · ')}
              className={`flex items-center gap-2 px-2.5 py-1 rounded-lg border text-xs shrink-0 transition-all ${
                isActive
                  ? m.color + ' opacity-100'
                  : 'text-gray-600 border-gray-800 bg-transparent opacity-60'
              }`}
            >
              {m.icon}
              <span className="font-medium">{m.title}</span>
              {isActive && (
                <span className="w-1.5 h-1.5 rounded-full bg-current animate-pulse" />
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

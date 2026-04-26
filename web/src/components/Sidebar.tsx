import { CalendarClock, Shield, Network, Star, Lightbulb, ChevronLeft, ChevronRight } from 'lucide-react'
import { SchedulePanel } from './SchedulePanel'
import { AuditPanel } from './AuditPanel'
import { DigitalTwinPanel } from './DigitalTwinPanel'
import { FeedbackChart } from './FeedbackChart'
import { ImprovementsPanel } from './ImprovementsPanel'

export type SidebarTab = 'schedules' | 'audit' | 'twin' | 'feedback' | 'improvements'

interface SidebarProps {
  open: boolean
  onToggle: () => void
  activeTab: SidebarTab
  onTabChange: (tab: SidebarTab) => void
  sessionId: string
}

const TABS: { id: SidebarTab; label: string; icon: React.ReactNode }[] = [
  { id: 'schedules', label: 'Schedules', icon: <CalendarClock size={14} /> },
  { id: 'audit', label: 'Audit', icon: <Shield size={14} /> },
  { id: 'twin', label: 'Twin', icon: <Network size={14} /> },
  { id: 'feedback', label: 'Ratings', icon: <Star size={14} /> },
  { id: 'improvements', label: 'Ideas', icon: <Lightbulb size={14} /> },
]

export function Sidebar({ open, onToggle, activeTab, onTabChange, sessionId }: SidebarProps) {
  return (
    <aside
      className={`flex flex-col border-r border-jarvis-border bg-jarvis-card/60 transition-all duration-300 shrink-0 ${
        open ? 'w-72' : 'w-10'
      }`}
    >
      {/* Toggle */}
      <button
        onClick={onToggle}
        className="self-end m-2 p-1.5 rounded-md text-gray-500 hover:text-cyan-400 hover:bg-cyan-500/10 transition-all"
        title={open ? 'Collapse sidebar' : 'Expand sidebar'}
      >
        {open ? <ChevronLeft size={14} /> : <ChevronRight size={14} />}
      </button>

      {open && (
        <>
          {/* Tab bar — scrollable for narrow screens */}
          <div className="flex border-b border-jarvis-border shrink-0 overflow-x-auto">
            {TABS.map(tab => (
              <button
                key={tab.id}
                onClick={() => onTabChange(tab.id)}
                className={`flex-shrink-0 flex items-center justify-center gap-1 px-2 py-2.5 text-xs font-medium transition-all ${
                  activeTab === tab.id
                    ? 'text-cyan-400 border-b border-cyan-400'
                    : 'text-gray-500 hover:text-gray-300'
                }`}
                title={tab.label}
              >
                {tab.icon}
                <span className="hidden sm:inline">{tab.label}</span>
              </button>
            ))}
          </div>

          {/* Content */}
          <div className="flex-1 overflow-y-auto">
            {activeTab === 'schedules' && <SchedulePanel sessionId={sessionId} />}
            {activeTab === 'audit' && <AuditPanel sessionId={sessionId} />}
            {activeTab === 'twin' && <DigitalTwinPanel />}
            {activeTab === 'feedback' && <FeedbackChart sessionId={sessionId} />}
            {activeTab === 'improvements' && <ImprovementsPanel />}
          </div>
        </>
      )}
    </aside>
  )
}

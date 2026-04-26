import { useEffect } from 'react'
import { X, Bell, AlertTriangle, Info } from 'lucide-react'
import type { ProactiveNotification } from '../types'

interface ProactiveToastProps {
  notifications: ProactiveNotification[]
  onDismiss: (id: string) => void
}

const SEVERITY_STYLES = {
  info: {
    border: 'border-cyan-500/30',
    bg: 'bg-cyan-500/10',
    icon: <Info size={15} className="text-cyan-400 shrink-0" />,
    label: 'text-cyan-400',
  },
  warning: {
    border: 'border-yellow-500/30',
    bg: 'bg-yellow-500/10',
    icon: <AlertTriangle size={15} className="text-yellow-400 shrink-0" />,
    label: 'text-yellow-400',
  },
  critical: {
    border: 'border-red-500/40',
    bg: 'bg-red-500/10',
    icon: <AlertTriangle size={15} className="text-red-400 shrink-0" />,
    label: 'text-red-400',
  },
}

function Toast({ n, onDismiss }: { n: ProactiveNotification; onDismiss: () => void }) {
  const s = SEVERITY_STYLES[n.severity]

  useEffect(() => {
    const t = setTimeout(onDismiss, 12000)
    return () => clearTimeout(t)
  }, [onDismiss])

  return (
    <div className={`flex items-start gap-3 px-4 py-3 rounded-xl border ${s.border} ${s.bg} shadow-xl animate-slideUp max-w-sm backdrop-blur-sm`}>
      {s.icon}
      <div className="flex-1 min-w-0">
        <div className={`text-xs font-semibold ${s.label} uppercase tracking-wide mb-0.5`}>
          <Bell size={10} className="inline mr-1" />
          {n.trigger.replace(/_/g, ' ')}
        </div>
        <p className="text-gray-200 text-xs leading-snug line-clamp-3">{n.text}</p>
      </div>
      <button
        onClick={onDismiss}
        className="text-gray-500 hover:text-gray-300 shrink-0 mt-0.5 transition-colors"
      >
        <X size={12} />
      </button>
    </div>
  )
}

export function ProactiveToast({ notifications, onDismiss }: ProactiveToastProps) {
  if (!notifications.length) return null

  return (
    <div className="fixed bottom-20 right-4 z-40 flex flex-col gap-2">
      {notifications.map(n => (
        <Toast key={n.id} n={n} onDismiss={() => onDismiss(n.id)} />
      ))}
    </div>
  )
}

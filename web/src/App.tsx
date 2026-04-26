import { useCallback, useRef, useState } from 'react'
import { Header, type AgentMode } from './components/Header'
import { Sidebar, type SidebarTab } from './components/Sidebar'
import { ChatWindow } from './components/ChatWindow'
import { ApprovalModal } from './components/ApprovalModal'
import { ProactiveToast } from './components/ProactiveToast'
import { TeamRoster } from './components/TeamRoster'
import { useJarvisWS } from './hooks/useJarvisWS'
import type {
  ChatMessage,
  ApprovalRequest,
  ProactiveNotification,
  WsServerMessage,
} from './types'

// Map tool names to team roles for the roster indicator
const TOOL_TO_ROLE: Record<string, string> = {
  delegate_to_team_member: 'manager',
}

function makeId() {
  return crypto.randomUUID()
}

function newSession() {
  return makeId()
}

export default function App() {
  const [sessionId, setSessionId] = useState(newSession)
  const [agentMode, setAgentMode] = useState<AgentMode>('planner')
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [sidebarTab, setSidebarTab] = useState<SidebarTab>('schedules')
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isThinking, setIsThinking] = useState(false)
  const [pendingApproval, setPendingApproval] = useState<ApprovalRequest | null>(null)
  const [notifications, setNotifications] = useState<ProactiveNotification[]>([])
  const [activeRoles, setActiveRoles] = useState<Set<string>>(new Set())

  // Track current streaming assistant message id
  const streamingIdRef = useRef<string | null>(null)

  const handleWsMessage = useCallback((msg: WsServerMessage) => {
    switch (msg.type) {
      case 'thinking':
        setIsThinking(true)
        break

      case 'chunk': {
        setIsThinking(false)
        setMessages(prev => {
          const streamId = streamingIdRef.current
          if (streamId) {
            return prev.map(m =>
              m.id === streamId
                ? { ...m, content: m.content + msg.text, isStreaming: true }
                : m,
            )
          }
          // Start new streaming message
          const id = makeId()
          streamingIdRef.current = id
          return [...prev, {
            id,
            role: 'assistant',
            content: msg.text,
            toolCalls: [],
            isStreaming: true,
            timestamp: new Date(),
          }]
        })
        break
      }

      case 'tool_call': {
        // Light up team roster card when delegation happens
        if (msg.tool === 'delegate_to_team_member') {
          setActiveRoles(prev => new Set([...prev, 'manager']))
        }
        setMessages(prev => {
          const streamId = streamingIdRef.current
          if (streamId) {
            return prev.map(m =>
              m.id === streamId
                ? { ...m, toolCalls: [...m.toolCalls, msg.tool] }
                : m,
            )
          }
          // No streaming message yet — start one to hold tool badges
          const id = makeId()
          streamingIdRef.current = id
          return [...prev, {
            id,
            role: 'assistant',
            content: '',
            toolCalls: [msg.tool],
            isStreaming: true,
            timestamp: new Date(),
          }]
        })
        break
      }

      case 'done': {
        setActiveRoles(new Set())
        setIsThinking(false)
        const streamId = streamingIdRef.current
        streamingIdRef.current = null
        setMessages(prev => {
          if (streamId) {
            return prev.map(m =>
              m.id === streamId
                ? { ...m, content: msg.text, isStreaming: false, usage: msg.usage }
                : m,
            )
          }
          // No streaming was started (e.g. cached)
          return [...prev, {
            id: makeId(),
            role: 'assistant',
            content: msg.text,
            toolCalls: [],
            isStreaming: false,
            usage: msg.usage,
            timestamp: new Date(),
          }]
        })
        break
      }

      case 'error': {
        setIsThinking(false)
        streamingIdRef.current = null
        setMessages(prev => [...prev, {
          id: makeId(),
          role: 'error',
          content: msg.message,
          toolCalls: [],
          isStreaming: false,
          timestamp: new Date(),
        }])
        break
      }

      case 'approval_request':
        setPendingApproval({
          request_id: msg.request_id,
          tool_name: msg.tool_name,
          description: msg.description,
          risk_level: msg.risk_level,
          expires_in: msg.expires_in,
        })
        break

      case 'proactive':
        setNotifications(prev => [...prev, {
          id: makeId(),
          trigger: msg.trigger,
          text: msg.text,
          severity: msg.severity,
          timestamp: new Date(),
        }])
        break
    }
  }, [])

  const { connected, send } = useJarvisWS(sessionId, handleWsMessage)

  const handleSend = useCallback((text: string) => {
    setMessages(prev => [...prev, {
      id: makeId(),
      role: 'user',
      content: text,
      toolCalls: [],
      isStreaming: false,
      timestamp: new Date(),
    }])
    setIsThinking(true)
    streamingIdRef.current = null
    send({
      type: 'chat',
      message: text,
      researcher_mode: agentMode === 'researcher',
      team_mode: agentMode === 'team',
    })
  }, [send, agentMode])

  const handleApprove = useCallback((requestId: string) => {
    send({ type: 'approval_response', request_id: requestId, approved: true })
    setPendingApproval(null)
  }, [send])

  const handleDeny = useCallback((requestId: string) => {
    send({ type: 'approval_response', request_id: requestId, approved: false })
    setPendingApproval(null)
  }, [send])

  const handleNewSession = () => {
    setSessionId(newSession())
    setMessages([])
    setIsThinking(false)
    setPendingApproval(null)
    setActiveRoles(new Set())
    streamingIdRef.current = null
  }

  const dismissNotification = useCallback((id: string) => {
    setNotifications(prev => prev.filter(n => n.id !== id))
  }, [])

  return (
    <div className="flex flex-col h-screen bg-jarvis-bg overflow-hidden">
      <Header
        connected={connected}
        sessionId={sessionId}
        agentMode={agentMode}
        onAgentModeChange={setAgentMode}
        onNewSession={handleNewSession}
      />

      <div className="flex flex-1 min-h-0">
        <Sidebar
          open={sidebarOpen}
          onToggle={() => setSidebarOpen(o => !o)}
          activeTab={sidebarTab}
          onTabChange={setSidebarTab}
          sessionId={sessionId}
        />

        <main className="flex flex-col flex-1 min-w-0 min-h-0">
          {agentMode === 'team' && <TeamRoster activeRoles={activeRoles} />}
          <ChatWindow
            messages={messages}
            isThinking={isThinking}
            sessionId={sessionId}
            onSend={handleSend}
            disabled={!connected || isThinking}
          />
        </main>
      </div>

      {pendingApproval && (
        <ApprovalModal
          request={pendingApproval}
          onApprove={handleApprove}
          onDeny={handleDeny}
        />
      )}

      <ProactiveToast
        notifications={notifications}
        onDismiss={dismissNotification}
      />
    </div>
  )
}

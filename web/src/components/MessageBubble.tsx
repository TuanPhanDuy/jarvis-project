import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { ThumbsUp, ThumbsDown, Wrench, ChevronDown, ChevronUp, Bot, User, AlertCircle, Users } from 'lucide-react'
import type { ChatMessage } from '../types'
import { api } from '../api'

interface MessageBubbleProps {
  message: ChatMessage
  sessionId: string
}

const TEAM_ROLE_LABELS: Record<string, { label: string; color: string }> = {
  manager: { label: 'Manager', color: 'text-purple-300 border-purple-500/30 bg-purple-500/10' },
  team_lead: { label: 'Team Lead', color: 'text-yellow-300 border-yellow-500/30 bg-yellow-500/10' },
  frontend: { label: 'Frontend', color: 'text-cyan-300 border-cyan-500/30 bg-cyan-500/10' },
  backend: { label: 'Backend', color: 'text-green-300 border-green-500/30 bg-green-500/10' },
}

function ToolBadge({ tool }: { tool: string }) {
  // Parse team delegation tool calls like "delegate_to_team_member:frontend"
  if (tool.startsWith('delegate_to_team_member:')) {
    const role = tool.split(':')[1]
    const info = TEAM_ROLE_LABELS[role]
    if (info) {
      return (
        <span className={`inline-flex items-center gap-1 px-2 py-0.5 border rounded-full text-xs font-medium ${info.color}`}>
          <Users size={10} />
          {info.label}
        </span>
      )
    }
  }
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-blue-500/10 border border-blue-500/30 rounded-full text-blue-300 text-xs font-mono">
      <Wrench size={10} />
      {tool}
    </span>
  )
}

function UsageStats({ usage }: { usage: NonNullable<ChatMessage['usage']> }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="mt-2">
      <button
        onClick={() => setOpen(o => !o)}
        className="flex items-center gap-1 text-xs text-gray-600 hover:text-gray-400 transition-colors"
      >
        {open ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
        <span>${usage.estimated_cost_usd.toFixed(5)}</span>
        <span className="text-gray-700">·</span>
        <span>{(usage.input_tokens + usage.output_tokens).toLocaleString()} tokens</span>
      </button>
      {open && (
        <div className="mt-1.5 grid grid-cols-2 gap-x-4 gap-y-0.5 text-xs text-gray-500 bg-jarvis-bg rounded-md px-2.5 py-2 border border-jarvis-border">
          <span>Input</span><span className="text-right text-gray-400">{usage.input_tokens.toLocaleString()}</span>
          <span>Output</span><span className="text-right text-gray-400">{usage.output_tokens.toLocaleString()}</span>
          <span>Cache write</span><span className="text-right text-gray-400">{usage.cache_write_tokens.toLocaleString()}</span>
          <span>Cache read</span><span className="text-right text-gray-400">{usage.cache_read_tokens.toLocaleString()}</span>
          <span className="text-cyan-600 font-medium">Cost</span>
          <span className="text-right text-cyan-500 font-medium">${usage.estimated_cost_usd.toFixed(5)}</span>
        </div>
      )}
    </div>
  )
}

export function MessageBubble({ message, sessionId }: MessageBubbleProps) {
  const [feedback, setFeedback] = useState<number | null>(null)

  const handleFeedback = async (rating: number) => {
    if (feedback !== null) return
    setFeedback(rating)
    try {
      await api.submitFeedback({
        session_id: sessionId,
        rating,
        response_snippet: message.content.slice(0, 200),
      })
    } catch {
      // silent
    }
  }

  const isUser = message.role === 'user'
  const isError = message.role === 'error'

  if (isError) {
    return (
      <div className="flex items-start gap-2 animate-slideUp">
        <div className="shrink-0 w-7 h-7 rounded-full bg-red-500/10 border border-red-500/30 flex items-center justify-center mt-0.5">
          <AlertCircle size={14} className="text-red-400" />
        </div>
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl rounded-tl-sm px-4 py-2.5 max-w-2xl text-red-300 text-sm">
          {message.content}
        </div>
      </div>
    )
  }

  if (isUser) {
    return (
      <div className="flex items-start gap-2 justify-end animate-slideUp">
        <div className="bg-jarvis-blue/30 border border-blue-500/30 rounded-xl rounded-tr-sm px-4 py-2.5 max-w-2xl text-gray-100 text-sm leading-relaxed">
          {message.content}
        </div>
        <div className="shrink-0 w-7 h-7 rounded-full bg-blue-500/10 border border-blue-500/30 flex items-center justify-center mt-0.5">
          <User size={14} className="text-blue-400" />
        </div>
      </div>
    )
  }

  // Assistant message
  return (
    <div className="flex items-start gap-2 animate-slideUp">
      <div className="shrink-0 w-7 h-7 rounded-full bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center mt-0.5">
        <Bot size={14} className="text-cyan-400" />
      </div>
      <div className="flex-1 max-w-2xl">
        {/* Tool calls */}
        {message.toolCalls.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mb-2">
            {message.toolCalls.map((tool, i) => (
              <ToolBadge key={i} tool={tool} />
            ))}
          </div>
        )}

        {/* Message body */}
        <div className={`bg-jarvis-card border border-jarvis-border rounded-xl rounded-tl-sm px-4 py-3 text-sm ${message.isStreaming ? 'typing-cursor' : ''}`}>
          {message.content ? (
            <div className="prose-jarvis">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {message.content}
              </ReactMarkdown>
            </div>
          ) : (
            <span className="text-gray-500 italic text-xs">Thinking…</span>
          )}
        </div>

        {/* Footer: usage + feedback */}
        {!message.isStreaming && (
          <div className="flex items-start justify-between mt-1 px-1">
            <div>
              {message.usage && <UsageStats usage={message.usage} />}
            </div>
            <div className="flex items-center gap-1">
              <button
                onClick={() => handleFeedback(1)}
                title="Helpful"
                className={`p-1 rounded transition-colors ${
                  feedback === 1 ? 'text-cyan-400' : 'text-gray-600 hover:text-gray-400'
                }`}
              >
                <ThumbsUp size={12} />
              </button>
              <button
                onClick={() => handleFeedback(-1)}
                title="Not helpful"
                className={`p-1 rounded transition-colors ${
                  feedback === -1 ? 'text-red-400' : 'text-gray-600 hover:text-gray-400'
                }`}
              >
                <ThumbsDown size={12} />
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

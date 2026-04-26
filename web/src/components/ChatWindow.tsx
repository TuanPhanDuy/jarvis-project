import { useEffect, useRef, useState, KeyboardEvent } from 'react'
import { Send, Loader2, Zap } from 'lucide-react'
import { MessageBubble } from './MessageBubble'
import type { ChatMessage } from '../types'

interface ChatWindowProps {
  messages: ChatMessage[]
  isThinking: boolean
  sessionId: string
  onSend: (text: string) => void
  disabled: boolean
}

const SUGGESTIONS = [
  'Explain transformer attention mechanisms',
  'How does RLHF work in language models?',
  'What is Constitutional AI?',
  'Summarize recent advances in multimodal models',
]

export function ChatWindow({ messages, isThinking, sessionId, onSend, disabled }: ChatWindowProps) {
  const bottomRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const [draft, setDraft] = useState('')

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isThinking])

  const handleSend = () => {
    const text = draft.trim()
    if (!text || disabled) return
    onSend(text)
    setDraft('')
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleInput = () => {
    const ta = textareaRef.current
    if (!ta) return
    ta.style.height = 'auto'
    ta.style.height = Math.min(ta.scrollHeight, 200) + 'px'
  }

  const isEmpty = messages.length === 0

  return (
    <div className="flex flex-col flex-1 min-h-0">
      {/* Messages area */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
        {isEmpty && !isThinking && (
          <div className="flex flex-col items-center justify-center h-full gap-6 py-12 animate-fadeIn">
            {/* Logo */}
            <div className="relative">
              <div className="w-16 h-16 rounded-full bg-cyan-500/10 border-2 border-cyan-500/40 flex items-center justify-center glow-cyan">
                <span className="text-cyan-400 font-bold text-2xl glow-cyan-text">J</span>
              </div>
              <div className="absolute inset-0 rounded-full border border-cyan-400/20 scale-150 animate-pulse" />
            </div>
            <div className="text-center">
              <h2 className="text-cyan-400 text-xl font-bold glow-cyan-text">JARVIS Online</h2>
              <p className="text-gray-500 text-sm mt-1">Just A Rather Very Intelligent System</p>
            </div>
            {/* Suggestions */}
            <div className="grid grid-cols-2 gap-2 max-w-xl w-full">
              {SUGGESTIONS.map(s => (
                <button
                  key={s}
                  onClick={() => onSend(s)}
                  className="text-left px-3 py-2.5 rounded-lg border border-jarvis-border bg-jarvis-card/60 hover:border-cyan-500/40 hover:bg-cyan-500/5 transition-all text-xs text-gray-400 hover:text-gray-200"
                >
                  <Zap size={11} className="inline mr-1.5 text-cyan-500" />
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map(msg => (
          <MessageBubble key={msg.id} message={msg} sessionId={sessionId} />
        ))}

        {isThinking && (
          <div className="flex items-center gap-2 animate-fadeIn">
            <div className="w-7 h-7 rounded-full bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center">
              <Loader2 size={13} className="text-cyan-400 animate-spin" />
            </div>
            <div className="flex gap-1">
              {[0, 1, 2].map(i => (
                <span
                  key={i}
                  className="w-1.5 h-1.5 rounded-full bg-cyan-500/60 animate-bounce"
                  style={{ animationDelay: `${i * 150}ms` }}
                />
              ))}
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input area */}
      <div className="shrink-0 px-4 py-3 border-t border-jarvis-border bg-jarvis-card/40">
        <div className="flex items-end gap-2 bg-jarvis-card border border-jarvis-border rounded-xl px-3 py-2 focus-within:border-cyan-500/50 focus-within:shadow-sm focus-within:shadow-cyan-500/10 transition-all">
          <textarea
            ref={textareaRef}
            value={draft}
            onChange={e => setDraft(e.target.value)}
            onKeyDown={handleKeyDown}
            onInput={handleInput}
            placeholder="Ask JARVIS anything… (Enter to send, Shift+Enter for newline)"
            rows={1}
            disabled={disabled}
            className="flex-1 bg-transparent text-gray-100 text-sm placeholder-gray-600 resize-none focus:outline-none leading-relaxed min-h-[24px] max-h-[200px]"
          />
          <button
            onClick={handleSend}
            disabled={!draft.trim() || disabled}
            className="shrink-0 w-8 h-8 rounded-lg bg-cyan-500/20 border border-cyan-500/40 flex items-center justify-center text-cyan-400 hover:bg-cyan-500/30 disabled:opacity-30 disabled:cursor-not-allowed transition-all glow-cyan"
          >
            <Send size={14} />
          </button>
        </div>
        <p className="text-xs text-gray-700 mt-1.5 text-center">
          JARVIS can make mistakes. Verify important information.
        </p>
      </div>
    </div>
  )
}

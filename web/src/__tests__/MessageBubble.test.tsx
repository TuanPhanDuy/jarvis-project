import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { MessageBubble } from '../components/MessageBubble'
import type { ChatMessage } from '../types'

vi.mock('../api', () => ({
  api: {
    submitFeedback: vi.fn().mockResolvedValue({}),
  },
}))

function makeMsg(overrides: Partial<ChatMessage> = {}): ChatMessage {
  return {
    id: 'msg-001',
    role: 'assistant',
    content: 'Hello, world!',
    toolCalls: [],
    isStreaming: false,
    timestamp: new Date('2024-01-01T00:00:00Z'),
    ...overrides,
  }
}

describe('MessageBubble', () => {
  it('renders user message content', () => {
    render(<MessageBubble message={makeMsg({ role: 'user', content: 'What is RLHF?' })} sessionId="s1" />)
    expect(screen.getByText('What is RLHF?')).toBeInTheDocument()
  })

  it('renders assistant message content', () => {
    render(<MessageBubble message={makeMsg({ content: 'RLHF stands for...' })} sessionId="s1" />)
    expect(screen.getByText(/RLHF stands for/)).toBeInTheDocument()
  })

  it('renders error message with red styling', () => {
    render(<MessageBubble message={makeMsg({ role: 'error', content: 'Something went wrong' })} sessionId="s1" />)
    const errorText = screen.getByText('Something went wrong')
    expect(errorText).toBeInTheDocument()
    // Error bubble uses red border class
    expect(errorText.closest('div')).toHaveClass('text-red-300')
  })

  it('renders tool call badges when toolCalls is non-empty', () => {
    render(
      <MessageBubble
        message={makeMsg({ toolCalls: ['web_search', 'save_report'] })}
        sessionId="s1"
      />
    )
    expect(screen.getByText('web_search')).toBeInTheDocument()
    expect(screen.getByText('save_report')).toBeInTheDocument()
  })

  it('does not render tool badges when toolCalls is empty', () => {
    render(<MessageBubble message={makeMsg({ toolCalls: [] })} sessionId="s1" />)
    expect(screen.queryByText('web_search')).not.toBeInTheDocument()
  })

  it('shows "Thinking…" text when streaming and no content', () => {
    render(
      <MessageBubble
        message={makeMsg({ isStreaming: true, content: '' })}
        sessionId="s1"
      />
    )
    expect(screen.getByText('Thinking…')).toBeInTheDocument()
  })

  it('hides feedback buttons while streaming', () => {
    render(
      <MessageBubble
        message={makeMsg({ isStreaming: true })}
        sessionId="s1"
      />
    )
    expect(screen.queryByTitle('Helpful')).not.toBeInTheDocument()
  })

  it('shows feedback thumbs buttons after streaming completes', () => {
    render(
      <MessageBubble
        message={makeMsg({ isStreaming: false })}
        sessionId="s1"
      />
    )
    expect(screen.getByTitle('Helpful')).toBeInTheDocument()
    expect(screen.getByTitle('Not helpful')).toBeInTheDocument()
  })

  it('submits feedback on thumbs up click', async () => {
    const { api } = await import('../api')
    render(
      <MessageBubble
        message={makeMsg({ isStreaming: false, content: 'Good response' })}
        sessionId="session-123"
      />
    )
    await userEvent.click(screen.getByTitle('Helpful'))
    await waitFor(() => {
      expect(api.submitFeedback).toHaveBeenCalledWith(
        expect.objectContaining({ session_id: 'session-123', rating: 1 })
      )
    })
  })
})

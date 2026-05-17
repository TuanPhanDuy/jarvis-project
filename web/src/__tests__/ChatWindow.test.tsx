import { render, screen, fireEvent } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, beforeAll } from 'vitest'
import { ChatWindow } from '../components/ChatWindow'
import type { ChatMessage } from '../types'

beforeAll(() => {
  window.HTMLElement.prototype.scrollIntoView = vi.fn()
})

const defaultProps = {
  messages: [] as ChatMessage[],
  isThinking: false,
  sessionId: 'sess-1',
  onSend: vi.fn(),
  disabled: false,
}

function makeMessage(overrides: Partial<ChatMessage> = {}): ChatMessage {
  return {
    id: 'msg-1',
    role: 'user',
    content: 'Hello JARVIS',
    toolCalls: [],
    isStreaming: false,
    timestamp: new Date(),
    ...overrides,
  }
}

describe('ChatWindow', () => {
  it('renders empty state with logo and suggestions when messages is empty', () => {
    render(<ChatWindow {...defaultProps} />)
    expect(screen.getByText('JARVIS Online')).toBeInTheDocument()
    expect(screen.getByText('Explain transformer attention mechanisms')).toBeInTheDocument()
    expect(screen.getByText('How does RLHF work in language models?')).toBeInTheDocument()
    expect(screen.getByText('What is Constitutional AI?')).toBeInTheDocument()
  })

  it('renders provided messages', () => {
    const msg = makeMessage({ content: 'Tell me about PPO' })
    render(<ChatWindow {...defaultProps} messages={[msg]} />)
    expect(screen.getByText('Tell me about PPO')).toBeInTheDocument()
    expect(screen.queryByText('JARVIS Online')).not.toBeInTheDocument()
  })

  it('clicking a suggestion calls onSend with the suggestion text', () => {
    const onSend = vi.fn()
    render(<ChatWindow {...defaultProps} onSend={onSend} />)
    fireEvent.click(screen.getByText('What is Constitutional AI?'))
    expect(onSend).toHaveBeenCalledWith('What is Constitutional AI?')
  })

  it('Enter key sends the draft and clears the input', async () => {
    const user = userEvent.setup()
    const onSend = vi.fn()
    render(<ChatWindow {...defaultProps} onSend={onSend} />)
    const textarea = screen.getByPlaceholderText(/Ask JARVIS anything/i)
    await user.type(textarea, 'hello world')
    await user.keyboard('{Enter}')
    expect(onSend).toHaveBeenCalledWith('hello world')
    expect(textarea).toHaveValue('')
  })

  it('Shift+Enter does not send the message', async () => {
    const user = userEvent.setup()
    const onSend = vi.fn()
    render(<ChatWindow {...defaultProps} onSend={onSend} />)
    const textarea = screen.getByPlaceholderText(/Ask JARVIS anything/i)
    await user.type(textarea, 'hello')
    await user.keyboard('{Shift>}{Enter}{/Shift}')
    expect(onSend).not.toHaveBeenCalled()
  })

  it('send button is disabled when draft is empty', () => {
    render(<ChatWindow {...defaultProps} />)
    const sendBtn = screen.getByRole('button', { name: '' })
    // The send button (Send icon) should be disabled with empty draft
    const buttons = screen.getAllByRole('button')
    // The send button is the last button in the input area (after suggestion buttons in empty state)
    const sendButton = buttons.find(b => b.hasAttribute('disabled'))
    expect(sendButton).toBeTruthy()
  })

  it('shows thinking indicator when isThinking is true', () => {
    render(<ChatWindow {...defaultProps} isThinking={true} />)
    // Thinking shows a spinner — the empty state is hidden and the loader appears
    expect(screen.queryByText('JARVIS Online')).not.toBeInTheDocument()
    // Three bouncing dots are rendered
    const dots = document.querySelectorAll('.animate-bounce')
    expect(dots.length).toBe(3)
  })
})

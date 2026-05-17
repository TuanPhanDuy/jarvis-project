import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { Header } from '../components/Header'

const defaultProps = {
  connected: true,
  sessionId: 'sess-abcdef1234567890',
  agentMode: 'planner' as const,
  onAgentModeChange: vi.fn(),
  onNewSession: vi.fn(),
}

beforeEach(() => {
  vi.clearAllMocks()
  global.fetch = vi.fn().mockResolvedValue({ ok: false })
})

describe('Header', () => {
  it('renders JARVIS brand name', () => {
    render(<Header {...defaultProps} />)
    expect(screen.getByText('JARVIS')).toBeInTheDocument()
  })

  it('shows Connected indicator when connected', () => {
    render(<Header {...defaultProps} connected={true} />)
    expect(screen.getByText('Connected')).toBeInTheDocument()
  })

  it('shows Reconnecting indicator when disconnected', () => {
    render(<Header {...defaultProps} connected={false} />)
    expect(screen.getByText(/Reconnecting/i)).toBeInTheDocument()
  })

  it('calls onAgentModeChange when mode button clicked', async () => {
    const handler = vi.fn()
    render(<Header {...defaultProps} onAgentModeChange={handler} />)
    await userEvent.click(screen.getByText('Researcher'))
    expect(handler).toHaveBeenCalledWith('researcher')
  })

  it('calls onNewSession when refresh button clicked', async () => {
    const handler = vi.fn()
    render(<Header {...defaultProps} onNewSession={handler} />)
    await userEvent.click(screen.getByTitle('New Session'))
    expect(handler).toHaveBeenCalled()
  })

  it('shows truncated session id in header', () => {
    render(<Header {...defaultProps} sessionId="sess-abcdef1234567890" />)
    expect(screen.getByText('sess-abc…')).toBeInTheDocument()
  })
})

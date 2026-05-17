import { render, screen, act, fireEvent } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { ProactiveToast } from '../components/ProactiveToast'
import type { ProactiveNotification } from '../types'

const makeNotification = (overrides: Partial<ProactiveNotification> = {}): ProactiveNotification => ({
  id: 'n1',
  trigger: 'daily_digest',
  text: 'Your morning briefing is ready.',
  severity: 'info',
  timestamp: new Date(),
  ...overrides,
})

describe('ProactiveToast', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('renders nothing when notifications array is empty', () => {
    const { container } = render(<ProactiveToast notifications={[]} onDismiss={vi.fn()} />)
    expect(container.firstChild).toBeNull()
  })

  it('renders notification text', () => {
    const n = makeNotification({ text: 'System snapshot complete.' })
    render(<ProactiveToast notifications={[n]} onDismiss={vi.fn()} />)
    expect(screen.getByText('System snapshot complete.')).toBeInTheDocument()
  })

  it('renders multiple notifications', () => {
    const notifications = [
      makeNotification({ id: 'n1', text: 'First notification' }),
      makeNotification({ id: 'n2', text: 'Second notification' }),
    ]
    render(<ProactiveToast notifications={notifications} onDismiss={vi.fn()} />)
    expect(screen.getByText('First notification')).toBeInTheDocument()
    expect(screen.getByText('Second notification')).toBeInTheDocument()
  })

  it('calls onDismiss with correct id when X button clicked', () => {
    const onDismiss = vi.fn()
    const n = makeNotification({ id: 'abc123' })
    render(<ProactiveToast notifications={[n]} onDismiss={onDismiss} />)
    fireEvent.click(screen.getByRole('button'))
    expect(onDismiss).toHaveBeenCalledWith('abc123')
  })

  it('auto-dismisses after 12 seconds', () => {
    const onDismiss = vi.fn()
    const n = makeNotification({ id: 'auto1' })
    render(<ProactiveToast notifications={[n]} onDismiss={onDismiss} />)
    expect(onDismiss).not.toHaveBeenCalled()
    act(() => { vi.advanceTimersByTime(12000) })
    expect(onDismiss).toHaveBeenCalledWith('auto1')
  })

  it('displays trigger label with underscores replaced by spaces', () => {
    const n = makeNotification({ trigger: 'memory_consolidation' })
    render(<ProactiveToast notifications={[n]} onDismiss={vi.fn()} />)
    expect(screen.getByText(/memory consolidation/i)).toBeInTheDocument()
  })
})

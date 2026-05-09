import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi } from 'vitest'
import { ApprovalModal } from '../components/ApprovalModal'
import type { ApprovalRequest } from '../types'

const makeRequest = (overrides: Partial<ApprovalRequest> = {}): ApprovalRequest => ({
  request_id: 'req-001',
  tool_name: 'run_command',
  description: 'Run ls -la in the project root',
  risk_level: 'MEDIUM',
  expires_in: 30,
  ...overrides,
})

describe('ApprovalModal', () => {

  it('renders tool name', () => {
    render(
      <ApprovalModal
        request={makeRequest()}
        onApprove={vi.fn()}
        onDeny={vi.fn()}
      />
    )
    expect(screen.getByText('run_command')).toBeInTheDocument()
  })

  it('renders description', () => {
    render(
      <ApprovalModal
        request={makeRequest()}
        onApprove={vi.fn()}
        onDeny={vi.fn()}
      />
    )
    expect(screen.getByText('Run ls -la in the project root')).toBeInTheDocument()
  })

  it('renders risk level badge', () => {
    render(
      <ApprovalModal
        request={makeRequest({ risk_level: 'HIGH' })}
        onApprove={vi.fn()}
        onDeny={vi.fn()}
      />
    )
    expect(screen.getByText('HIGH')).toBeInTheDocument()
  })

  it('calls onApprove with request_id when Approve is clicked', async () => {
    const onApprove = vi.fn()
    render(
      <ApprovalModal
        request={makeRequest()}
        onApprove={onApprove}
        onDeny={vi.fn()}
      />
    )
    await userEvent.click(screen.getByText('Approve'))
    expect(onApprove).toHaveBeenCalledWith('req-001')
  })

  it('calls onDeny with request_id when Deny is clicked', async () => {
    const onDeny = vi.fn()
    render(
      <ApprovalModal
        request={makeRequest()}
        onApprove={vi.fn()}
        onDeny={onDeny}
      />
    )
    await userEvent.click(screen.getByText('Deny'))
    expect(onDeny).toHaveBeenCalledWith('req-001')
  })

  it('calls onDeny when X button is clicked', async () => {
    const onDeny = vi.fn()
    render(
      <ApprovalModal
        request={makeRequest()}
        onApprove={vi.fn()}
        onDeny={onDeny}
      />
    )
    // X button is the first button (close icon)
    const buttons = screen.getAllByRole('button')
    const closeBtn = buttons.find(b => !b.textContent?.includes('Approve') && !b.textContent?.includes('Deny'))!
    await userEvent.click(closeBtn)
    expect(onDeny).toHaveBeenCalledWith('req-001')
  })

  it('renders countdown timer', () => {
    render(
      <ApprovalModal
        request={makeRequest({ expires_in: 60 })}
        onApprove={vi.fn()}
        onDeny={vi.fn()}
      />
    )
    expect(screen.getByText('60s')).toBeInTheDocument()
  })
})

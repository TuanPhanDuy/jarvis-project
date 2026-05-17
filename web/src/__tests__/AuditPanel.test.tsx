import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { AuditPanel } from '../components/AuditPanel'

vi.mock('../api', () => ({
  api: {
    listAudit: vi.fn(),
  },
}))

import { api } from '../api'

const mockListAudit = api.listAudit as ReturnType<typeof vi.fn>

const SAMPLE_ENTRIES = [
  { session_id: 'sess1', tool_name: 'web_search', risk_level: 'LOW', approved: true, timestamp: '2024-01-01T10:00:00Z' },
  { session_id: 'sess1', tool_name: 'run_command', risk_level: 'HIGH', approved: false, timestamp: '2024-01-01T10:01:00Z' },
]

describe('AuditPanel', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders audit entries from api', async () => {
    mockListAudit.mockResolvedValue(SAMPLE_ENTRIES)
    render(<AuditPanel sessionId="sess1" />)
    await waitFor(() => {
      expect(screen.getByText('web_search')).toBeInTheDocument()
      expect(screen.getByText('run_command')).toBeInTheDocument()
    })
  })

  it('shows tool name for each entry', async () => {
    mockListAudit.mockResolvedValue(SAMPLE_ENTRIES)
    render(<AuditPanel sessionId="sess1" />)
    await waitFor(() => {
      expect(screen.getByText('web_search')).toBeInTheDocument()
    })
  })

  it('shows HIGH risk label', async () => {
    mockListAudit.mockResolvedValue(SAMPLE_ENTRIES)
    render(<AuditPanel sessionId="sess1" />)
    await waitFor(() => {
      expect(screen.getByText('HIGH')).toBeInTheDocument()
    })
  })

  it('calls api with the given sessionId', async () => {
    mockListAudit.mockResolvedValue([])
    render(<AuditPanel sessionId="my-session-42" />)
    await waitFor(() => {
      expect(mockListAudit).toHaveBeenCalledWith('my-session-42', 50)
    })
  })

  it('shows placeholder when entries list is empty', async () => {
    mockListAudit.mockResolvedValue([])
    render(<AuditPanel sessionId="sess1" />)
    await waitFor(() => {
      expect(screen.getByText(/No tool calls recorded yet/i)).toBeInTheDocument()
    })
  })

  it('reloads entries when sessionId changes', async () => {
    mockListAudit.mockResolvedValue([])
    const { rerender } = render(<AuditPanel sessionId="sess-a" />)
    await waitFor(() => expect(mockListAudit).toHaveBeenCalledWith('sess-a', 50))

    mockListAudit.mockResolvedValue(SAMPLE_ENTRIES)
    rerender(<AuditPanel sessionId="sess-b" />)
    await waitFor(() => expect(mockListAudit).toHaveBeenCalledWith('sess-b', 50))
  })
})

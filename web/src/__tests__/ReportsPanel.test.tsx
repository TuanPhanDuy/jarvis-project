import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { ReportsPanel } from '../components/ReportsPanel'

vi.mock('../api', () => ({
  api: {
    listReports: vi.fn(),
    getReport: vi.fn(),
  },
}))

import { api } from '../api'

const mockList = api.listReports as ReturnType<typeof vi.fn>
const mockGet = api.getReport as ReturnType<typeof vi.fn>

const SAMPLE_REPORTS = [
  { name: 'rlhf_notes.md', size_bytes: 2048, modified: 1700000000 },
  { name: 'transformer_deep_dive.md', size_bytes: 5120, modified: 1700100000 },
]

describe('ReportsPanel', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders report list after load', async () => {
    mockList.mockResolvedValue(SAMPLE_REPORTS)

    render(<ReportsPanel />)

    await waitFor(() => {
      expect(screen.getByText('rlhf_notes')).toBeInTheDocument()
      expect(screen.getByText('transformer_deep_dive')).toBeInTheDocument()
    })
  })

  it('shows "No reports yet" when list is empty', async () => {
    mockList.mockResolvedValue([])

    render(<ReportsPanel />)

    await waitFor(() => {
      expect(screen.getByText(/No reports yet/i)).toBeInTheDocument()
    })
  })

  it('opens report content when clicked', async () => {
    mockList.mockResolvedValue(SAMPLE_REPORTS)
    mockGet.mockResolvedValue({ name: 'rlhf_notes.md', content: '# RLHF Notes\n\nContent here.' })

    render(<ReportsPanel />)

    await waitFor(() => screen.getByText('rlhf_notes'))
    await userEvent.click(screen.getByText('rlhf_notes'))

    await waitFor(() => {
      expect(screen.getByText('rlhf_notes.md')).toBeInTheDocument()
    })
    expect(mockGet).toHaveBeenCalledWith('rlhf_notes.md')
  })

  it('shows error content when report fetch fails', async () => {
    mockList.mockResolvedValue(SAMPLE_REPORTS)
    mockGet.mockRejectedValue(new Error('500'))

    render(<ReportsPanel />)

    await waitFor(() => screen.getByText('rlhf_notes'))
    await userEvent.click(screen.getByText('rlhf_notes'))

    await waitFor(() => {
      expect(screen.getByText(/Failed to load report/i)).toBeInTheDocument()
    })
  })

  it('navigates back to list from detail view', async () => {
    mockList.mockResolvedValue(SAMPLE_REPORTS)
    mockGet.mockResolvedValue({ name: 'rlhf_notes.md', content: '# Content' })

    render(<ReportsPanel />)

    await waitFor(() => screen.getByText('rlhf_notes'))
    await userEvent.click(screen.getByText('rlhf_notes'))
    await waitFor(() => screen.getByTitle('Back to list'))
    await userEvent.click(screen.getByTitle('Back to list'))

    await waitFor(() => {
      expect(screen.getByText('rlhf_notes')).toBeInTheDocument()
    })
  })
})

import { render, screen, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { FeedbackChart } from '../components/FeedbackChart'

vi.mock('../api', () => ({
  api: {
    getFeedbackStats: vi.fn(),
  },
}))

import { api } from '../api'

const mockGetStats = api.getFeedbackStats as ReturnType<typeof vi.fn>

describe('FeedbackChart', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('shows "No feedback yet" when stats are null (API returns nothing)', async () => {
    mockGetStats.mockRejectedValue(new Error('404'))

    render(<FeedbackChart />)

    await waitFor(() => {
      expect(screen.getByText(/No feedback yet/i)).toBeInTheDocument()
    })
  })

  it('renders satisfaction percentage when stats load', async () => {
    mockGetStats.mockResolvedValue({
      total: 10,
      avg_rating: 0.6,   // => 80% satisfaction
      recent: [{ rating: 1, comment: '' }, { rating: -1, comment: '' }],
    })

    render(<FeedbackChart />)

    await waitFor(() => {
      expect(screen.getByText(/80%/)).toBeInTheDocument()
    })
  })

  it('renders total ratings count', async () => {
    mockGetStats.mockResolvedValue({
      total: 42,
      avg_rating: 0.0,
      recent: [],
    })

    render(<FeedbackChart />)

    await waitFor(() => {
      expect(screen.getByText(/42 total ratings/)).toBeInTheDocument()
    })
  })

  it('renders recent comments when present', async () => {
    mockGetStats.mockResolvedValue({
      total: 3,
      avg_rating: 1.0,
      recent: [
        { rating: 1, comment: 'Great response!' },
        { rating: -1, comment: 'Not helpful' },
        { rating: 1, comment: '' },
      ],
    })

    render(<FeedbackChart />)

    await waitFor(() => {
      expect(screen.getByText('Great response!')).toBeInTheDocument()
      expect(screen.getByText('Not helpful')).toBeInTheDocument()
    })
  })

  it('does not crash with empty recent array', async () => {
    mockGetStats.mockResolvedValue({
      total: 0,
      avg_rating: 0,
      recent: [],
    })

    render(<FeedbackChart />)

    await waitFor(() => {
      expect(screen.getByText(/0 total ratings/)).toBeInTheDocument()
    })
  })
})

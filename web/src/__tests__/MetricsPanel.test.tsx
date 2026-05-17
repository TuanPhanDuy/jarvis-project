import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { MetricsPanel } from '../components/MetricsPanel'

const SAMPLE_METRICS = {
  requests_total: { http: 42, websocket: 7 },
  tool_calls: { web_search: 15, save_report: 8 },
  active_ws_connections: 3,
  active_sessions: 5,
}

function mockFetch(data: unknown, ok = true) {
  global.fetch = vi.fn().mockResolvedValue({
    ok,
    json: () => Promise.resolve(data),
  })
}

describe('MetricsPanel', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('shows loading state before data arrives', () => {
    global.fetch = vi.fn().mockReturnValue(new Promise(() => {}))
    render(<MetricsPanel />)
    expect(screen.getByText(/Loading metrics/i)).toBeInTheDocument()
  })

  it('fetches metrics on mount', async () => {
    mockFetch(SAMPLE_METRICS)
    render(<MetricsPanel />)
    await waitFor(() => expect(global.fetch).toHaveBeenCalledWith('/api/metrics/summary'))
  })

  it('displays total request count', async () => {
    mockFetch(SAMPLE_METRICS)
    render(<MetricsPanel />)
    // 42 http + 7 websocket = 49
    await waitFor(() => expect(screen.getByText('49')).toBeInTheDocument())
  })

  it('displays active websocket connections', async () => {
    mockFetch(SAMPLE_METRICS)
    render(<MetricsPanel />)
    await waitFor(() => expect(screen.getByText('3')).toBeInTheDocument())
  })

  it('displays tool call bar chart entries', async () => {
    mockFetch(SAMPLE_METRICS)
    render(<MetricsPanel />)
    await waitFor(() => {
      expect(screen.getByText('web_search')).toBeInTheDocument()
      expect(screen.getByText('save_report')).toBeInTheDocument()
    })
  })

  it('shows no-data placeholder when fetch fails', async () => {
    global.fetch = vi.fn().mockRejectedValue(new Error('Network error'))
    render(<MetricsPanel />)
    await waitFor(() => {
      expect(screen.getByText(/No data yet/i)).toBeInTheDocument()
    })
  })
})

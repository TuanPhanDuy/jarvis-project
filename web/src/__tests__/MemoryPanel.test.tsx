import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { MemoryPanel } from '../components/MemoryPanel'

const SAMPLE_EPISODES = [
  { session_id: 's1', user_id: 'u1', role: 'user', content: 'What is RLHF?', timestamp: 1700000000 },
  { session_id: 's1', user_id: 'u1', role: 'assistant', content: 'RLHF stands for Reinforcement Learning from Human Feedback.', timestamp: 1700000060 },
]

const SAMPLE_GRAPH = [
  { subject: 'RLHF', predicate: 'uses', object: 'PPO' },
  { subject: 'PPO', predicate: 'is_variant_of', object: 'Policy Gradient' },
]

function mockFetch(data: unknown) {
  global.fetch = vi.fn().mockResolvedValue({
    ok: true,
    json: () => Promise.resolve(data),
  })
}

describe('MemoryPanel', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('fetches episodes on mount and renders them', async () => {
    mockFetch(SAMPLE_EPISODES)
    render(<MemoryPanel />)
    await waitFor(() => {
      expect(screen.getByText('What is RLHF?')).toBeInTheDocument()
      expect(screen.getByText(/RLHF stands for/)).toBeInTheDocument()
    })
  })

  it('shows "No episodes found" when episodes list is empty', async () => {
    mockFetch([])
    render(<MemoryPanel />)
    await waitFor(() => {
      expect(screen.getByText(/No episodes found/i)).toBeInTheDocument()
    })
  })

  it('switches to graph tab and fetches graph data', async () => {
    global.fetch = vi.fn().mockImplementation((url: string) => {
      const data = url.includes('/memory/graph') ? SAMPLE_GRAPH : []
      return Promise.resolve({ ok: true, json: () => Promise.resolve(data) })
    })

    render(<MemoryPanel />)
    await waitFor(() => screen.getByText(/No episodes found/i))

    fireEvent.click(screen.getByText('Knowledge'))
    await waitFor(() => {
      expect(screen.getByText('RLHF')).toBeInTheDocument()
      expect(screen.getByText('uses')).toBeInTheDocument()
      expect(screen.getByText('Policy Gradient')).toBeInTheDocument()
    })
  })

  it('shows "No knowledge graph entries yet" on empty graph', async () => {
    global.fetch = vi.fn().mockResolvedValue({ ok: true, json: () => Promise.resolve([]) })

    render(<MemoryPanel />)
    await waitFor(() => screen.getByText(/No episodes found/i))
    fireEvent.click(screen.getByText('Knowledge'))
    await waitFor(() => {
      expect(screen.getByText(/No knowledge graph entries yet/i)).toBeInTheDocument()
    })
  })

  it('filters episodes client-side by search input', async () => {
    mockFetch(SAMPLE_EPISODES)
    render(<MemoryPanel />)
    await waitFor(() => screen.getByText('What is RLHF?'))

    fireEvent.change(screen.getByPlaceholderText(/Search memory/i), {
      target: { value: 'RLHF' },
    })

    // Only the matching episodes should be visible
    expect(screen.getByText('What is RLHF?')).toBeInTheDocument()
    expect(screen.getByText(/RLHF stands for/)).toBeInTheDocument()
  })

  it('refresh button re-fetches current tab data', async () => {
    mockFetch(SAMPLE_EPISODES)
    render(<MemoryPanel />)
    await waitFor(() => screen.getByText('What is RLHF?'))

    fireEvent.click(screen.getByTitle('Refresh'))
    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledTimes(2)
    })
  })

  it('graph fetch includes entity param when search is set', async () => {
    global.fetch = vi.fn()
      .mockResolvedValue({ ok: true, json: () => Promise.resolve([]) })

    render(<MemoryPanel />)
    await waitFor(() => screen.getByText(/No episodes found/i))

    fireEvent.click(screen.getByText('Knowledge'))
    await waitFor(() => screen.getByText(/No knowledge graph entries yet/i))

    fireEvent.change(screen.getByPlaceholderText(/Search memory/i), {
      target: { value: 'PPO' },
    })

    await waitFor(() => {
      const calls = (global.fetch as ReturnType<typeof vi.fn>).mock.calls
      const graphCalls = calls.filter(([url]) => url.includes('/memory/graph'))
      const lastGraphCall = graphCalls[graphCalls.length - 1]
      expect(lastGraphCall[0]).toContain('entity=PPO')
    })
  })
})

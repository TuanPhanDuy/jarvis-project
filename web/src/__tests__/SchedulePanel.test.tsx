import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { SchedulePanel } from '../components/SchedulePanel'

vi.mock('../api', () => ({
  api: {
    listSchedules: vi.fn(),
    createSchedule: vi.fn(),
    deleteSchedule: vi.fn(),
  },
}))

import { api } from '../api'

const mockList = api.listSchedules as ReturnType<typeof vi.fn>
const mockCreate = api.createSchedule as ReturnType<typeof vi.fn>
const mockDelete = api.deleteSchedule as ReturnType<typeof vi.fn>

const SAMPLE_SCHEDULES = [
  { job_id: 'job-1', job_type: 'research', subject: 'RLHF overview', cron: '0 9 * * *', next_run: '2024-06-01T09:00:00Z' },
  { job_id: 'job-2', job_type: 'monitor', subject: 'LLM news', cron: '0 8 * * 1', next_run: null },
]

describe('SchedulePanel', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockCreate.mockResolvedValue({})
    mockDelete.mockResolvedValue({})
  })

  it('loads and renders schedule list on mount', async () => {
    mockList.mockResolvedValue(SAMPLE_SCHEDULES)
    render(<SchedulePanel sessionId="sess-1" />)
    await waitFor(() => {
      expect(screen.getByText('RLHF overview')).toBeInTheDocument()
      expect(screen.getByText('LLM news')).toBeInTheDocument()
    })
  })

  it('shows "No schedules yet" when list is empty', async () => {
    mockList.mockResolvedValue([])
    render(<SchedulePanel sessionId="sess-1" />)
    await waitFor(() => {
      expect(screen.getByText(/No schedules yet/i)).toBeInTheDocument()
    })
  })

  it('Plus button toggles form visibility', async () => {
    mockList.mockResolvedValue([])
    render(<SchedulePanel sessionId="sess-1" />)
    await waitFor(() => screen.getByText(/No schedules yet/i))

    const buttons = screen.getAllByRole('button')
    const plusBtn = buttons.find(b => b.querySelector('svg'))!
    // Click the + button (second one, after refresh)
    fireEvent.click(buttons[1])
    expect(screen.getByText('Create')).toBeInTheDocument()
    expect(screen.getByText('Cancel')).toBeInTheDocument()
  })

  it('Cancel button hides the form', async () => {
    mockList.mockResolvedValue([])
    render(<SchedulePanel sessionId="sess-1" />)
    await waitFor(() => screen.getByText(/No schedules yet/i))

    fireEvent.click(screen.getAllByRole('button')[1])  // Plus button
    expect(screen.getByText('Create')).toBeInTheDocument()

    fireEvent.click(screen.getByText('Cancel'))
    expect(screen.queryByText('Create')).not.toBeInTheDocument()
  })

  it('Create calls api.createSchedule with correct payload and reloads list', async () => {
    mockList.mockResolvedValue([])
    render(<SchedulePanel sessionId="sess-42" />)
    await waitFor(() => screen.getByText(/No schedules yet/i))

    fireEvent.click(screen.getAllByRole('button')[1])  // Plus

    fireEvent.change(screen.getByPlaceholderText('Topic…'), {
      target: { value: 'Constitutional AI' },
    })
    fireEvent.click(screen.getByText('Create'))

    await waitFor(() => {
      expect(mockCreate).toHaveBeenCalledWith({
        job_type: 'research',
        params: { topic: 'Constitutional AI' },
        cron: '0 9 * * *',
        session_id: 'sess-42',
      })
    })
    expect(mockList).toHaveBeenCalledTimes(2)  // initial load + reload after create
    expect(screen.queryByText('Cancel')).not.toBeInTheDocument()
  })

  it('Create does nothing when topic is empty', async () => {
    mockList.mockResolvedValue([])
    render(<SchedulePanel sessionId="sess-1" />)
    await waitFor(() => screen.getByText(/No schedules yet/i))

    fireEvent.click(screen.getAllByRole('button')[1])  // Plus
    fireEvent.click(screen.getByText('Create'))  // topic is empty

    expect(mockCreate).not.toHaveBeenCalled()
  })

  it('topic placeholder changes based on job_type', async () => {
    mockList.mockResolvedValue([])
    render(<SchedulePanel sessionId="sess-1" />)
    await waitFor(() => screen.getByText(/No schedules yet/i))

    fireEvent.click(screen.getAllByRole('button')[1])  // Plus
    expect(screen.getByPlaceholderText('Topic…')).toBeInTheDocument()

    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'monitor' } })
    expect(screen.getByPlaceholderText('Query to monitor…')).toBeInTheDocument()
  })

  it('Delete button calls api.deleteSchedule and removes item from list', async () => {
    mockList.mockResolvedValue(SAMPLE_SCHEDULES)
    render(<SchedulePanel sessionId="sess-1" />)
    await waitFor(() => screen.getByText('RLHF overview'))

    // Delete buttons are hidden (opacity-0) until hover — trigger via direct click
    const deleteButtons = screen.getAllByRole('button').filter(b =>
      b.querySelector('svg') && b.className.includes('opacity-0')
    )
    fireEvent.click(deleteButtons[0])

    await waitFor(() => {
      expect(mockDelete).toHaveBeenCalledWith('job-1')
      expect(screen.queryByText('RLHF overview')).not.toBeInTheDocument()
      expect(screen.getByText('LLM news')).toBeInTheDocument()
    })
  })
})

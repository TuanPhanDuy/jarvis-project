export interface UsageSummary {
  input_tokens: number
  output_tokens: number
  cache_write_tokens: number
  cache_read_tokens: number
  estimated_cost_usd: number
}

// WebSocket messages: server → client
export interface WsThinking { type: 'thinking'; message: string }
export interface WsChunk { type: 'chunk'; text: string }
export interface WsToolCall { type: 'tool_call'; tool: string }
export interface WsDone { type: 'done'; text: string; usage: UsageSummary }
export interface WsError { type: 'error'; message: string }
export interface WsApprovalRequest {
  type: 'approval_request'
  request_id: string
  tool_name: string
  description: string
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'
  expires_in: number
}
export interface WsProactive {
  type: 'proactive'
  trigger: string
  text: string
  severity: 'info' | 'warning' | 'critical'
}

export type WsServerMessage =
  | WsThinking | WsChunk | WsToolCall | WsDone
  | WsError | WsApprovalRequest | WsProactive

// App-level message model
export interface ChatMessage {
  id: string
  role: 'user' | 'assistant' | 'error'
  content: string
  toolCalls: string[]
  usage?: UsageSummary
  isStreaming: boolean
  timestamp: Date
}

export interface ApprovalRequest {
  request_id: string
  tool_name: string
  description: string
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'
  expires_in: number
}

export interface ProactiveNotification {
  id: string
  trigger: string
  text: string
  severity: 'info' | 'warning' | 'critical'
  timestamp: Date
}

export interface ScheduleItem {
  job_id: string
  job_type: string
  subject: string
  cron: string
  next_run: string | null
}

export interface AuditEntry {
  session_id: string
  tool_name: string
  risk_level: string
  approved: boolean | null
  timestamp: string
}

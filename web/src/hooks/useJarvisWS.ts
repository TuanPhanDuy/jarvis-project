import { useEffect, useRef, useCallback, useState } from 'react'
import type { WsServerMessage } from '../types'

const MAX_RETRIES = 10
const BASE_DELAY_MS = 3000

function getWsBase() {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${proto}//${window.location.host}`
}

export function useJarvisWS(
  sessionId: string,
  onMessage: (msg: WsServerMessage) => void,
) {
  const wsRef = useRef<WebSocket | null>(null)
  const [connected, setConnected] = useState(false)
  const [permanentError, setPermanentError] = useState(false)
  const onMessageRef = useRef(onMessage)
  onMessageRef.current = onMessage

  const retryCount = useRef(0)
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    const ws = new WebSocket(`${getWsBase()}/api/ws/${sessionId}`)
    wsRef.current = ws

    ws.onopen = () => {
      setConnected(true)
      setPermanentError(false)
      retryCount.current = 0
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
    }

    ws.onclose = () => {
      setConnected(false)
      if (retryCount.current >= MAX_RETRIES) {
        setPermanentError(true)
        return
      }
      const delay = Math.min(BASE_DELAY_MS * Math.pow(2, retryCount.current), 60_000)
      retryCount.current += 1
      reconnectTimer.current = setTimeout(connect, delay)
    }

    ws.onerror = () => ws.close()

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data) as WsServerMessage
        onMessageRef.current(msg)
      } catch {
        // ignore parse errors
      }
    }
  }, [sessionId])

  const resetAndReconnect = useCallback(() => {
    retryCount.current = 0
    setPermanentError(false)
    connect()
  }, [connect])

  useEffect(() => {
    connect()
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
      wsRef.current?.close()
    }
  }, [connect])

  const send = useCallback((data: object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data))
    }
  }, [])

  return { connected, permanentError, resetAndReconnect, send }
}

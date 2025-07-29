import { useEffect, useRef, useState, useCallback } from 'react'
import { useQueryClient } from '@tanstack/react-query'

export interface WebSocketMessage {
  type: string
  data: any
  timestamp: string
}

interface UseWebSocketOptions {
  reconnectAttempts?: number
  reconnectInterval?: number
  onMessage?: (message: WebSocketMessage) => void
  onConnect?: () => void
  onDisconnect?: () => void
  onError?: (error: Event) => void
}

export const useWebSocket = (url: string, options: UseWebSocketOptions = {}) => {
  const {
    reconnectAttempts = 5,
    reconnectInterval = 3000,
    onMessage,
    onConnect,
    onDisconnect,
    onError,
  } = options

  const [isConnected, setIsConnected] = useState(false)
  const [reconnectCount, setReconnectCount] = useState(0)
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null)
  const [connectionError, setConnectionError] = useState<string | null>(null)

  const ws = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const queryClient = useQueryClient()

  const connect = useCallback(() => {
    try {
      // Use wss:// for https and ws:// for http
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const wsUrl = url.startsWith('ws') ? url : `${protocol}//localhost:8080${url}`
      
      ws.current = new WebSocket(wsUrl)

      ws.current.onopen = () => {
        console.log('WebSocket connected to:', wsUrl)
        setIsConnected(true)
        setReconnectCount(0)
        setConnectionError(null)
        onConnect?.()
      }

      ws.current.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data)
          setLastMessage(message)
          onMessage?.(message)

          // Auto-invalidate queries based on message type
          switch (message.type) {
            case 'agent_status_update':
            case 'agent_heartbeat':
              queryClient.invalidateQueries({ queryKey: ['agents'] })
              break
            case 'idea_status_update':
            case 'workflow_transition':
              queryClient.invalidateQueries({ queryKey: ['ideas'] })
              break
            case 'system_metrics_update':
              queryClient.invalidateQueries({ queryKey: ['system-metrics'] })
              break
            default:
              break
          }
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error)
        }
      }

      ws.current.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason)
        setIsConnected(false)
        onDisconnect?.()

        // Attempt to reconnect if not a normal closure
        if (event.code !== 1000 && reconnectCount < reconnectAttempts) {
          setConnectionError(`Connection lost. Reconnecting... (${reconnectCount + 1}/${reconnectAttempts})`)
          reconnectTimeoutRef.current = setTimeout(() => {
            setReconnectCount(prev => prev + 1)
            connect()
          }, reconnectInterval)
        } else if (reconnectCount >= reconnectAttempts) {
          setConnectionError('Failed to reconnect. Please refresh the page.')
        }
      }

      ws.current.onerror = (error) => {
        console.error('WebSocket error:', error)
        setConnectionError('WebSocket connection error')
        onError?.(error)
      }
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error)
      setConnectionError('Failed to establish connection')
    }
  }, [url, reconnectCount, reconnectAttempts, reconnectInterval, onMessage, onConnect, onDisconnect, onError, queryClient])

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }
    
    if (ws.current) {
      ws.current.close(1000, 'User initiated disconnect')
      ws.current = null
    }
    
    setIsConnected(false)
    setReconnectCount(0)
    setConnectionError(null)
  }, [])

  const sendMessage = useCallback((message: any) => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(message))
      return true
    }
    console.warn('WebSocket is not connected. Cannot send message:', message)
    return false
  }, [])

  // Connect on mount
  useEffect(() => {
    connect()
    return disconnect
  }, [connect, disconnect])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect()
    }
  }, [disconnect])

  return {
    isConnected,
    lastMessage,
    connectionError,
    reconnectCount,
    sendMessage,
    disconnect,
    reconnect: connect,
  }
}

export default useWebSocket
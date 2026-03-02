import React, { useEffect, useState, useRef, useCallback } from 'react'
import Sidebar from './components/Sidebar'
import Chat from './components/Chat'
import Dataset from './components/Dataset'
import Models from './components/Models'

export default function App() {
    const [wsUrl, setWsUrl] = useState(null)
    const [activePage, setActivePage] = useState('chat')

    // Lift WebSocket state to App level to persist across tab switches
    const [messages, setMessages] = useState([])
    const [status, setStatus] = useState('disconnected')
    const [isProcessing, setIsProcessing] = useState(false)
    const [confirmationRequest, setConfirmationRequest] = useState(null)
    const wsRef = useRef(null)
    const reconnectTimeoutRef = useRef(null)
    const pingIntervalRef = useRef(null)

    useEffect(() => {
        // Try to get backend URL from environment or build a sensible default
        const backendUrl = import.meta.env.VITE_BACKEND_URL || window.location.origin

        // Determine WebSocket protocol based on location protocol
        const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'

        // Build WebSocket URL
        // If backend is same origin, use /ws; otherwise use full URL
        let wsUrlFinal = `${proto}//${window.location.host}/ws`
        if (backendUrl !== window.location.origin) {
            // Backend is on a different host/port  
            const backendHost = new URL(backendUrl).host
            wsUrlFinal = `${proto}//${backendHost}/ws`
        }

        setWsUrl(wsUrlFinal)
    }, [])

    const connect = useCallback(() => {
        if (!wsUrl) return
        if (wsRef.current?.readyState === WebSocket.OPEN) return

        console.log('Connecting to WebSocket:', wsUrl)
        const ws = new WebSocket(wsUrl)
        wsRef.current = ws

        ws.onopen = () => {
            console.log('WebSocket connected')
            setStatus('connected')
            // Clear any pending reconnect
            if (reconnectTimeoutRef.current) {
                clearTimeout(reconnectTimeoutRef.current)
                reconnectTimeoutRef.current = null
            }
            // Start ping interval to keep connection alive
            pingIntervalRef.current = setInterval(() => {
                if (ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ type: 'ping' }))
                }
            }, 30000) // Ping every 30 seconds
        }

        ws.onclose = (event) => {
            console.log('WebSocket closed:', event.code, event.reason)
            setStatus('disconnected')
            // Don't reset isProcessing here - the backend might still be processing
            // Clear ping interval
            if (pingIntervalRef.current) {
                clearInterval(pingIntervalRef.current)
                pingIntervalRef.current = null
            }
            // Auto-reconnect after 3 seconds (unless it was a clean close)
            if (event.code !== 1000) {
                reconnectTimeoutRef.current = setTimeout(() => {
                    console.log('Attempting to reconnect...')
                    connect()
                }, 3000)
            }
        }

        ws.onerror = (error) => {
            console.error('WebSocket error:', error)
            setStatus('error')
        }

        ws.onmessage = (ev) => {
            try {
                const data = JSON.parse(ev.data)

                // Handle different message types
                if (data.type === 'pong') {
                    // Ignore pong responses
                    return
                }

                if (data.type === 'confirmation_request') {
                    // Show confirmation dialog
                    setConfirmationRequest(data)
                    return
                }

                if (data.type === 'thinking' || data.role === 'thinking') {
                    // Thinking message - agent is still processing
                    setMessages((m) => [...m, data])
                } else if (data.role === 'assistant') {
                    // Final response - agent finished processing
                    setIsProcessing(false)
                    setMessages((m) => [...m, data])
                } else {
                    setMessages((m) => [...m, data])
                }
            } catch (e) {
                console.error('invalid ws message', e)
            }
        }
    }, [wsUrl])

    // Connect when wsUrl is available
    useEffect(() => {
        if (wsUrl) {
            connect()
        }

        return () => {
            if (reconnectTimeoutRef.current) {
                clearTimeout(reconnectTimeoutRef.current)
            }
            if (pingIntervalRef.current) {
                clearInterval(pingIntervalRef.current)
            }
            // Don't close WebSocket on cleanup - we want it to persist
        }
    }, [wsUrl, connect])

    const sendMessage = useCallback((content) => {
        if (!content.trim() || isProcessing) return false
        if (wsRef.current?.readyState !== WebSocket.OPEN) {
            console.error('WebSocket not connected')
            return false
        }

        const msg = { role: 'user', content }
        setMessages((m) => [...m, msg])
        setIsProcessing(true)
        wsRef.current.send(JSON.stringify({ content }))
        return true
    }, [isProcessing])

    const sendConfirmationResponse = useCallback((requestId, confirmed) => {
        if (wsRef.current?.readyState !== WebSocket.OPEN) {
            console.error('WebSocket not connected')
            return
        }

        // Send confirmation response
        wsRef.current.send(JSON.stringify({
            type: 'confirmation_response',
            id: requestId,
            confirmed: confirmed
        }))

        // Add a message showing the user's decision
        setMessages((m) => [...m, {
            role: 'user',
            content: confirmed ? '✓ Confirmed' : '✗ Rejected'
        }])

        // Clear the confirmation request
        setConfirmationRequest(null)
    }, [])

    const stopProcessing = useCallback(() => {
        if (wsRef.current?.readyState !== WebSocket.OPEN) {
            console.error('WebSocket not connected')
            return
        }

        wsRef.current.send(JSON.stringify({ type: 'stop' }))
        setIsProcessing(false)
        setMessages((m) => [...m, {
            role: 'assistant',
            content: 'Stopped by user'
        }])
    }, [])

    return (
        <div className="app-root">
            <Sidebar activePage={activePage} setActivePage={setActivePage} />
            <main className="app-main">
                {/* Use CSS to show/hide instead of conditional rendering to preserve state */}
                <div style={{ display: activePage === 'chat' ? 'flex' : 'none', flexDirection: 'column', height: '100%' }}>
                    <Chat
                        messages={messages}
                        status={status}
                        isProcessing={isProcessing}
                        onSendMessage={sendMessage}
                        onStopProcessing={stopProcessing}
                        confirmationRequest={confirmationRequest}
                        onConfirmationResponse={sendConfirmationResponse}
                    />
                </div>
                <div style={{ display: activePage === 'dataset' ? 'block' : 'none', height: '100%' }}>
                    <Dataset />
                </div>
                <div style={{ display: activePage === 'models' ? 'block' : 'none', height: '100%', overflow: 'auto' }}>
                    <Models />
                </div>
            </main>
        </div>
    )
}

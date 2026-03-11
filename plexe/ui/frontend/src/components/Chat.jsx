import React, { useEffect, useRef, useState, useMemo } from 'react'

// ===================== Training Progress Component =====================

const TrainingProgress = ({ steps }) => {
    // Aggregate all training_progress events to build the full picture
    const progressState = useMemo(() => {
        let phase = 'preparing'
        let currentEpoch = 0
        let totalEpochs = 0
        let currentLoss = null
        let batchProgress = null
        let currentBatch = 0
        let totalBatches = 0
        let bestMetricName = null
        let bestMetricValue = null
        let latestMetrics = {}
        let epochHistory = []
        let lastMessage = ''
        let isCompleted = false

        const seenEpochs = new Set()

        for (const step of steps) {
            const p = step.progress
            if (!p) continue

            if (p.phase) phase = p.phase
            if (p.message) lastMessage = p.message
            if (p.total_epochs) totalEpochs = p.total_epochs
            if (p.best_metric_name) bestMetricName = p.best_metric_name
            if (p.best_metric_value !== undefined && p.best_metric_value !== null) bestMetricValue = p.best_metric_value
            if (p.metrics && Object.keys(p.metrics).length > 0) latestMetrics = p.metrics
            if (p.batch_progress !== undefined) batchProgress = p.batch_progress
            if (p.current_batch) currentBatch = p.current_batch
            if (p.total_batches) totalBatches = p.total_batches

            // Build epoch history from epoch-level events
            if (p.current_epoch && p.loss !== undefined && p.loss !== null && p.phase === 'training') {
                currentEpoch = p.current_epoch
                currentLoss = p.loss

                if (!seenEpochs.has(p.current_epoch)) {
                    seenEpochs.add(p.current_epoch)
                    epochHistory.push({
                        epoch: p.current_epoch,
                        loss: p.loss,
                        metrics: p.metrics || {},
                        isBest: p.is_best || false,
                    })
                }
            } else if (p.current_epoch) {
                currentEpoch = p.current_epoch
            }

            if (p.is_best && epochHistory.length > 0) {
                epochHistory[epochHistory.length - 1].isBest = true
            }

            if (p.phase === 'completed') isCompleted = true

            // Also accept epoch_history from backend if provided
            if (p.epoch_history && p.epoch_history.length > epochHistory.length) {
                epochHistory = p.epoch_history.map(e => ({
                    epoch: e.epoch,
                    loss: e.loss,
                    metrics: e.metrics || {},
                    isBest: e.is_best || false,
                }))
            }
        }

        return {
            phase, currentEpoch, totalEpochs, currentLoss, batchProgress,
            currentBatch, totalBatches, bestMetricName, bestMetricValue,
            latestMetrics, epochHistory, lastMessage, isCompleted
        }
    }, [steps])

    const {
        phase, currentEpoch, totalEpochs, currentLoss, batchProgress,
        currentBatch, totalBatches, bestMetricName, bestMetricValue,
        latestMetrics, epochHistory, lastMessage, isCompleted
    } = progressState

    const epochPercent = totalEpochs > 0 ? Math.round((currentEpoch / totalEpochs) * 100) : 0

    // Mini loss chart using SVG
    const LossChart = ({ history }) => {
        if (!history || history.length < 2) return null

        const width = 280
        const height = 80
        const padding = { top: 8, right: 8, bottom: 20, left: 35 }
        const chartW = width - padding.left - padding.right
        const chartH = height - padding.top - padding.bottom

        const losses = history.map(h => h.loss)
        const maxLoss = Math.max(...losses)
        const minLoss = Math.min(...losses)
        const lossRange = maxLoss - minLoss || 1

        const points = history.map((h, i) => {
            const x = padding.left + (i / (history.length - 1)) * chartW
            const y = padding.top + (1 - (h.loss - minLoss) / lossRange) * chartH
            return { x, y, ...h }
        })

        const linePath = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ')

        // Area path (fill under line)
        const areaPath = linePath +
            ` L ${points[points.length - 1].x} ${padding.top + chartH}` +
            ` L ${points[0].x} ${padding.top + chartH} Z`

        return (
            <div className="training-loss-chart">
                <div className="chart-label">Loss</div>
                <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
                    {/* Grid lines */}
                    <line x1={padding.left} y1={padding.top} x2={padding.left} y2={padding.top + chartH}
                          stroke="#e5e7eb" strokeWidth="1" />
                    <line x1={padding.left} y1={padding.top + chartH} x2={padding.left + chartW} y2={padding.top + chartH}
                          stroke="#e5e7eb" strokeWidth="1" />

                    {/* Y-axis labels */}
                    <text x={padding.left - 4} y={padding.top + 4} textAnchor="end"
                          className="chart-axis-label">{maxLoss.toFixed(3)}</text>
                    <text x={padding.left - 4} y={padding.top + chartH} textAnchor="end"
                          className="chart-axis-label">{minLoss.toFixed(3)}</text>

                    {/* X-axis labels */}
                    <text x={padding.left} y={height - 2} textAnchor="middle"
                          className="chart-axis-label">1</text>
                    <text x={padding.left + chartW} y={height - 2} textAnchor="middle"
                          className="chart-axis-label">{history.length}</text>

                    {/* Area fill */}
                    <path d={areaPath} fill="rgba(37, 99, 235, 0.08)" />

                    {/* Loss line */}
                    <path d={linePath} fill="none" stroke="#2563eb" strokeWidth="2" strokeLinejoin="round" />

                    {/* Best epoch markers */}
                    {points.filter(p => p.isBest).map((p, i) => (
                        <circle key={i} cx={p.x} cy={p.y} r="3" fill="#10b981" stroke="#fff" strokeWidth="1" />
                    ))}

                    {/* Current point */}
                    {points.length > 0 && (
                        <circle cx={points[points.length - 1].x} cy={points[points.length - 1].y}
                                r="3" fill="#2563eb" stroke="#fff" strokeWidth="1" />
                    )}
                </svg>
            </div>
        )
    }

    const phaseLabels = {
        preparing: 'Preparing Data',
        embedding: 'Embedding Data',
        training: 'Training Model',
        evaluating: 'Evaluating Model',
        completed: 'Training Complete',
    }

    const phaseIcons = {
        preparing: '\u{1F4E6}',
        embedding: '\u{1F9E9}',
        training: '\u{1F3CB}',
        evaluating: '\u{1F4CA}',
        completed: '\u2705',
    }

    return (
        <div className={`training-progress-card ${isCompleted ? 'completed' : ''}`}>
            {/* Phase header */}
            <div className="training-progress-header">
                <div className="training-phase-info">
                    <span className="training-phase-icon">{phaseIcons[phase] || '\u{1F504}'}</span>
                    <span className="training-phase-label">{phaseLabels[phase] || phase}</span>
                </div>
                {totalEpochs > 0 && (
                    <span className="training-epoch-counter">
                        Epoch {currentEpoch}/{totalEpochs}
                    </span>
                )}
            </div>

            {/* Epoch progress bar */}
            {totalEpochs > 0 && (
                <div className="training-progress-bar-container">
                    <div className="training-progress-bar">
                        <div
                            className={`training-progress-fill ${isCompleted ? 'completed' : ''}`}
                            style={{ width: `${epochPercent}%` }}
                        />
                    </div>
                    <span className="training-progress-percent">{epochPercent}%</span>
                </div>
            )}

            {/* Batch-level progress (within epoch) */}
            {batchProgress !== null && batchProgress < 100 && phase === 'training' && !isCompleted && (
                <div className="training-batch-progress">
                    <div className="training-batch-bar">
                        <div className="training-batch-fill" style={{ width: `${batchProgress}%` }} />
                    </div>
                    <span className="training-batch-label">
                        Batch {currentBatch}/{totalBatches}
                    </span>
                </div>
            )}

            {/* Stats row */}
            <div className="training-stats-row">
                {currentLoss !== null && (
                    <div className="training-stat">
                        <span className="training-stat-label">Loss</span>
                        <span className="training-stat-value">{currentLoss.toFixed(4)}</span>
                    </div>
                )}
                {bestMetricName && bestMetricValue !== null && (
                    <div className="training-stat best">
                        <span className="training-stat-label">Best {bestMetricName}</span>
                        <span className="training-stat-value">{bestMetricValue.toFixed(6)}</span>
                    </div>
                )}
                {Object.entries(latestMetrics).slice(0, 3).map(([key, val]) => (
                    <div className="training-stat" key={key}>
                        <span className="training-stat-label">{key}</span>
                        <span className="training-stat-value">{typeof val === 'number' ? val.toFixed(4) : val}</span>
                    </div>
                ))}
            </div>

            {/* Loss chart */}
            <LossChart history={epochHistory} />

            {/* Status message */}
            {lastMessage && (
                <div className="training-status-message">{lastMessage}</div>
            )}
        </div>
    )
}

// ===================== Original Chat Components =====================

// Icons for different event types
const EventIcon = ({ eventType }) => {
    const icons = {
        agent_start: '🚀',
        thinking: '💭',
        tool_call: '🔧',
        tool_result: '✅',
        agent_end: '🎯'
    }
    return <span className="event-icon">{icons[eventType] || '•'}</span>
}

// Specific color mapping per agent for consistent identity visuals
const getAgentTheme = (name = '') => {
    const lowerName = name.toLowerCase()

    // ErrorHandler agent - red theme to indicate error state
    if (lowerName.includes('errorhandler') || lowerName.includes('error_handler')) {
        return {
            accent: '#dc2626',
            tint: 'rgba(220, 38, 38, 0.08)',
            wash: 'rgba(220, 38, 38, 0.04)',
            border: 'rgba(220, 38, 38, 0.3)'
        }
    }

    // Specific colors for known agents
    if (lowerName.includes('orchestrator')) {
        return {
            accent: '#2563eb',
            tint: 'rgba(37, 99, 235, 0.08)',
            wash: 'rgba(37, 99, 235, 0.04)',
            border: 'rgba(37, 99, 235, 0.3)'
        }
    }

    if (lowerName.includes('conversational')) {
        return {
            accent: '#10b981',
            tint: 'rgba(16, 185, 129, 0.08)',
            wash: 'rgba(16, 185, 129, 0.04)',
            border: 'rgba(16, 185, 129, 0.3)'
        }
    }

    if (lowerName.includes('eda')) {
        return {
            accent: '#7c3aed',
            tint: 'rgba(124, 58, 237, 0.08)',
            wash: 'rgba(124, 58, 237, 0.04)',
            border: 'rgba(124, 58, 237, 0.3)'
        }
    }

    if (lowerName.includes('taskbuilder') || lowerName.includes('task_builder')) {
        return {
            accent: '#2c3e50',
            tint: 'rgba(44, 62, 80, 0.08)',
            wash: 'rgba(44, 62, 80, 0.04)',
            border: 'rgba(44, 62, 80, 0.3)'
        }
    }

    if (lowerName.includes('operation')) {
        return {
            accent: '#d97706',
            tint: 'rgba(217, 119, 6, 0.08)',
            wash: 'rgba(217, 119, 6, 0.04)',
            border: 'rgba(217, 119, 6, 0.3)'
        }
    }

    // Fallback to deterministic color for other agents
    let hash = 0
    for (let i = 0; i < name.length; i += 1) {
        hash = name.charCodeAt(i) + ((hash << 5) - hash)
    }
    const hue = Math.abs(hash) % 360
    return {
        accent: `hsl(${hue}, 70%, 50%)`,
        tint: `hsla(${hue}, 70%, 50%, 0.08)`,
        wash: `hsla(${hue}, 70%, 50%, 0.04)`,
        border: `hsla(${hue}, 70%, 50%, 0.3)`
    }
}

// Group consecutive messages from the same agent
const groupMessagesByAgent = (messages) => {
    const groups = []
    let currentGroup = null

    messages.forEach((msg) => {
        if (msg.role === 'thinking') {
            const agentName = msg.agent_name || 'Agent'

            // Start a new group if agent changes or no current group
            if (!currentGroup || currentGroup.agent !== agentName) {
                if (currentGroup) {
                    groups.push(currentGroup)
                }
                currentGroup = {
                    agent: agentName,
                    steps: [],
                    startStep: msg.step_number,
                    endStep: msg.step_number
                }
            }

            // Add step to current group
            currentGroup.steps.push(msg)
            currentGroup.endStep = msg.step_number
        } else {
            // Non-thinking messages close the current group
            if (currentGroup) {
                groups.push(currentGroup)
                currentGroup = null
            }
            groups.push({ type: 'message', message: msg })
        }
    })

    // Add the last group if exists
    if (currentGroup) {
        groups.push(currentGroup)
    }

    return groups
}

// Render a single event within an agent group
const EventItem = ({ step }) => {
    const [isExpanded, setIsExpanded] = useState(false)
    const eventType = step.event_type || 'thinking'

    // Smarter error detection: only mark as error if it's actually an error response
    // Check for status: "error" in JSON or explicit error event types
    const hasError = (() => {
        if (!step.message) return false

        // Check if this is a tool result with error status
        if (step.event_type === 'tool_result') {
            try {
                // Try to extract and parse JSON from the message
                // Handle both raw JSON and "content='...'" format
                let jsonStr = step.message
                const contentMatch = step.message.match(/content='({.*})'/)
                if (contentMatch) {
                    jsonStr = contentMatch[1]
                }

                // Parse the JSON and check status field
                const parsed = JSON.parse(jsonStr)
                if (parsed.status === 'error') {
                    return true
                }
                // If status is "success" or any other value, it's not an error
                if (parsed.status) {
                    return false
                }
            } catch (e) {
                // Not valid JSON, fall through to other checks
            }
        }

        // Check if message starts with explicit error indicators
        if (step.message.match(/^(Error:|ERROR:|Failed:|FAILED:|Exception:)/i)) {
            return true
        }

        // Don't mark as error just because it contains the word "error" in normal text
        return false
    })()

    // Check if this is a tool result with content
    const isToolResult = step.message && step.message.startsWith('Tool result:')
    // Check if this is a reasoning/thinking message
    const isReasoning = step.message && (
        step.message.startsWith('💭') ||
        step.message.startsWith('💡') ||
        step.message.includes('Reasoning') ||
        step.message.includes('Analysis')
    )
    const MAX_LINES = 5

    const renderMessage = () => {
        if (!step.message) return null

        // For tool results, show first 5 lines with expand option
        if (isToolResult) {
            const content = step.message.replace(/^Tool result:\n?/, '')
            const lines = content.split('\n')
            const hasMoreLines = lines.length > MAX_LINES
            const displayLines = isExpanded ? lines : lines.slice(0, MAX_LINES)

            return (
                <div className="tool-result-content">
                    <pre className="tool-result-text">{displayLines.join('\n')}</pre>
                    {hasMoreLines && (
                        <button
                            className="expand-btn"
                            onClick={() => setIsExpanded(!isExpanded)}
                        >
                            {isExpanded ? '▲ Collapse' : `▼ Expand (${lines.length - MAX_LINES} more lines)`}
                        </button>
                    )}
                </div>
            )
        }

        // For reasoning messages, format nicely with expand option
        if (isReasoning) {
            const lines = step.message.split('\n')
            const hasMoreLines = lines.length > MAX_LINES + 2
            const displayLines = isExpanded ? lines : lines.slice(0, MAX_LINES + 2)

            return (
                <div className="reasoning-content">
                    <div className="reasoning-text">{displayLines.join('\n')}</div>
                    {hasMoreLines && (
                        <button
                            className="expand-btn"
                            onClick={() => setIsExpanded(!isExpanded)}
                        >
                            {isExpanded ? '▲ Collapse' : `▼ Show more (${lines.length - MAX_LINES - 2} more lines)`}
                        </button>
                    )}
                </div>
            )
        }

        if (hasError) {
            // For actual errors, render with prominent styling and bold text
            return (
                <div className="error-message-box">
                    <div className="error-icon">⚠️</div>
                    <div className="error-content">
                        <strong style={{ fontWeight: 700 }}>{step.message}</strong>
                    </div>
                </div>
            )
        }

        return step.message
    }

    // Determine the icon based on message content
    const getEventIcon = () => {
        if (step.message) {
            if (step.message.startsWith('💭')) return '💭'
            if (step.message.startsWith('💡')) return '💡'
        }
        const icons = {
            agent_start: '🚀',
            thinking: '💭',
            tool_call: '🔧',
            tool_result: '✅',
            agent_end: '🎯'
        }
        return icons[eventType] || '•'
    }

    // Determine the label based on message content
    const getEventLabel = () => {
        if (step.message) {
            // Extract model info if present (e.g., "[openai/gpt-4o]")
            const modelMatch = step.message.match(/\[([\w/-]+)\]/)
            const modelInfo = modelMatch ? ` (${modelMatch[1]})` : ''

            if (step.message.startsWith('💭 Reasoning')) return `Reasoning${modelInfo}`
            if (step.message.startsWith('💡 Analysis')) return `Analysis${modelInfo}`
        }

        if (eventType === 'agent_start') return 'Starting'
        if (eventType === 'thinking') return 'Reasoning'
        if (eventType === 'tool_call') return `Tool: ${step.tool_name || 'Unknown'}`
        if (eventType === 'tool_result') return 'Result'
        if (eventType === 'agent_end') return 'Completed'
        return eventType
    }

    return (
        <div className={`event-item ${eventType} ${hasError ? 'has-error' : ''} ${isReasoning ? 'reasoning' : ''}`}>
            <div className="event-header">
                <span className="event-icon">{getEventIcon()}</span>
                <span className="event-label">{getEventLabel()}</span>
                {step.timestamp && (
                    <span className="event-time">{step.timestamp}</span>
                )}
            </div>
            <div className="event-content">
                {renderMessage()}
            </div>
        </div>
    )
}

// Render an agent group (multiple steps from same agent)
const AgentGroup = ({ group }) => {
    const theme = getAgentTheme(group.agent)

    // Check for actual errors: status:"error" in JSON content or explicit error messages
    const hasError = group.steps.some(s => {
        if (!s.message) return false

        // Check for JSON with status: "error"
        try {
            // Handle content='...' format
            const contentMatch = s.message.match(/content='({.*})'/)
            const jsonStr = contentMatch ? contentMatch[1] : s.message
            const parsed = JSON.parse(jsonStr)
            if (parsed.status === 'error') {
                return true
            }
        } catch (e) {
            // Not valid JSON, continue to other checks
        }

        // Check for explicit error message prefixes
        if (s.message.match(/^(Error:|ERROR:|Failed:|FAILED:|Exception:)/i)) {
            return true
        }

        return false
    })

    // Extract model_id from agent_start step if available
    const modelId = group.steps.find(s => s.event_type === 'agent_start')?.model_id || ''

    // Calculate total token usage for this agent
    const totalTokens = group.steps.reduce((sum, step) => {
        if (step.token_usage && step.token_usage.total_tokens) {
            return sum + step.token_usage.total_tokens
        }
        return sum
    }, 0)

    const stepRange = group.startStep === group.endStep
        ? `Step ${group.startStep}`
        : `Steps ${group.startStep}-${group.endStep}`

    return (
        <div className="agent-group">
            <div
                className={`agent-group-bubble ${hasError ? 'has-error' : ''}`}
                style={{
                    borderLeftColor: hasError ? '#ef4444' : theme.accent,
                    borderColor: hasError ? 'rgba(239, 68, 68, 0.3)' : theme.border
                }}
            >
                <div className="agent-group-header" style={{
                    background: hasError
                        ? 'linear-gradient(135deg, rgba(254, 242, 242, 0.3) 0%, rgba(254, 226, 226, 0.2) 100%)'
                        : `linear-gradient(135deg, ${theme.wash} 0%, ${theme.tint} 100%)`
                }}>
                    <div className="agent-header-left">
                        <span className="agent-name" style={{ color: theme.accent }}>
                            {group.agent}
                        </span>
                        {modelId && (
                            <span className="model-badge" title={modelId}>
                                {modelId.split('/').pop()}
                            </span>
                        )}
                        {totalTokens > 0 && (
                            <span className="token-badge" title={`Total tokens used: ${totalTokens.toLocaleString()}`}>
                                🪙 {totalTokens.toLocaleString()}
                            </span>
                        )}
                    </div>
                    <span className="step-range" style={{
                        color: theme.accent,
                        opacity: 0.7
                    }}>
                        {stepRange}
                    </span>
                </div>
                <div className="agent-group-content">
                    {(() => {
                        // Separate training_progress steps from other steps
                        const trainingSteps = group.steps.filter(s => s.event_type === 'training_progress')
                        const otherSteps = group.steps.filter(s => s.event_type !== 'training_progress')

                        return (
                            <>
                                {otherSteps.map((step, idx) => (
                                    <EventItem key={`evt-${idx}`} step={step} />
                                ))}
                                {trainingSteps.length > 0 && (
                                    <TrainingProgress steps={trainingSteps} />
                                )}
                            </>
                        )
                    })()}
                </div>
            </div>
        </div>
    )
}

function Message({ msg }) {
    const isUser = msg.role === 'user'
    const isError = msg.role === 'error'

    return (
        <div className={`message ${isUser ? 'user' : isError ? 'error' : 'assistant'}`}>
            <div className={`bubble ${isError ? 'error' : ''}`}>{msg.content}</div>
        </div>
    )
}

function ThinkingIndicator() {
    return (
        <div className="message assistant">
            <div className="bubble thinking-indicator">
                <div className="dot-typing">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        </div>
    )
}

function ConfirmationDialog({ request, onConfirm, onReject }) {
    const [isExpanded, setIsExpanded] = useState(false)

    const renderContent = () => {
        const { content, content_type } = request

        // Truncate content if too long and not expanded
        const maxPreviewLength = 500
        const shouldTruncate = content.length > maxPreviewLength && !isExpanded
        const displayContent = shouldTruncate
            ? content.substring(0, maxPreviewLength) + '...'
            : content

        if (content_type === 'code') {
            return (
                <pre className="confirmation-code">
                    <code>{displayContent}</code>
                </pre>
            )
        } else if (content_type === 'json') {
            try {
                const parsed = JSON.parse(content)
                return (
                    <pre className="confirmation-json">
                        {isExpanded ? JSON.stringify(parsed, null, 2) : JSON.stringify(parsed, null, 2).substring(0, maxPreviewLength) + (JSON.stringify(parsed, null, 2).length > maxPreviewLength ? '...' : '')}
                    </pre>
                )
            } catch {
                return <pre className="confirmation-text">{displayContent}</pre>
            }
        } else if (content_type === 'markdown') {
            // Simple markdown rendering (just preserve formatting)
            return <div className="confirmation-markdown"><pre>{displayContent}</pre></div>
        }
        return <div className="confirmation-text"><pre>{displayContent}</pre></div>
    }

    return (
        <div className="confirmation-dialog-overlay">
            <div className="confirmation-dialog">
                <div className="confirmation-header">
                    <h3>{request.title}</h3>
                </div>
                <div className="confirmation-body">
                    {renderContent()}
                    {request.content.length > 500 && (
                        <button
                            className="expand-toggle"
                            onClick={() => setIsExpanded(!isExpanded)}
                        >
                            {isExpanded ? ' Compact' : ' See more'}
                        </button>
                    )}
                </div>
                <div className="confirmation-footer">
                    <button className="btn-reject" onClick={onReject}>
                        Reject
                    </button>
                    <button className="btn-confirm" onClick={onConfirm}>
                        Confirm
                    </button>
                </div>
            </div>
        </div>
    )
}

export default function Chat({ messages, status, isProcessing, onSendMessage, onStopProcessing, confirmationRequest, onConfirmationResponse }) {
    const [input, setInput] = useState('')
    const messagesEndRef = useRef(null)
    const textareaRef = useRef(null)

    // Auto-scroll to bottom when new messages arrive
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [messages, isProcessing])

    // Auto-resize textarea to fit content
    useEffect(() => {
        const el = textareaRef.current
        if (!el) return
        el.style.height = 'auto'
        el.style.height = Math.min(el.scrollHeight, 160) + 'px'
    }, [input])

    const send = () => {
        if (onSendMessage(input)) {
            setInput('')
        }
    }

    const getStatusClass = () => {
        if (isProcessing) return 'processing'
        return status
    }

    const getStatusText = () => {
        if (isProcessing) return 'Processing...'
        if (status === 'connected') return 'Connected'
        if (status === 'disconnected') return 'Disconnected - Reconnecting...'
        if (status === 'error') return 'Connection Error'
        return status
    }

    // Group messages by agent
    const messageGroups = groupMessagesByAgent(messages)

    return (
        <div className="chat-root">
            {confirmationRequest && (
                <ConfirmationDialog
                    request={confirmationRequest}
                    onConfirm={() => onConfirmationResponse(confirmationRequest.id, true)}
                    onReject={() => onConfirmationResponse(confirmationRequest.id, false)}
                />
            )}
            <div className={`status ${getStatusClass()}`}>{getStatusText()}</div>
            <div className="messages">
                {messageGroups.map((group, i) => {
                    if (group.type === 'message') {
                        return <Message key={i} msg={group.message} />
                    } else {
                        return <AgentGroup key={i} group={group} />
                    }
                })}
                {isProcessing && <ThinkingIndicator />}
                <div ref={messagesEndRef} />
            </div>
            <div className="composer">
                <textarea
                    ref={textareaRef}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault()
                            send()
                        }
                    }}
                    placeholder="Type your message... (Shift+Enter for new line)"
                    disabled={isProcessing || status !== 'connected'}
                    rows={1}
                />
                {isProcessing ? (
                    <button onClick={onStopProcessing} className="stop-btn">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                            <rect x="4" y="4" width="16" height="16" rx="2" />
                        </svg>
                        Stop
                    </button>
                ) : (
                    <button onClick={send} disabled={status !== 'connected'} className="send-btn">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <line x1="22" y1="2" x2="11" y2="13" />
                            <polygon points="22 2 15 22 11 13 2 9 22 2" />
                        </svg>
                    </button>
                )}
            </div>
        </div>
    )
}

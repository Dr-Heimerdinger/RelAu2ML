import React, { useEffect, useRef, useState, useMemo } from 'react'
import {
    Rocket, MessageCircle, Wrench, CheckCircle, Target,
    Package, Puzzle, Dumbbell, BarChart3, CheckCircle2,
    Coins, AlertTriangle, Lightbulb, Brain, MessageSquare,
    ClipboardList, Play, AlertOctagon, ChevronDown, ChevronUp,
    Award, Send, Square, Database, Share2, Bot,
    CircleDot
} from 'lucide-react'
import { formatAgentName, getAgentIconKey, AGENT_ICON_MAP } from '../utils'

// ===================== Agent Icon Resolver =====================

const AGENT_ICON_COMPONENTS = {
    Brain, MessageSquare, BarChart3, ClipboardList, Play,
    AlertOctagon, Database, Share2
}

const AgentIcon = ({ agentName, size = 16, color }) => {
    const key = getAgentIconKey(agentName)
    if (key && AGENT_ICON_MAP[key]) {
        const IconComp = AGENT_ICON_COMPONENTS[AGENT_ICON_MAP[key]]
        if (IconComp) return <IconComp size={size} color={color} />
    }
    return <Bot size={size} color={color} />
}

// ===================== Pipeline Progress Component =====================

const PIPELINE_STAGES = [
    { key: 'conversational', label: 'Conversation', Icon: MessageSquare },
    { key: 'eda', label: 'EDA', Icon: BarChart3 },
    { key: 'taskbuilder', label: 'Task Building', Icon: ClipboardList },
    { key: 'training', label: 'Training', Icon: Dumbbell },
    { key: 'evaluation', label: 'Evaluation', Icon: Award },
    { key: 'complete', label: 'Complete', Icon: CheckCircle2 },
]

const PipelineProgress = ({ messages, cumulativeTokens }) => {
    const [collapsed, setCollapsed] = useState(false)

    const stageStatus = useMemo(() => {
        const completed = new Set()
        let current = null

        for (const msg of messages) {
            if (msg.role !== 'thinking') continue
            const name = (msg.agent_name || '').toLowerCase()

            let stageKey = null
            if (name.includes('conversational')) stageKey = 'conversational'
            else if (name.includes('eda')) stageKey = 'eda'
            else if (name.includes('taskbuilder') || name.includes('task_builder')) stageKey = 'taskbuilder'
            else if (name.includes('operation')) stageKey = 'training'

            if (stageKey) {
                // Mark all previous stages as completed
                if (current && current !== stageKey) {
                    completed.add(current)
                }
                current = stageKey
            }

            // Check for training completion
            if (msg.event_type === 'training_progress' && msg.progress?.phase === 'completed') {
                completed.add('training')
                current = 'evaluation'
            }
            if (msg.event_type === 'training_progress' && msg.progress?.phase === 'evaluating') {
                completed.add('training')
                current = 'evaluation'
            }
        }

        const stageOrder = PIPELINE_STAGES.map(s => s.key)
        const currentIdx = current ? stageOrder.indexOf(current) : -1

        const statuses = {}
        for (let i = 0; i < PIPELINE_STAGES.length; i++) {
            const key = PIPELINE_STAGES[i].key
            if (completed.has(key) || (currentIdx > -1 && i < currentIdx)) {
                statuses[key] = 'completed'
            } else if (key === current) {
                statuses[key] = 'active'
            } else {
                statuses[key] = 'pending'
            }
        }
        return statuses
    }, [messages])

    const hasStarted = Object.values(stageStatus).some(s => s !== 'pending')
    if (!hasStarted) return null

    const completedCount = Object.values(stageStatus).filter(s => s === 'completed').length
    const totalStages = PIPELINE_STAGES.length
    const progressPercent = Math.round((completedCount / totalStages) * 100)

    return (
        <div className="pipeline-progress">
            <button className="pipeline-toggle" onClick={() => setCollapsed(!collapsed)}>
                <div className="pipeline-toggle-left">
                    <CircleDot size={14} />
                    <span>Pipeline Progress</span>
                    <span className="pipeline-percent">{progressPercent}%</span>
                    {cumulativeTokens && cumulativeTokens.total > 0 && (
                        <span className="cumulative-token-counter" title="Cumulative tokens used across all agents">
                            <Coins size={12} />
                            <span>{cumulativeTokens.total.toLocaleString()}</span>
                            {cumulativeTokens.budget && (
                                <span className="token-budget-label">/ {cumulativeTokens.budget.toLocaleString()}</span>
                            )}
                        </span>
                    )}
                </div>
                {collapsed ? <ChevronDown size={16} /> : <ChevronUp size={16} />}
            </button>
            {!collapsed && (
                <div className="pipeline-stages">
                    {PIPELINE_STAGES.map((stage, i) => {
                        const status = stageStatus[stage.key]
                        return (
                            <React.Fragment key={stage.key}>
                                <div className={`pipeline-stage ${status}`}>
                                    <div className="pipeline-stage-icon">
                                        {status === 'completed'
                                            ? <CheckCircle2 size={20} />
                                            : <stage.Icon size={20} />
                                        }
                                    </div>
                                    <span className="pipeline-stage-label">{stage.label}</span>
                                </div>
                                {i < PIPELINE_STAGES.length - 1 && (
                                    <div className={`pipeline-connector ${status === 'completed' ? 'completed' : ''}`} />
                                )}
                            </React.Fragment>
                        )
                    })}
                </div>
            )}
        </div>
    )
}

// ===================== Training Progress Component =====================

const TrainingProgress = ({ steps }) => {
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

        const areaPath = linePath +
            ` L ${points[points.length - 1].x} ${padding.top + chartH}` +
            ` L ${points[0].x} ${padding.top + chartH} Z`

        return (
            <div className="training-loss-chart">
                <div className="chart-label">Loss</div>
                <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
                    <line x1={padding.left} y1={padding.top} x2={padding.left} y2={padding.top + chartH}
                          stroke="var(--border)" strokeWidth="1" />
                    <line x1={padding.left} y1={padding.top + chartH} x2={padding.left + chartW} y2={padding.top + chartH}
                          stroke="var(--border)" strokeWidth="1" />

                    <text x={padding.left - 4} y={padding.top + 4} textAnchor="end"
                          className="chart-axis-label">{maxLoss.toFixed(3)}</text>
                    <text x={padding.left - 4} y={padding.top + chartH} textAnchor="end"
                          className="chart-axis-label">{minLoss.toFixed(3)}</text>

                    <text x={padding.left} y={height - 2} textAnchor="middle"
                          className="chart-axis-label">1</text>
                    <text x={padding.left + chartW} y={height - 2} textAnchor="middle"
                          className="chart-axis-label">{history.length}</text>

                    <path d={areaPath} fill="rgba(37, 99, 235, 0.08)" />
                    <path d={linePath} fill="none" stroke="var(--primary)" strokeWidth="2" strokeLinejoin="round" />

                    {points.filter(p => p.isBest).map((p, i) => (
                        <circle key={i} cx={p.x} cy={p.y} r="3" fill="var(--success)" stroke="var(--card)" strokeWidth="1" />
                    ))}

                    {points.length > 0 && (
                        <circle cx={points[points.length - 1].x} cy={points[points.length - 1].y}
                                r="3" fill="var(--primary)" stroke="var(--card)" strokeWidth="1" />
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

    const phaseIconMap = {
        preparing: Package,
        embedding: Puzzle,
        training: Dumbbell,
        evaluating: BarChart3,
        completed: CheckCircle2,
    }

    const PhaseIcon = phaseIconMap[phase] || Package

    return (
        <div className={`training-progress-card ${isCompleted ? 'completed' : ''}`}>
            <div className="training-progress-header">
                <div className="training-phase-info">
                    <span className="training-phase-icon"><PhaseIcon size={18} /></span>
                    <span className="training-phase-label">{phaseLabels[phase] || phase}</span>
                </div>
                {totalEpochs > 0 && (
                    <span className="training-epoch-counter">
                        Epoch {currentEpoch}/{totalEpochs}
                    </span>
                )}
            </div>

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

            <LossChart history={epochHistory} />

            {lastMessage && (
                <div className="training-status-message">{lastMessage}</div>
            )}
        </div>
    )
}

// ===================== Chat Components =====================

// Specific color mapping per agent for consistent identity visuals
const getAgentTheme = (name = '') => {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark'
    const tintOpacity = isDark ? 0.15 : 0.08
    const washOpacity = isDark ? 0.08 : 0.04
    const borderOpacity = isDark ? 0.4 : 0.3
    const lowerName = name.toLowerCase()

    if (lowerName.includes('errorhandler') || lowerName.includes('error_handler')) {
        return {
            accent: '#dc2626',
            tint: `rgba(220, 38, 38, ${tintOpacity})`,
            wash: `rgba(220, 38, 38, ${washOpacity})`,
            border: `rgba(220, 38, 38, ${borderOpacity})`
        }
    }

    if (lowerName.includes('orchestrator')) {
        return {
            accent: isDark ? '#60a5fa' : '#2563eb',
            tint: `rgba(37, 99, 235, ${tintOpacity})`,
            wash: `rgba(37, 99, 235, ${washOpacity})`,
            border: `rgba(37, 99, 235, ${borderOpacity})`
        }
    }

    if (lowerName.includes('conversational')) {
        return {
            accent: isDark ? '#34d399' : '#10b981',
            tint: `rgba(16, 185, 129, ${tintOpacity})`,
            wash: `rgba(16, 185, 129, ${washOpacity})`,
            border: `rgba(16, 185, 129, ${borderOpacity})`
        }
    }

    if (lowerName.includes('eda')) {
        return {
            accent: isDark ? '#a78bfa' : '#7c3aed',
            tint: `rgba(124, 58, 237, ${tintOpacity})`,
            wash: `rgba(124, 58, 237, ${washOpacity})`,
            border: `rgba(124, 58, 237, ${borderOpacity})`
        }
    }

    if (lowerName.includes('taskbuilder') || lowerName.includes('task_builder')) {
        return {
            accent: isDark ? '#94a3b8' : '#2c3e50',
            tint: `rgba(44, 62, 80, ${tintOpacity})`,
            wash: `rgba(44, 62, 80, ${washOpacity})`,
            border: `rgba(44, 62, 80, ${borderOpacity})`
        }
    }

    if (lowerName.includes('operation')) {
        return {
            accent: isDark ? '#fbbf24' : '#d97706',
            tint: `rgba(217, 119, 6, ${tintOpacity})`,
            wash: `rgba(217, 119, 6, ${washOpacity})`,
            border: `rgba(217, 119, 6, ${borderOpacity})`
        }
    }

    let hash = 0
    for (let i = 0; i < name.length; i += 1) {
        hash = name.charCodeAt(i) + ((hash << 5) - hash)
    }
    const hue = Math.abs(hash) % 360
    return {
        accent: `hsl(${hue}, 70%, ${isDark ? '65%' : '50%'})`,
        tint: `hsla(${hue}, 70%, 50%, ${tintOpacity})`,
        wash: `hsla(${hue}, 70%, 50%, ${washOpacity})`,
        border: `hsla(${hue}, 70%, 50%, ${borderOpacity})`
    }
}

// Group consecutive messages from the same agent
const groupMessagesByAgent = (messages) => {
    const groups = []
    let currentGroup = null

    messages.forEach((msg) => {
        // Skip token_update events — they're consumed by the cumulative counter only
        if (msg.event_type === 'token_update') return

        if (msg.role === 'thinking') {
            const agentName = msg.agent_name || 'Agent'

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

            currentGroup.steps.push(msg)
            currentGroup.endStep = msg.step_number
        } else {
            if (currentGroup) {
                groups.push(currentGroup)
                currentGroup = null
            }
            groups.push({ type: 'message', message: msg })
        }
    })

    if (currentGroup) {
        groups.push(currentGroup)
    }

    return groups
}

// Event icon component mapping
const EVENT_ICONS = {
    agent_start: Rocket,
    thinking: MessageCircle,
    tool_call: Wrench,
    tool_result: CheckCircle,
    agent_end: Target,
}

// Render a single event within an agent group
const EventItem = ({ step }) => {
    const [isExpanded, setIsExpanded] = useState(false)
    const eventType = step.event_type || 'thinking'

    const hasError = (() => {
        if (!step.message) return false

        if (step.event_type === 'tool_result') {
            try {
                let jsonStr = step.message
                const contentMatch = step.message.match(/content='({.*})'/)
                if (contentMatch) {
                    jsonStr = contentMatch[1]
                }
                const parsed = JSON.parse(jsonStr)
                if (parsed.status === 'error') {
                    return true
                }
                if (parsed.status) {
                    return false
                }
            } catch (e) {
                // Not valid JSON
            }
        }

        if (step.message.match(/^(Error:|ERROR:|Failed:|FAILED:|Exception:)/i)) {
            return true
        }

        return false
    })()

    const isToolResult = step.message && step.message.startsWith('Tool result:')
    const isReasoning = step.message && (
        step.message.startsWith('\u{1F4AD}') ||
        step.message.startsWith('\u{1F4A1}') ||
        step.message.includes('Reasoning') ||
        step.message.includes('Analysis')
    )
    const MAX_LINES = 5

    const renderMessage = () => {
        if (!step.message) return null

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
                            {isExpanded ? <><ChevronUp size={12} /> Collapse</> : <><ChevronDown size={12} /> Expand ({lines.length - MAX_LINES} more lines)</>}
                        </button>
                    )}
                </div>
            )
        }

        if (isReasoning) {
            // Strip emoji prefix for clean display
            const cleanMessage = step.message.replace(/^[\u{1F300}-\u{1FEFF}\u{2600}-\u{27BF}]\s*/u, '')
            const lines = cleanMessage.split('\n')
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
                            {isExpanded ? <><ChevronUp size={12} /> Collapse</> : <><ChevronDown size={12} /> Show more ({lines.length - MAX_LINES - 2} more lines)</>}
                        </button>
                    )}
                </div>
            )
        }

        if (hasError) {
            return (
                <div className="error-message-box">
                    <div className="error-icon"><AlertTriangle size={18} /></div>
                    <div className="error-content">
                        <strong style={{ fontWeight: 700 }}>{step.message}</strong>
                    </div>
                </div>
            )
        }

        return step.message
    }

    const getEventIcon = () => {
        if (step.message) {
            if (step.message.startsWith('\u{1F4AD}')) return <MessageCircle size={14} />
            if (step.message.startsWith('\u{1F4A1}')) return <Lightbulb size={14} />
        }
        const IconComp = EVENT_ICONS[eventType]
        if (IconComp) return <IconComp size={14} />
        return <CircleDot size={14} />
    }

    const getEventLabel = () => {
        if (step.message) {
            const modelMatch = step.message.match(/\[([\w/-]+)\]/)
            const modelInfo = modelMatch ? ` (${modelMatch[1]})` : ''

            if (step.message.startsWith('\u{1F4AD}') && step.message.includes('Reasoning')) return `Reasoning${modelInfo}`
            if (step.message.startsWith('\u{1F4A1}') && step.message.includes('Analysis')) return `Analysis${modelInfo}`
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

    const hasError = group.steps.some(s => {
        if (!s.message) return false
        try {
            const contentMatch = s.message.match(/content='({.*})'/)
            const jsonStr = contentMatch ? contentMatch[1] : s.message
            const parsed = JSON.parse(jsonStr)
            if (parsed.status === 'error') {
                return true
            }
        } catch (e) {
            // Not valid JSON
        }
        if (s.message.match(/^(Error:|ERROR:|Failed:|FAILED:|Exception:)/i)) {
            return true
        }
        return false
    })

    const modelId = group.steps.find(s => s.event_type === 'agent_start')?.model_id || ''

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
                    borderLeftColor: hasError ? 'var(--error)' : theme.accent,
                    borderColor: hasError ? 'rgba(239, 68, 68, 0.3)' : theme.border
                }}
            >
                <div className="agent-group-header" style={{
                    background: hasError
                        ? 'linear-gradient(135deg, rgba(254, 242, 242, 0.3) 0%, rgba(254, 226, 226, 0.2) 100%)'
                        : `linear-gradient(135deg, ${theme.wash} 0%, ${theme.tint} 100%)`
                }}>
                    <div className="agent-header-left">
                        <span className="agent-icon" style={{ color: theme.accent }}>
                            <AgentIcon agentName={group.agent} size={18} color={theme.accent} />
                        </span>
                        <span className="agent-name" style={{ color: theme.accent }}>
                            {formatAgentName(group.agent)}
                        </span>
                        {modelId && (
                            <span className="model-badge" title={modelId}>
                                {modelId.split('/').pop()}
                            </span>
                        )}
                        {totalTokens > 0 && (
                            <span className="token-badge" title={`Total tokens used: ${totalTokens.toLocaleString()}`}>
                                <Coins size={12} /> {totalTokens.toLocaleString()}
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
            {!isUser && (
                <div className="message-avatar assistant-avatar">
                    <Bot size={16} />
                </div>
            )}
            <div className={`bubble ${isError ? 'error' : ''}`}>{msg.content}</div>
        </div>
    )
}

function ThinkingIndicator() {
    return (
        <div className="message assistant">
            <div className="message-avatar assistant-avatar">
                <Bot size={16} />
            </div>
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
                            {isExpanded ? <><ChevronUp size={12} /> Compact</> : <><ChevronDown size={12} /> See more</>}
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

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [messages, isProcessing])

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

    const messageGroups = groupMessagesByAgent(messages)

    // Extract latest cumulative token usage from token_update events
    const cumulativeTokens = useMemo(() => {
        for (let i = messages.length - 1; i >= 0; i--) {
            if (messages[i].event_type === 'token_update' && messages[i].cumulative_tokens) {
                return messages[i].cumulative_tokens
            }
        }
        return null
    }, [messages])

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
            <PipelineProgress messages={messages} cumulativeTokens={cumulativeTokens} />
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
                        <Square size={14} />
                        Stop
                    </button>
                ) : (
                    <button onClick={send} disabled={status !== 'connected'} className="send-btn">
                        <Send size={18} />
                    </button>
                )}
            </div>
        </div>
    )
}

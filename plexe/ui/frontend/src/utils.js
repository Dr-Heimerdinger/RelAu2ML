/**
 * Format raw agent names to human-readable labels.
 * "TaskBuilderAgent" → "Task Builder"
 * "error_handler" → "Error Handler"
 */
export function formatAgentName(rawName) {
    if (!rawName) return 'Agent'

    let name = rawName
        // Split CamelCase: "TaskBuilder" → "Task Builder"
        .replace(/([a-z])([A-Z])/g, '$1 $2')
        // Replace underscores with spaces
        .replace(/_/g, ' ')
        // Remove trailing "Agent" word
        .replace(/\bAgent\b/gi, '')
        .trim()

    // Title-case each word
    name = name
        .split(' ')
        .filter(Boolean)
        .map(w => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
        .join(' ')

    return name || 'Agent'
}

/**
 * Map of agent identifier keys to their lucide-react icon component names.
 * Used by components to resolve which icon to show for each agent.
 */
export const AGENT_ICON_MAP = {
    orchestrator: 'Brain',
    conversational: 'MessageSquare',
    eda: 'BarChart3',
    taskbuilder: 'ClipboardList',
    task_builder: 'ClipboardList',
    operation: 'Play',
    errorhandler: 'AlertOctagon',
    error_handler: 'AlertOctagon',
    dataset_builder: 'Database',
    datasetbuilder: 'Database',
    gnn_specialist: 'Share2',
    gnnspecialist: 'Share2',
}

/**
 * Resolve an agent name string to its AGENT_ICON_MAP key.
 * Returns the matching key or null for fallback.
 */
export function getAgentIconKey(agentName) {
    if (!agentName) return null
    const lower = agentName.toLowerCase().replace(/[\s_]/g, '')
    // Try exact match first
    for (const key of Object.keys(AGENT_ICON_MAP)) {
        if (lower.includes(key.replace(/_/g, ''))) {
            return key
        }
    }
    return null
}

/**
 * Strip emoji prefixes from backend messages for clean display.
 */
export function stripEmojiPrefix(text) {
    if (!text) return text
    // Remove common emoji prefixes (💭, 💡, etc.)
    return text.replace(/^[\u{1F300}-\u{1FEFF}\u{2600}-\u{27BF}\u{FE00}-\u{FEFF}]\s*/u, '')
}

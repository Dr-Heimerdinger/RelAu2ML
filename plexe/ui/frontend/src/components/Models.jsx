import React, { useState, useEffect, useCallback } from 'react'
import { listModels, deleteModel, renameModel } from '../api/client'
import InferencePanel from './InferencePanel'

function formatBytes(bytes) {
    if (bytes < 1024) return bytes + ' B'
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
}

function formatMetric(value) {
    if (typeof value !== 'number') return '-'
    return value.toFixed(2)
}

function TaskTypeBadge({ type }) {
    const labels = {
        binary_classification: 'Binary Classification',
        multiclass_classification: 'Multiclass',
        regression: 'Regression',
    }
    const classMap = {
        binary_classification: 'task-badge-binary',
        multiclass_classification: 'task-badge-multiclass',
        regression: 'task-badge-regression',
    }
    return (
        <span className={`task-type-badge ${classMap[type] || ''}`}>
            {labels[type] || type || 'Unknown'}
        </span>
    )
}

function ModelCard({ model, onDelete, onRename, onInfer }) {
    const [isRenaming, setIsRenaming] = useState(false)
    const [newName, setNewName] = useState(model.name)
    const [confirmDelete, setConfirmDelete] = useState(false)

    const taskMeta = model.task_meta || {}
    const valMetrics = model.val_metrics || {}
    const testMetrics = model.test_metrics || {}
    const tuneMetric = model.tune_metric || ''

    const handleRename = () => {
        if (newName.trim() && newName !== model.name) {
            onRename(model.id, newName.trim())
        }
        setIsRenaming(false)
    }

    const handleDelete = () => {
        if (confirmDelete) {
            onDelete(model.id)
            setConfirmDelete(false)
        } else {
            setConfirmDelete(true)
            setTimeout(() => setConfirmDelete(false), 3000)
        }
    }

    return (
        <div className="model-card">
            <div className="model-card-header">
                {isRenaming ? (
                    <input
                        className="rename-input"
                        value={newName}
                        onChange={(e) => setNewName(e.target.value)}
                        onBlur={handleRename}
                        onKeyDown={(e) => {
                            if (e.key === 'Enter') handleRename()
                            if (e.key === 'Escape') { setNewName(model.name); setIsRenaming(false) }
                        }}
                        autoFocus
                    />
                ) : (
                    <h4 className="model-card-title">{model.name}</h4>
                )}
                <TaskTypeBadge type={taskMeta.task_type} />
            </div>

            <div className="model-card-info">
                {taskMeta.entity_table && (
                    <div className="info-row">
                        <span className="info-label">Entity</span>
                        <span className="info-value">{taskMeta.entity_table}.{taskMeta.entity_col}</span>
                    </div>
                )}
                {taskMeta.target_col && (
                    <div className="info-row">
                        <span className="info-label">Target</span>
                        <span className="info-value">{taskMeta.target_col}</span>
                    </div>
                )}
                {taskMeta.timedelta_days && (
                    <div className="info-row">
                        <span className="info-label">Window</span>
                        <span className="info-value">{taskMeta.timedelta_days} days</span>
                    </div>
                )}
                <div className="info-row">
                    <span className="info-label">Epochs</span>
                    <span className="info-value">{model.epochs_trained || '-'}</span>
                </div>
                <div className="info-row">
                    <span className="info-label">Size</span>
                    <span className="info-value">{formatBytes(model.model_size)}</span>
                </div>
            </div>

            {(Object.keys(valMetrics).length > 0 || Object.keys(testMetrics).length > 0) && (
                <div className="model-card-metrics">
                    <div className="metrics-header">Metrics</div>
                    <table className="metrics-table">
                        <thead>
                            <tr>
                                <th>Metric</th>
                                <th>Val</th>
                                <th>Test</th>
                            </tr>
                        </thead>
                        <tbody>
                            {Object.keys({ ...valMetrics, ...testMetrics }).map(metric => (
                                <tr key={metric} className={metric === tuneMetric ? 'tune-metric-row' : ''}>
                                    <td>{metric}</td>
                                    <td>{formatMetric(valMetrics[metric])}</td>
                                    <td>{formatMetric(testMetrics[metric])}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}

            <div className="model-card-actions">
                <button className="model-btn model-btn-infer" onClick={() => onInfer(model)}>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
                    </svg>
                    Infer
                </button>
                <button className="model-btn model-btn-icon" onClick={() => { setNewName(model.name); setIsRenaming(true) }} title="Rename">
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
                        <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
                    </svg>
                </button>
                <button
                    className={`model-btn ${confirmDelete ? 'model-btn-icon-danger-confirm' : 'model-btn-icon-danger'}`}
                    onClick={handleDelete}
                    title={confirmDelete ? 'Click again to confirm' : 'Delete'}
                >
                    {confirmDelete ? (
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                            <polyline points="20 6 9 17 4 12" />
                        </svg>
                    ) : (
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <polyline points="3 6 5 6 21 6" />
                            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                        </svg>
                    )}
                </button>
            </div>
        </div>
    )
}

export default function Models() {
    const [models, setModels] = useState([])
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)
    const [selectedModel, setSelectedModel] = useState(null)

    const fetchModels = useCallback(async () => {
        setLoading(true)
        setError(null)
        try {
            const data = await listModels()
            setModels(data.models || [])
        } catch (e) {
            setError(e.message)
        } finally {
            setLoading(false)
        }
    }, [])

    useEffect(() => {
        fetchModels()
    }, [fetchModels])

    const handleDelete = async (modelId) => {
        try {
            await deleteModel(modelId)
            setModels(prev => prev.filter(m => m.id !== modelId))
            if (selectedModel && selectedModel.id === modelId) {
                setSelectedModel(null)
            }
        } catch (e) {
            setError(e.message)
        }
    }

    const handleRename = async (modelId, newName) => {
        try {
            await renameModel(modelId, newName)
            fetchModels()
        } catch (e) {
            setError(e.message)
        }
    }

    const handleInfer = (model) => {
        setSelectedModel(model)
    }

    if (selectedModel) {
        return (
            <div className="models-container">
                <InferencePanel model={selectedModel} onBack={() => setSelectedModel(null)} />
            </div>
        )
    }

    return (
        <div className="models-container">
            <div className="models-header">
                <div>
                    <h2>Trained Models</h2>
                    <p>Manage and run inference on your trained GNN models</p>
                </div>
                <button className="model-btn model-btn-secondary" onClick={fetchModels} disabled={loading}>
                    {loading ? 'Loading...' : 'Refresh'}
                </button>
            </div>

            {error && (
                <div className="models-error">
                    {error}
                    <button onClick={() => setError(null)}>Dismiss</button>
                </div>
            )}

            {loading && models.length === 0 && (
                <div className="models-loading">
                    <div className="loading-spinner"></div>
                    <p>Loading models...</p>
                </div>
            )}

            {!loading && models.length === 0 && (
                <div className="models-empty">
                    <div className="empty-icon">
                        <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                            <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
                            <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
                            <line x1="12" y1="22.08" x2="12" y2="12" />
                        </svg>
                    </div>
                    <h3>No trained models found</h3>
                    <p>Use the Chat to define a prediction task and train a model. Models will appear here once training completes.</p>
                </div>
            )}

            <div className="models-grid">
                {models.map(model => (
                    <ModelCard
                        key={model.id}
                        model={model}
                        onDelete={handleDelete}
                        onRename={handleRename}
                        onInfer={handleInfer}
                    />
                ))}
            </div>
        </div>
    )
}

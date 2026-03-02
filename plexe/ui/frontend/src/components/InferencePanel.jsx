import React, { useState, useEffect, useRef } from 'react'
import { getModelSchema, runInference } from '../api/client'

function formatMetric(value) {
    if (typeof value !== 'number') return '-'
    return value.toFixed(4)
}

function SchemaViewer({ schema }) {
    if (!schema) return null

    const taskMeta = schema.task_meta || {}
    const tables = schema.tables || {}
    const tableNames = Object.keys(tables)

    return (
        <div className="schema-viewer">
            <div className="schema-section">
                <h4>Task Configuration</h4>
                <div className="schema-props">
                    {taskMeta.task_type && (
                        <div className="schema-prop">
                            <span className="prop-key">Task Type</span>
                            <span className="prop-val">{taskMeta.task_type}</span>
                        </div>
                    )}
                    {taskMeta.entity_table && (
                        <div className="schema-prop">
                            <span className="prop-key">Entity Table</span>
                            <span className="prop-val">{taskMeta.entity_table}</span>
                        </div>
                    )}
                    {taskMeta.entity_col && (
                        <div className="schema-prop">
                            <span className="prop-key">Entity Column</span>
                            <span className="prop-val">{taskMeta.entity_col}</span>
                        </div>
                    )}
                    {taskMeta.target_col && (
                        <div className="schema-prop">
                            <span className="prop-key">Target Column</span>
                            <span className="prop-val">{taskMeta.target_col}</span>
                        </div>
                    )}
                    {taskMeta.time_col && (
                        <div className="schema-prop">
                            <span className="prop-key">Time Column</span>
                            <span className="prop-val">{taskMeta.time_col}</span>
                        </div>
                    )}
                    {taskMeta.timedelta_days && (
                        <div className="schema-prop">
                            <span className="prop-key">Prediction Window</span>
                            <span className="prop-val">{taskMeta.timedelta_days} days</span>
                        </div>
                    )}
                </div>
            </div>

            {tableNames.length > 0 && (
                <div className="schema-section">
                    <h4>Database Tables ({tableNames.length})</h4>
                    <div className="schema-tables-list">
                        {tableNames.map(name => (
                            <div key={name} className="schema-table-item">
                                <span className="table-name-label">{name}</span>
                                <span className="table-cols-count">{(tables[name].columns || []).length} columns</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    )
}

function ResultsTable({ results }) {
    if (!results || !results.predictions || results.predictions.length === 0) return null

    const columns = results.columns || Object.keys(results.predictions[0])
    const rows = results.predictions

    const downloadCSV = () => {
        const header = columns.join(',')
        const body = rows.map(row => columns.map(col => {
            const val = row[col]
            if (val === null || val === undefined) return ''
            return typeof val === 'string' && val.includes(',') ? `"${val}"` : val
        }).join(',')).join('\n')
        const csv = header + '\n' + body
        const blob = new Blob([csv], { type: 'text/csv' })
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = 'predictions.csv'
        a.click()
        URL.revokeObjectURL(url)
    }

    return (
        <div className="results-container">
            <div className="results-header">
                <h4>Predictions ({rows.length} rows)</h4>
                <button className="model-btn model-btn-secondary" onClick={downloadCSV}>
                    Download CSV
                </button>
            </div>
            <div className="results-table-wrap">
                <table className="results-table">
                    <thead>
                        <tr>
                            {columns.map(col => <th key={col}>{col}</th>)}
                        </tr>
                    </thead>
                    <tbody>
                        {rows.slice(0, 100).map((row, i) => (
                            <tr key={i}>
                                {columns.map(col => (
                                    <td key={col}>
                                        {typeof row[col] === 'number' ? row[col].toFixed(4) : String(row[col] ?? '')}
                                    </td>
                                ))}
                            </tr>
                        ))}
                    </tbody>
                </table>
                {rows.length > 100 && (
                    <p className="results-truncated">Showing first 100 of {rows.length} rows. Download CSV for full results.</p>
                )}
            </div>
        </div>
    )
}

export default function InferencePanel({ model, onBack }) {
    const [schema, setSchema] = useState(null)
    const [schemaLoading, setSchemaLoading] = useState(true)
    const [file, setFile] = useState(null)
    const [inferring, setInferring] = useState(false)
    const [results, setResults] = useState(null)
    const [error, setError] = useState(null)
    const fileInputRef = useRef(null)

    const taskMeta = model.task_meta || {}
    const valMetrics = model.val_metrics || {}
    const testMetrics = model.test_metrics || {}

    useEffect(() => {
        let cancelled = false
        setSchemaLoading(true)
        getModelSchema(model.id)
            .then(data => { if (!cancelled) setSchema(data) })
            .catch(e => { if (!cancelled) setError(e.message) })
            .finally(() => { if (!cancelled) setSchemaLoading(false) })
        return () => { cancelled = true }
    }, [model.id])

    const handleFileChange = (e) => {
        const f = e.target.files[0]
        if (f) setFile(f)
    }

    const handleDrop = (e) => {
        e.preventDefault()
        const f = e.dataTransfer.files[0]
        if (f && f.name.endsWith('.csv')) setFile(f)
    }

    const handleInfer = async () => {
        if (!file) return
        setInferring(true)
        setError(null)
        setResults(null)
        try {
            const data = await runInference(model.id, file)
            setResults(data)
        } catch (e) {
            setError(e.message)
        } finally {
            setInferring(false)
        }
    }

    const entityCol = taskMeta.entity_col || schema?.entity_col || 'id'
    const timeCol = taskMeta.time_col || schema?.task_meta?.time_col || 'timestamp'

    return (
        <div className="inference-panel">
            <div className="inference-top-bar">
                <button className="model-btn model-btn-secondary" onClick={onBack}>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ marginRight: 4 }}>
                        <polyline points="15 18 9 12 15 6" />
                    </svg>
                    Back to Models
                </button>
                <h3>{model.name}</h3>
            </div>

            <div className="inference-layout">
                <div className="inference-sidebar">
                    <div className="inference-model-summary">
                        <h4>Model Summary</h4>
                        {Object.keys(valMetrics).length > 0 && (
                            <table className="metrics-table compact">
                                <thead>
                                    <tr><th>Metric</th><th>Val</th><th>Test</th></tr>
                                </thead>
                                <tbody>
                                    {Object.keys({ ...valMetrics, ...testMetrics }).map(m => (
                                        <tr key={m}>
                                            <td>{m}</td>
                                            <td>{formatMetric(valMetrics[m])}</td>
                                            <td>{formatMetric(testMetrics[m])}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        )}
                    </div>

                    {schemaLoading ? (
                        <div className="schema-loading">Loading schema...</div>
                    ) : (
                        <SchemaViewer schema={schema} />
                    )}
                </div>

                <div className="inference-main">
                    <div className="inference-instructions">
                        <h4>Data Preparation</h4>
                        <p>
                            Prepare a CSV file with the following required columns:
                        </p>
                        <ol>
                            <li>
                                <strong>{entityCol}</strong> - Entity identifier
                                (e.g., user ID from the <code>{taskMeta.entity_table || 'entity'}</code> table)
                            </li>
                            <li>
                                <strong>{timeCol}</strong> - Prediction timestamp
                                in ISO format (e.g., <code>2024-01-15</code>). The model predicts
                                {taskMeta.timedelta_days
                                    ? ` outcomes within a ${taskMeta.timedelta_days}-day window starting from this date.`
                                    : ' outcomes at this point in time.'}
                            </li>
                        </ol>

                        <div className="csv-example">
                            <div className="csv-example-header">Example CSV</div>
                            <pre className="csv-example-content">{entityCol},{timeCol}\n1,2024-01-15\n2,2024-01-15\n3,2024-02-01</pre>
                        </div>

                        <div className="inference-notes">
                            <h5>Notes</h5>
                            <ul>
                                <li>Entity IDs must exist in the original training database.</li>
                                <li>Timestamps should be after the training data period for meaningful predictions.</li>
                                <li>The model uses the full relational graph structure for predictions, not just the input entities.</li>
                                {taskMeta.task_type === 'binary_classification' && (
                                    <li>Output includes both probability scores and binary predictions (threshold = 0.5).</li>
                                )}
                            </ul>
                        </div>
                    </div>

                    <div className="inference-upload-section">
                        <h4>Upload Inference Data</h4>
                        <div
                            className={`inference-drop-zone ${file ? 'has-file' : ''}`}
                            onClick={() => fileInputRef.current?.click()}
                            onDragOver={(e) => e.preventDefault()}
                            onDrop={handleDrop}
                        >
                            <input
                                ref={fileInputRef}
                                type="file"
                                accept=".csv"
                                onChange={handleFileChange}
                                style={{ display: 'none' }}
                            />
                            {file ? (
                                <div className="selected-file">
                                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#10b981" strokeWidth="2">
                                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                                        <polyline points="14 2 14 8 20 8" />
                                    </svg>
                                    <div>
                                        <span className="file-name">{file.name}</span>
                                        <span className="file-size">{(file.size / 1024).toFixed(1)} KB</span>
                                    </div>
                                    <button className="remove-file-btn" onClick={(e) => { e.stopPropagation(); setFile(null) }}>
                                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                            <line x1="18" y1="6" x2="6" y2="18" />
                                            <line x1="6" y1="6" x2="18" y2="18" />
                                        </svg>
                                    </button>
                                </div>
                            ) : (
                                <div className="drop-zone-content">
                                    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                                        <polyline points="17 8 12 3 7 8" />
                                        <line x1="12" y1="3" x2="12" y2="15" />
                                    </svg>
                                    <p>Drop a CSV file here or click to browse</p>
                                </div>
                            )}
                        </div>

                        <button
                            className="model-btn model-btn-primary inference-run-btn"
                            onClick={handleInfer}
                            disabled={!file || inferring}
                        >
                            {inferring ? (
                                <>
                                    <span className="btn-spinner"></span>
                                    Running Inference...
                                </>
                            ) : (
                                'Run Inference'
                            )}
                        </button>
                    </div>

                    {error && (
                        <div className="inference-error">
                            <strong>Error</strong>
                            <pre>{error}</pre>
                        </div>
                    )}

                    <ResultsTable results={results} />
                </div>
            </div>
        </div>
    )
}

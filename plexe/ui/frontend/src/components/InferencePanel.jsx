import React, { useState, useEffect, useRef } from 'react'
import { getModelSchema, runInference, validateDatabase } from '../api/client'

function formatMetric(value) {
    if (typeof value !== 'number') return '-'
    return value.toFixed(2)
}

/* ─── Expandable Table Detail Card ─── */
function TableDetailCard({ name, info, isEntity }) {
    const [open, setOpen] = useState(isEntity)
    const fkeys = info.foreign_keys || {}
    const hasFkeys = Object.keys(fkeys).length > 0

    return (
        <div className={`tbl-detail-card ${isEntity ? 'tbl-entity' : ''}`}>
            <button className="tbl-detail-toggle" onClick={() => setOpen(!open)}>
                <div className="tbl-detail-left">
                    <svg className={`tbl-chevron ${open ? 'open' : ''}`} width="14" height="14" viewBox="0 0 24 24"
                        fill="none" stroke="currentColor" strokeWidth="2">
                        <polyline points="9 18 15 12 9 6" />
                    </svg>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"
                        style={{ opacity: 0.6 }}>
                        <rect x="3" y="3" width="18" height="18" rx="2" />
                        <line x1="3" y1="9" x2="21" y2="9" />
                        <line x1="9" y1="3" x2="9" y2="21" />
                    </svg>
                    <span className="tbl-detail-name">{name}</span>
                    {isEntity && <span className="tbl-entity-badge">Entity</span>}
                </div>
                <div className="tbl-detail-right">
                    {hasFkeys && <span className="tbl-fk-count">{Object.keys(fkeys).length} FK</span>}
                    <span className="tbl-col-count">{(info.columns || []).length} cols</span>
                    {info.row_count != null && <span className="tbl-row-count">{info.row_count.toLocaleString()} rows</span>}
                </div>
            </button>

            {open && (
                <div className="tbl-detail-body">
                    <table className="tbl-columns-table">
                        <thead>
                            <tr>
                                <th>Column</th>
                                <th>Type</th>
                                <th>Role</th>
                            </tr>
                        </thead>
                        <tbody>
                            {(info.columns || []).map(col => {
                                const isPK = info.primary_key === col
                                const isFK = col in fkeys
                                const isTimeCol = info.time_col === col
                                let role = ''
                                let roleClass = ''
                                if (isPK) { role = 'PK'; roleClass = 'role-pk' }
                                else if (isFK) { role = `FK → ${fkeys[col]}`; roleClass = 'role-fk' }
                                else if (isTimeCol) { role = 'Time'; roleClass = 'role-time' }

                                return (
                                    <tr key={col} className={isPK ? 'row-pk' : ''}>
                                        <td className="col-name-cell">
                                            {isPK && <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor"><path d="M7 17a5.007 5.007 0 0 0 4.898-4H14v2h2v-2h2v3h2V11h-8.1A5.007 5.007 0 0 0 7 7a5 5 0 0 0 0 10zm0-8a3 3 0 1 1 0 6 3 3 0 0 1 0-6z" /></svg>}
                                            {isFK && <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="7" y1="7" x2="17" y2="17" /><polyline points="17 7 17 17 7 17" /></svg>}
                                            <span>{col}</span>
                                        </td>
                                        <td className="col-type-cell">{(info.dtypes || {})[col] || '—'}</td>
                                        <td>{role && <span className={`col-role-badge ${roleClass}`}>{role}</span>}</td>
                                    </tr>
                                )
                            })}
                        </tbody>
                    </table>

                    {info.sample_rows && info.sample_rows.length > 0 && (
                        <div className="tbl-sample">
                            <div className="tbl-sample-header">Sample Data</div>
                            <div className="tbl-sample-scroll">
                                <table className="tbl-sample-table">
                                    <thead>
                                        <tr>{(info.columns || []).map(c => <th key={c}>{c}</th>)}</tr>
                                    </thead>
                                    <tbody>
                                        {info.sample_rows.slice(0, 2).map((row, i) => (
                                            <tr key={i}>
                                                {(info.columns || []).map(c => (
                                                    <td key={c}>{row[c] != null ? String(row[c]).substring(0, 40) : '—'}</td>
                                                ))}
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    )
}

/* ─── Database Validation Panel ─── */
function DatabaseValidator({ modelId, requiredTables }) {
    const [connStr, setConnStr] = useState('')
    const [validating, setValidating] = useState(false)
    const [result, setResult] = useState(null)
    const [error, setError] = useState(null)

    const handleValidate = async () => {
        if (!connStr.trim()) return
        setValidating(true)
        setError(null)
        setResult(null)
        try {
            const data = await validateDatabase(modelId, connStr.trim())
            setResult(data)
        } catch (e) {
            setError(e.message)
        } finally {
            setValidating(false)
        }
    }

    return (
        <div className="db-validator">
            <div className="db-validator-header">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <ellipse cx="12" cy="5" rx="9" ry="3" />
                    <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" />
                    <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
                </svg>
                <h4>Validate Your Database</h4>
            </div>
            <p className="db-validator-desc">
                Paste your PostgreSQL connection string to check if your database has the required tables and columns.
            </p>
            <div className="db-validator-input-row">
                <input
                    type="text"
                    className="db-conn-input"
                    value={connStr}
                    onChange={e => setConnStr(e.target.value)}
                    placeholder="postgresql+psycopg2://user:pass@host:5432/dbname"
                    onKeyDown={e => e.key === 'Enter' && handleValidate()}
                />
                <button
                    className="model-btn model-btn-validate"
                    onClick={handleValidate}
                    disabled={validating || !connStr.trim()}
                >
                    {validating ? (
                        <><span className="btn-spinner" /> Validating...</>
                    ) : (
                        <>
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <polyline points="20 6 9 17 4 12" />
                            </svg>
                            Validate
                        </>
                    )}
                </button>
            </div>

            {error && (
                <div className="db-val-error">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z" />
                    </svg>
                    {error}
                </div>
            )}

            {result && (
                <div className={`db-val-result ${result.compatible ? 'compatible' : 'incompatible'}`}>
                    <div className="db-val-summary">
                        {result.compatible ? (
                            <><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#059669" strokeWidth="2.5"><polyline points="20 6 9 17 4 12" /></svg>
                                <span>Database is compatible! All {result.total_required} tables found with required columns.</span></>
                        ) : (
                            <><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#dc2626" strokeWidth="2"><circle cx="12" cy="12" r="10" /><line x1="15" y1="9" x2="9" y2="15" /><line x1="9" y1="9" x2="15" y2="15" /></svg>
                                <span>Incompatible — {result.total_found}/{result.total_required} tables found.</span></>
                        )}
                    </div>
                    <div className="db-val-tables">
                        {Object.entries(result.tables).map(([name, info]) => (
                            <div key={name} className={`db-val-table-row ${info.compatible ? 'ok' : 'fail'}`}>
                                <span className="db-val-table-icon">
                                    {info.compatible ? '✓' : info.found ? '⚠' : '✗'}
                                </span>
                                <span className="db-val-table-name">{name}</span>
                                {!info.found && <span className="db-val-note">Table not found</span>}
                                {info.found && info.missing_columns.length > 0 && (
                                    <span className="db-val-note">Missing: {info.missing_columns.join(', ')}</span>
                                )}
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    )
}

/* ─── Results Table ─── */
function ResultsTable({ results }) {
    if (!results || !results.predictions || results.predictions.length === 0) return null

    const allColumns = results.columns || Object.keys(results.predictions[0])
    const columns = allColumns.filter(col => !col.endsWith('_probability'))
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
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                        <polyline points="7 10 12 15 17 10" />
                        <line x1="12" y1="15" x2="12" y2="3" />
                    </svg>
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
                                        {typeof row[col] === 'number' ? row[col].toFixed(2) : String(row[col] ?? '')}
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

/* ─── Input validation helpers ─── */
function validateEntityId(val) {
    if (!val.trim()) return null // empty row will be skipped
    if (!/^\d+$/.test(val.trim())) return 'Must be a whole number (e.g. 42)'
    return null
}

function validateTimestamp(val) {
    if (!val.trim()) return null // empty row will be skipped
    if (!/^\d{4}-\d{2}-\d{2}$/.test(val.trim())) return 'Use YYYY-MM-DD format (e.g. 2024-01-15)'
    const d = new Date(val.trim() + 'T00:00:00')
    if (isNaN(d.getTime())) return 'Invalid date'
    return null
}

function validateManualRow(row) {
    const hasId = !!row.entityId.trim()
    const hasTs = !!row.timestamp.trim()
    return {
        entityId: hasId
            ? validateEntityId(row.entityId)
            : (hasTs ? 'Required when timestamp is set' : null),
        timestamp: hasTs
            ? validateTimestamp(row.timestamp)
            : (hasId ? 'Required when entity ID is set' : null),
    }
}

/* ─── Main Inference Panel ─── */
export default function InferencePanel({ model, onBack }) {
    const [schema, setSchema] = useState(null)
    const [schemaLoading, setSchemaLoading] = useState(true)
    const [file, setFile] = useState(null)
    const [inferring, setInferring] = useState(false)
    const [results, setResults] = useState(null)
    const [error, setError] = useState(null)
    const [activeTab, setActiveTab] = useState('schema')
    const [inputMode, setInputMode] = useState('upload') // 'upload' | 'manual'
    const [manualRows, setManualRows] = useState([
        { entityId: '', timestamp: '' },
        { entityId: '', timestamp: '' },
        { entityId: '', timestamp: '' },
    ])
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

    const addRow = () => setManualRows(prev => [...prev, { entityId: '', timestamp: '' }])
    const removeRow = (i) => setManualRows(prev => prev.filter((_, idx) => idx !== i))
    const updateRow = (i, field, val) =>
        setManualRows(prev => prev.map((r, idx) => idx === i ? { ...r, [field]: val } : r))

    const handleInfer = async () => {
        let fileToSend = file
        if (inputMode === 'manual') {
            const validRows = manualRows.filter(r => r.entityId.trim() && r.timestamp.trim())
            if (validRows.length === 0) {
                setError('Please enter at least one row with an entity ID and timestamp.')
                return
            }
            const csv = [
                `${entityCol},${timeCol}`,
                ...validRows.map(r => `${r.entityId.trim()},${r.timestamp.trim()}`)
            ].join('\n')
            fileToSend = new File([csv], 'input.csv', { type: 'text/csv' })
        }
        if (!fileToSend) return
        setInferring(true)
        setError(null)
        setResults(null)
        try {
            const data = await runInference(model.id, fileToSend)
            setResults(data)
        } catch (e) {
            setError(e.message)
        } finally {
            setInferring(false)
        }
    }

    const entityCol = taskMeta.entity_col || schema?.entity_col || 'id'
    const timeCol = taskMeta.time_col || schema?.task_meta?.time_col || 'timestamp'
    const tables = schema?.tables || {}
    const tableNames = Object.keys(tables)
    const entityTableName = taskMeta.entity_table || schema?.entity_table || ''

    // Build a relationship summary
    const relationshipSummary = []
    for (const [tName, tInfo] of Object.entries(tables)) {
        const fks = tInfo.foreign_keys || {}
        for (const [fkCol, refTable] of Object.entries(fks)) {
            relationshipSummary.push({ from: tName, column: fkCol, to: refTable })
        }
    }

    // Manual entry validation
    const rowErrors = manualRows.map(validateManualRow)
    const invalidRowIndices = rowErrors
        .map((e, i) => (e.entityId || e.timestamp ? i + 1 : null))
        .filter(Boolean)
    const hasAnyValidRow = manualRows.some(r => r.entityId.trim() && r.timestamp.trim())

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
                {taskMeta.task_type && (
                    <span className="infer-task-badge">{taskMeta.task_type.replace(/_/g, ' ')}</span>
                )}
            </div>

            {/* ─── Model Summary Bar ─── */}
            <div className="infer-summary-bar">
                <div className="infer-summary-item">
                    <span className="infer-summary-label">Entity</span>
                    <span className="infer-summary-value">{entityTableName}.{entityCol}</span>
                </div>
                {taskMeta.target_col && (
                    <div className="infer-summary-item">
                        <span className="infer-summary-label">Target</span>
                        <span className="infer-summary-value">{taskMeta.target_col}</span>
                    </div>
                )}
                {taskMeta.timedelta_days && (
                    <div className="infer-summary-item">
                        <span className="infer-summary-label">Window</span>
                        <span className="infer-summary-value">{taskMeta.timedelta_days} days</span>
                    </div>
                )}
                <div className="infer-summary-item">
                    <span className="infer-summary-label">Tables</span>
                    <span className="infer-summary-value">{tableNames.length}</span>
                </div>
                {Object.keys(valMetrics).length > 0 && (
                    <div className="infer-summary-item">
                        <span className="infer-summary-label">Best Metric</span>
                        <span className="infer-summary-value">
                            {model.tune_metric}: {formatMetric(valMetrics[model.tune_metric])}
                        </span>
                    </div>
                )}
            </div>

            {/* ─── Tab Navigation ─── */}
            <div className="infer-tabs">
                <button
                    className={`infer-tab ${activeTab === 'schema' ? 'active' : ''}`}
                    onClick={() => setActiveTab('schema')}
                >
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <rect x="3" y="3" width="18" height="18" rx="2" />
                        <line x1="3" y1="9" x2="21" y2="9" />
                        <line x1="9" y1="3" x2="9" y2="21" />
                    </svg>
                    Database Schema
                </button>
                <button
                    className={`infer-tab ${activeTab === 'validate' ? 'active' : ''}`}
                    onClick={() => setActiveTab('validate')}
                >
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <ellipse cx="12" cy="5" rx="9" ry="3" />
                        <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" />
                        <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
                    </svg>
                    Validate Connection
                </button>
                <button
                    className={`infer-tab ${activeTab === 'run' ? 'active' : ''}`}
                    onClick={() => setActiveTab('run')}
                >
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
                    </svg>
                    Run Inference
                </button>
            </div>

            {/* ─── Tab Content ─── */}
            <div className="infer-tab-content">

                {/* Schema Tab */}
                {activeTab === 'schema' && (
                    <div className="infer-schema-tab">
                        {schemaLoading ? (
                            <div className="models-loading">
                                <div className="loading-spinner" />
                                <p>Loading database schema...</p>
                            </div>
                        ) : (
                            <>
                                {/* Relationship overview */}
                                {relationshipSummary.length > 0 && (
                                    <div className="rel-overview">
                                        <div className="rel-overview-header">
                                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                                                <circle cx="5" cy="12" r="3" />
                                                <circle cx="19" cy="6" r="3" />
                                                <circle cx="19" cy="18" r="3" />
                                                <line x1="8" y1="11" x2="16" y2="7" />
                                                <line x1="8" y1="13" x2="16" y2="17" />
                                            </svg>
                                            <span>Table Relationships</span>
                                        </div>
                                        <div className="rel-list">
                                            {relationshipSummary.map((r, i) => (
                                                <div key={i} className="rel-item">
                                                    <span className="rel-from">{r.from}</span>
                                                    <span className="rel-arrow">
                                                        <span className="rel-col">.{r.column}</span>
                                                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="5" y1="12" x2="19" y2="12" /><polyline points="12 5 19 12 12 19" /></svg>
                                                    </span>
                                                    <span className="rel-to">{r.to}</span>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {/* Table detail cards */}
                                <div className="tbl-detail-list">
                                    {tableNames.sort((a, b) => {
                                        if (a === entityTableName) return -1
                                        if (b === entityTableName) return 1
                                        return a.localeCompare(b)
                                    }).map(name => (
                                        <TableDetailCard
                                            key={name}
                                            name={name}
                                            info={tables[name]}
                                            isEntity={name === entityTableName}
                                        />
                                    ))}
                                </div>
                            </>
                        )}
                    </div>
                )}

                {/* Validate Tab */}
                {activeTab === 'validate' && (
                    <div className="infer-validate-tab">
                        <DatabaseValidator modelId={model.id} requiredTables={tables} />
                    </div>
                )}

                {/* Run Inference Tab */}
                {activeTab === 'run' && (
                    <div className="infer-run-tab">
                        <div className="inference-instructions">
                            <h4>
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                                    <circle cx="12" cy="12" r="10" />
                                    <line x1="12" y1="16" x2="12" y2="12" />
                                    <line x1="12" y1="8" x2="12.01" y2="8" />
                                </svg>
                                Data Preparation
                            </h4>
                            <p>
                                This model uses a <strong>Graph Neural Network</strong> trained on a <strong>{tableNames.length}-table relational database</strong>.
                                The model already has access to the training data. You only need to provide a CSV specifying <em>which entities</em> to predict for and <em>at what time</em>.
                            </p>
                            <div className="infer-req-cols">
                                <div className="infer-req-col">
                                    <code>{entityCol}</code>
                                    <span>Entity ID from the <strong>{entityTableName}</strong> table</span>
                                </div>
                                <div className="infer-req-col">
                                    <code>{timeCol}</code>
                                    <span>Prediction timestamp (ISO format, e.g. <code>2024-01-15</code>)
                                        {taskMeta.timedelta_days
                                            ? ` — predicts ${taskMeta.timedelta_days}-day window from this date`
                                            : ''}
                                    </span>
                                </div>
                            </div>

                            <div className="csv-example">
                                <div className="csv-example-header">Example CSV</div>
                                <pre className="csv-example-content">{entityCol},{timeCol}{'\n'}1,2024-01-15{'\n'}2,2024-01-15{'\n'}3,2024-02-01</pre>
                            </div>

                            <div className="inference-notes">
                                <h5>Important Notes</h5>
                                <ul>
                                    <li>Entity IDs must exist in the original training database.</li>
                                    <li>Timestamps should be after the training data period for meaningful predictions.</li>
                                    <li>The model uses the full relational graph ({tableNames.length} tables, {relationshipSummary.length} relationships) for predictions.</li>
                                    {taskMeta.task_type === 'binary_classification' && (
                                        <li>Output includes both probability scores and binary predictions (threshold = 0.5).</li>
                                    )}
                                </ul>
                            </div>
                        </div>

                        <div className="inference-upload-section">
                            <div className="infer-input-toggle">
                                <button
                                    className={`infer-input-tab ${inputMode === 'upload' ? 'active' : ''}`}
                                    onClick={() => setInputMode('upload')}
                                >
                                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                                        <polyline points="17 8 12 3 7 8" />
                                        <line x1="12" y1="3" x2="12" y2="15" />
                                    </svg>
                                    Upload CSV
                                </button>
                                <button
                                    className={`infer-input-tab ${inputMode === 'manual' ? 'active' : ''}`}
                                    onClick={() => setInputMode('manual')}
                                >
                                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <line x1="8" y1="6" x2="21" y2="6" />
                                        <line x1="8" y1="12" x2="21" y2="12" />
                                        <line x1="8" y1="18" x2="21" y2="18" />
                                        <line x1="3" y1="6" x2="3.01" y2="6" />
                                        <line x1="3" y1="12" x2="3.01" y2="12" />
                                        <line x1="3" y1="18" x2="3.01" y2="18" />
                                    </svg>
                                    Manual Entry
                                </button>
                            </div>

                            {inputMode === 'upload' ? (
                                <>
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
                                </>
                            ) : (
                                <div className="manual-entry-wrap">
                                    <div className="manual-table-scroll">
                                        <table className="manual-table">
                                            <thead>
                                                <tr>
                                                    <th>#</th>
                                                    <th><code>{entityCol}</code> <span className="col-hint">integer</span></th>
                                                    <th><code>{timeCol}</code> <span className="col-hint">YYYY-MM-DD</span></th>
                                                    <th></th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {manualRows.map((row, i) => {
                                                    const errs = rowErrors[i]
                                                    const rowHasError = !!(errs.entityId || errs.timestamp)
                                                    return (
                                                        <tr key={i} className={rowHasError ? 'row-invalid' : ''}>
                                                            <td className={`row-num ${rowHasError ? 'has-error' : ''}`}>
                                                                {rowHasError ? (
                                                                    <svg width="12" height="12" viewBox="0 0 24 24" fill="#ef4444" title="This row has errors">
                                                                        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z" />
                                                                    </svg>
                                                                ) : i + 1}
                                                            </td>
                                                            <td>
                                                                <input
                                                                    className={`manual-cell-input ${errs.entityId ? 'is-invalid' : ''}`}
                                                                    value={row.entityId}
                                                                    onChange={e => updateRow(i, 'entityId', e.target.value)}
                                                                    placeholder="e.g. 42"
                                                                    type="text"
                                                                    title={errs.entityId || ''}
                                                                    aria-invalid={!!errs.entityId}
                                                                />
                                                            </td>
                                                            <td>
                                                                <input
                                                                    className={`manual-cell-input ${errs.timestamp ? 'is-invalid' : ''}`}
                                                                    value={row.timestamp}
                                                                    onChange={e => updateRow(i, 'timestamp', e.target.value)}
                                                                    placeholder="e.g. 2024-01-15"
                                                                    type="text"
                                                                    title={errs.timestamp || ''}
                                                                    aria-invalid={!!errs.timestamp}
                                                                />
                                                            </td>
                                                            <td>
                                                                <button
                                                                    className="manual-row-delete"
                                                                    onClick={() => removeRow(i)}
                                                                    disabled={manualRows.length === 1}
                                                                    title="Remove row"
                                                                >
                                                                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                                                        <line x1="18" y1="6" x2="6" y2="18" />
                                                                        <line x1="6" y1="6" x2="18" y2="18" />
                                                                    </svg>
                                                                </button>
                                                            </td>
                                                        </tr>
                                                    )
                                                })}
                                            </tbody>
                                        </table>
                                    </div>
                                    <button className="manual-add-row-btn" onClick={addRow}>
                                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                                            <line x1="12" y1="5" x2="12" y2="19" />
                                            <line x1="5" y1="12" x2="19" y2="12" />
                                        </svg>
                                        Add Row
                                    </button>
                                    {invalidRowIndices.length > 0 && (
                                        <div className="manual-validation-warning">
                                            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ flexShrink: 0 }}>
                                                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
                                                <line x1="12" y1="9" x2="12" y2="13" />
                                                <line x1="12" y1="17" x2="12.01" y2="17" />
                                            </svg>
                                            <span>
                                                Row{invalidRowIndices.length > 1 ? 's' : ''} <strong>{invalidRowIndices.join(', ')}</strong>{' '}
                                                {invalidRowIndices.length > 1 ? 'have' : 'has'} invalid values.
                                                Hover over a highlighted cell to see the error.
                                            </span>
                                        </div>
                                    )}
                                </div>
                            )}

                            <button
                                className="model-btn model-btn-infer inference-run-btn"
                                onClick={handleInfer}
                                disabled={(inputMode === 'upload' && !file) || inferring || (inputMode === 'manual' && (invalidRowIndices.length > 0 || !hasAnyValidRow))}
                            >
                                {inferring ? (
                                    <>
                                        <span className="btn-spinner" />
                                        Running Inference...
                                    </>
                                ) : (
                                    <>
                                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                            <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
                                        </svg>
                                        Run Inference
                                    </>
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
                )}
            </div>
        </div>
    )
}

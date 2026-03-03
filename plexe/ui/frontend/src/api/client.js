/**
 * API Client for Plexe Frontend
 * Handles all HTTP requests to the backend
 */

const API_BASE_URL = import.meta.env.VITE_BACKEND_URL || ''

/**
 * Upload file(s) to the backend
 * @param {File[]} files - Array of files to upload
 * @returns {Promise<Object>} Upload response
 */
export async function uploadFiles(files) {
    const formData = new FormData()

    // Add all files to FormData
    files.forEach((file) => {
        formData.append('files', file)
    })

    const response = await fetch(`${API_BASE_URL}/api/upload`, {
        method: 'POST',
        body: formData,
        headers: {
            // Don't set Content-Type, let browser set it with boundary
        },
    })

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || `Upload failed: ${response.statusText}`)
    }

    return await response.json()
}

/**
 * Test PostgreSQL connection
 * @param {Object} connectionConfig - PostgreSQL connection config
 * @returns {Promise<Object>} Test result
 */
export async function testPostgresConnection(connectionConfig) {
    const response = await fetch(`${API_BASE_URL}/api/postgres/test`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(connectionConfig),
    })

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || 'Connection test failed')
    }

    return await response.json()
}

/**
 * Execute a query on a PostgreSQL database
 * @param {Object} connectionConfig - PostgreSQL connection config
 * @returns {Promise<Object>} Query result
 */
export async function executePostgresQuery(connectionConfig) {
    const response = await fetch(`${API_BASE_URL}/api/postgres/execute`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(connectionConfig),
    })

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || 'Failed to execute query')
    }

    return await response.json()
}

/**
 * Get list of uploaded datasets
 * @returns {Promise<Array>} List of datasets
 */
export async function listDatasets() {
    const response = await fetch(`${API_BASE_URL}/api/datasets`, {
        method: 'GET',
    })

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || 'Failed to fetch datasets')
    }

    return await response.json()
}

/**
 * Delete a dataset
 * @param {string} datasetId - ID of dataset to delete
 * @returns {Promise<Object>} Delete result
 */
export async function deleteDataset(datasetId) {
    const response = await fetch(`${API_BASE_URL}/api/datasets/${datasetId}`, {
        method: 'DELETE',
    })

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || 'Failed to delete dataset')
    }

    return await response.json()
}

/**
 * Download a dataset
 * @param {string} datasetId - ID of dataset to download
 * @returns {Promise<Blob>} Downloaded file as a Blob
 */
export async function downloadDataset(datasetId) {
    const response = await fetch(`${API_BASE_URL}/api/datasets/${datasetId}/download`, {
        method: 'GET',
    })

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || 'Failed to download dataset')
    }

    return await response.blob()
}

/**
 * Combine datasets using featuretools
 * @param {Array} tables - List of tables
 * @param {Array} relationships - List of relationships
 * @param {Object} connection - PostgreSQL connection config
 * @returns {Promise<Object>} Combination result
 */
export async function combineDatasets(tables, relationships, connection) {
    const response = await fetch(`${API_BASE_URL}/api/datasets/combine`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ tables, relationships, connection }),
    })

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || 'Failed to combine datasets')
    }

    return await response.json()
}

/**
 * List all trained models
 * @returns {Promise<Object>} List of models
 */
export async function listModels() {
    const response = await fetch(`${API_BASE_URL}/api/models`, { method: 'GET' })
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || 'Failed to fetch models')
    }
    return await response.json()
}

/**
 * Get model details
 * @param {string} modelId - Model ID
 * @returns {Promise<Object>} Model details
 */
export async function getModel(modelId) {
    const response = await fetch(`${API_BASE_URL}/api/models/${encodeURIComponent(modelId)}`, { method: 'GET' })
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || 'Failed to fetch model')
    }
    return await response.json()
}

/**
 * Delete a model
 * @param {string} modelId - Model ID
 * @returns {Promise<Object>} Delete result
 */
export async function deleteModel(modelId) {
    const response = await fetch(`${API_BASE_URL}/api/models/${encodeURIComponent(modelId)}`, { method: 'DELETE' })
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || 'Failed to delete model')
    }
    return await response.json()
}

/**
 * Rename a model
 * @param {string} modelId - Model ID
 * @param {string} newName - New name
 * @returns {Promise<Object>} Rename result
 */
export async function renameModel(modelId, newName) {
    const response = await fetch(`${API_BASE_URL}/api/models/${encodeURIComponent(modelId)}/rename`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ new_name: newName }),
    })
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || 'Failed to rename model')
    }
    return await response.json()
}

/**
 * Get model input schema
 * @param {string} modelId - Model ID
 * @returns {Promise<Object>} Schema info
 */
export async function getModelSchema(modelId) {
    const response = await fetch(`${API_BASE_URL}/api/models/${encodeURIComponent(modelId)}/schema`, { method: 'GET' })
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || 'Failed to fetch model schema')
    }
    return await response.json()
}

/**
 * Run inference on a model
 * @param {string} modelId - Model ID
 * @param {File} file - CSV file with input data
 * @returns {Promise<Object>} Inference results
 */
export async function runInference(modelId, file) {
    const formData = new FormData()
    formData.append('file', file)
    const response = await fetch(`${API_BASE_URL}/api/models/${encodeURIComponent(modelId)}/infer`, {
        method: 'POST',
        body: formData,
    })
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || 'Inference failed')
    }
    return await response.json()
}

/**
 * Validate a database connection against a model's required schema
 * @param {string} modelId - Model ID
 * @param {string} connectionString - Database connection string
 * @returns {Promise<Object>} Validation results
 */
export async function validateDatabase(modelId, connectionString) {
    const response = await fetch(`${API_BASE_URL}/api/models/${encodeURIComponent(modelId)}/validate-db`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ connection_string: connectionString }),
    })
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || 'Database validation failed')
    }
    return await response.json()
}

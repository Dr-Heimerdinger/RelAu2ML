import React, { useRef, useState } from 'react'
import { FolderOpen, FileText, X, Check, AlertCircle, Upload } from 'lucide-react'
import { uploadFiles } from '../../api/client'

export default function UploadTab() {
    const [files, setFiles] = useState([])
    const [uploading, setUploading] = useState(false)
    const [uploadError, setUploadError] = useState(null)
    const [uploadSuccess, setUploadSuccess] = useState(false)
    const fileInputRef = useRef(null)

    const handleFileChange = (e) => {
        const selectedFiles = Array.from(e.target.files)
        setFiles((prevFiles) => [...prevFiles, ...selectedFiles])
    }

    const handleUpload = async () => {
        if (files.length === 0) {
            setUploadError('Please select files to upload')
            return
        }

        setUploading(true)
        setUploadError(null)
        setUploadSuccess(false)

        try {
            const result = await uploadFiles(files)
            setUploadSuccess(true)
            setFiles([])
            setTimeout(() => setUploadSuccess(false), 3000)
        } catch (error) {
            console.error('Upload error:', error)
            setUploadError(error.message || 'Upload failed')
        } finally {
            setUploading(false)
        }
    }

    const removeFile = (index) => {
        setFiles((prevFiles) => prevFiles.filter((_, i) => i !== index))
    }

    return (
        <div className="upload-tab">
            <div className="upload-area">
                <div
                    className="upload-box"
                    onClick={() => fileInputRef.current?.click()}
                    onDragOver={(e) => {
                        e.preventDefault()
                        e.currentTarget.style.borderColor = 'var(--primary)'
                    }}
                    onDragLeave={(e) => {
                        e.currentTarget.style.borderColor = 'var(--border)'
                    }}
                    onDrop={(e) => {
                        e.preventDefault()
                        e.currentTarget.style.borderColor = 'var(--border)'
                        const droppedFiles = Array.from(e.dataTransfer.files)
                        setFiles((prevFiles) => [...prevFiles, ...droppedFiles])
                    }}
                >
                    <input
                        ref={fileInputRef}
                        type="file"
                        multiple
                        onChange={handleFileChange}
                        style={{ display: 'none' }}
                        accept=".csv,.xlsx,.xls,.json,.parquet"
                    />
                    <div className="upload-icon"><FolderOpen size={32} /></div>
                    <p className="upload-text">Drag and drop files here or click to browse</p>
                    <p className="upload-hint">Supported formats: CSV, XLSX, JSON, Parquet</p>
                </div>
            </div>

            {uploadError && (
                <div className="status-message error">
                    <AlertCircle size={14} /> {uploadError}
                </div>
            )}

            {uploadSuccess && (
                <div className="status-message success">
                    <Check size={14} /> Files uploaded successfully!
                </div>
            )}

            {files.length > 0 && (
                <div className="files-list">
                    <h3>Selected Files ({files.length})</h3>
                    <ul className="file-items">
                        {files.map((file, index) => (
                            <li key={index} className="file-item">
                                <span className="file-name"><FileText size={14} /> {file.name}</span>
                                <span className="file-size">({(file.size / 1024).toFixed(2)} KB)</span>
                                <button
                                    className="remove-btn"
                                    onClick={() => removeFile(index)}
                                >
                                    <X size={14} />
                                </button>
                            </li>
                        ))}
                    </ul>

                    <button
                        className="upload-button"
                        onClick={handleUpload}
                        disabled={uploading}
                    >
                        {uploading ? 'Uploading...' : <><Upload size={16} /> Upload Files</>}
                    </button>
                </div>
            )}
        </div>
    )
}

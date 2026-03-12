import React, { useState } from 'react'
import { BarChart3, Upload, Database } from 'lucide-react'
import UploadTab from './dataset/UploadTab'
import PostgreSQLTab from './dataset/PostgreSQLTab'
import OverviewTab from './dataset/OverviewTab'

export default function Dataset() {
    const [activeTab, setActiveTab] = useState('upload')

    return (
        <div className="dataset-container">
            <div className="dataset-header">
                <h2>Dataset Management</h2>
                <p>Upload or connect to your data sources</p>
            </div>

            <div className="tabs">
                <div className="tab-buttons">
                    <button
                        className={`tab-button ${activeTab === 'overview' ? 'active' : ''}`}
                        onClick={() => setActiveTab('overview')}
                    >
                        <span className="icon"><BarChart3 size={16} /></span>
                        Overview
                    </button>
                    <button
                        className={`tab-button ${activeTab === 'upload' ? 'active' : ''}`}
                        onClick={() => setActiveTab('upload')}
                    >
                        <span className="icon"><Upload size={16} /></span>
                        Upload Data
                    </button>
                    <button
                        className={`tab-button ${activeTab === 'postgres' ? 'active' : ''}`}
                        onClick={() => setActiveTab('postgres')}
                    >
                        <span className="icon"><Database size={16} /></span>
                        PostgreSQL
                    </button>

                </div>

                <div className="tab-content">

                    {activeTab === 'upload' && <UploadTab />}
                    {activeTab === 'postgres' && <PostgreSQLTab />}
                    {activeTab === 'overview' && <OverviewTab />}
                </div>
            </div>
        </div>
    )
}

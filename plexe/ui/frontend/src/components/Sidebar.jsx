import React from 'react'

export default function Sidebar({ activePage, setActivePage }) {
    return (
        <aside className="sidebar">
            <div className="sidebar-header">
                <h2>Plexe</h2>
                <p>Assistant</p>
            </div>

            <nav className="sidebar-nav">
                <button
                    className={`nav-item ${activePage === 'chat' ? 'active' : ''}`}
                    onClick={() => setActivePage('chat')}
                >
                    <span className="icon">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                        </svg>
                    </span>
                    <span className="label">Chat</span>
                </button>

                <button
                    className={`nav-item ${activePage === 'dataset' ? 'active' : ''}`}
                    onClick={() => setActivePage('dataset')}
                >
                    <span className="icon">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <ellipse cx="12" cy="5" rx="9" ry="3" />
                            <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" />
                            <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
                        </svg>
                    </span>
                    <span className="label">Dataset</span>
                </button>

                <button
                    className={`nav-item ${activePage === 'models' ? 'active' : ''}`}
                    onClick={() => setActivePage('models')}
                >
                    <span className="icon">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
                            <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
                            <line x1="12" y1="22.08" x2="12" y2="12" />
                        </svg>
                    </span>
                    <span className="label">Models</span>
                </button>
            </nav>

            <div className="sidebar-footer">
                <p className="version">v0.26.2</p>
            </div>
        </aside>
    )
}

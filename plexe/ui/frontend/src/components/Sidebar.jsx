import React from 'react'
import { MessageSquare, Database, Box, Sun, Moon, Layers } from 'lucide-react'

export default function Sidebar({ activePage, setActivePage, theme, onToggleTheme }) {
    return (
        <aside className="sidebar">
            <div className="sidebar-header">
                <div className="sidebar-logo">
                    <Layers size={28} strokeWidth={1.75} />
                    <h2>Plexe</h2>
                </div>
                <p>AI Model Builder</p>
            </div>

            <nav className="sidebar-nav">
                <button
                    className={`nav-item ${activePage === 'chat' ? 'active' : ''}`}
                    onClick={() => setActivePage('chat')}
                >
                    <span className="icon">
                        <MessageSquare size={20} />
                    </span>
                    <span className="label">Chat</span>
                </button>

                <button
                    className={`nav-item ${activePage === 'dataset' ? 'active' : ''}`}
                    onClick={() => setActivePage('dataset')}
                >
                    <span className="icon">
                        <Database size={20} />
                    </span>
                    <span className="label">Dataset</span>
                </button>

                <button
                    className={`nav-item ${activePage === 'models' ? 'active' : ''}`}
                    onClick={() => setActivePage('models')}
                >
                    <span className="icon">
                        <Box size={20} />
                    </span>
                    <span className="label">Models</span>
                </button>
            </nav>

            <div className="sidebar-footer">
                <button
                    className="theme-toggle"
                    onClick={onToggleTheme}
                    title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
                >
                    {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
                    <span>{theme === 'dark' ? 'Light Mode' : 'Dark Mode'}</span>
                </button>
            </div>
        </aside>
    )
}

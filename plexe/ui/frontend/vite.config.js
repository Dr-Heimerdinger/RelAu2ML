import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Use environment variable for backend URL, defaulting to Docker service name
// When running in Docker, use 'backend:BACKEND_PORT'. When running locally, use 'localhost:BACKEND_PORT'
const backendHost = process.env.VITE_BACKEND_HOST || 'backend'
const backendPort = process.env.VITE_BACKEND_PORT || '8100'
const backendUrl = `http://${backendHost}:${backendPort}`
const frontendPort = parseInt(process.env.VITE_FRONTEND_PORT || '3000')

export default defineConfig({
    plugins: [react()],
    server: {
        host: '0.0.0.0',
        port: frontendPort,
        strictPort: false,
        hmr: {
            host: 'localhost',
            port: frontendPort,
            protocol: 'ws',
        },
        proxy: {
            // Proxy /ws to backend during development
            '/ws': {
                target: backendUrl,
                ws: true,
                changeOrigin: true,
            },
            // Proxy /api requests to backend
            '/api': {
                target: backendUrl,
                changeOrigin: true,
            },
            // Proxy /health to backend
            '/health': {
                target: backendUrl,
                changeOrigin: true,
            },
        },
    },
})

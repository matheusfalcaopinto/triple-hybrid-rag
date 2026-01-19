import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import fs from 'fs'
import path from 'path'

// Check if SSL certificates exist
const certsDir = path.resolve(__dirname, '../certs')
const keyPath = path.join(certsDir, 'key.pem')
const certPath = path.join(certsDir, 'cert.pem')

const httpsConfig = fs.existsSync(keyPath) && fs.existsSync(certPath)
  ? {
      key: fs.readFileSync(keyPath),
      cert: fs.readFileSync(certPath),
    }
  : undefined

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Bind to all interfaces for network access
    host: '0.0.0.0',
    port: 5173,
    // Enable HTTPS if certificates are available
    https: httpsConfig,
    // Proxy API requests to backend (HTTPS)
    proxy: {
      '/api': {
        target: 'https://localhost:8009',
        changeOrigin: true,
        secure: false, // Accept self-signed certificates
        ws: true,
        configure: (proxy) => {
          proxy.on('error', (err) => {
            console.log('Proxy error:', err);
          });
        },
      },
    },
  },
  preview: {
    host: '0.0.0.0',
    port: 5173,
    https: httpsConfig,
  },
})

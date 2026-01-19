# Triple-Hybrid-RAG Dashboard

A modern React + FastAPI dashboard for managing the Triple-Hybrid-RAG pipeline with multimodal support.

## Features

- **File Ingestion**: Upload PDF, DOCX, XLSX, CSV, images with OCR support
- **Triple-Hybrid Retrieval**: Query using lexical, semantic, and graph search
- **Configuration Management**: Adjust 60+ RAG parameters in real-time
- **Database Browser**: View documents, chunks, and entities
- **Metrics Dashboard**: Monitor pipeline status and statistics

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL with pgvector extension
- SSL certificates (auto-generated or custom)

### 1. Generate SSL Certificates (First Time Only)

SSL certificates are required for HTTPS access from other machines on your network.

```bash
# Create certificates directory and generate self-signed certificates
mkdir -p dashboard/certs
cd dashboard/certs
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes \
  -subj "/C=US/ST=Local/L=Local/O=Dev/CN=localhost" \
  -addext "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:0.0.0.0"
cd ../..
```

### 2. Start the Backend

```bash
# From project root (HTTPS with SSL certificates)
uv run python -m uvicorn dashboard.backend.main:app --reload --host 0.0.0.0 --port 8009 \
  --ssl-keyfile dashboard/certs/key.pem --ssl-certfile dashboard/certs/cert.pem

# OR using main.py directly (auto-detects SSL certs)
cd dashboard/backend && python main.py
```

The backend will start on `https://0.0.0.0:8009`.

### 3. Start the Frontend

```bash
# From project root
cd dashboard/frontend
npm install  # First time only
npm run dev
```

The frontend will start on `https://0.0.0.0:5173`.

## Accessing the Dashboard

### Local Access (Host Machine)

- **Frontend**: https://localhost:5173
- **Backend API**: https://localhost:8009/api
- **API Docs**: https://localhost:8009/docs

### Network Access (Other Machines)

Replace `<HOST_IP>` with your machine's IP address (e.g., `192.168.1.100`):

- **Frontend**: https://<HOST_IP>:5173
- **Backend API**: https://<HOST_IP>:8009/api

**Note**: Both frontend and backend use HTTPS. You'll need to accept the self-signed certificate warning for both ports (5173 and 8009) in your browser.

To find your IP address:
```bash
# Linux
ip addr show | grep "inet " | grep -v 127.0.0.1

# macOS
ifconfig | grep "inet " | grep -v 127.0.0.1

# Windows
ipconfig | findstr /i "IPv4"
```

### Accepting Self-Signed Certificates

When accessing via HTTPS with self-signed certificates, browsers will show a security warning:

1. **Chrome**: Click "Advanced" → "Proceed to <host> (unsafe)"
2. **Firefox**: Click "Advanced" → "Accept the Risk and Continue"
3. **Safari**: Click "Show Details" → "visit this website"

**Important**: You need to accept the certificate for both:
- Frontend: `https://localhost:5173` (or `https://<IP>:5173`)
- Backend: `https://localhost:8009` (or `https://<IP>:8009`)

The Vite dev server proxies API requests through `/api`, but for direct API access or when the proxy can't reach the backend, you'll need to accept the backend certificate as well.

## Architecture

```
dashboard/
├── backend/
│   ├── main.py          # FastAPI application with HTTPS support
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx      # Main application
│   │   ├── components/  # React components
│   │   └── hooks/       # API hooks with dynamic URL
│   └── vite.config.ts   # Vite config with HTTPS & proxy
└── certs/
    ├── cert.pem         # SSL certificate
    └── key.pem          # SSL private key
```

## Configuration

### Environment Variables

The backend reads configuration from the project root `.env` file:

```env
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/ragdb

# RAG Settings
RAG_ENABLED=true
RAG_LEXICAL_ENABLED=true
RAG_SEMANTIC_ENABLED=true
RAG_GRAPH_ENABLED=true

# OCR Settings
RAG_OCR_ENABLED=true
RAG_OCR_MODE=auto
```

### CORS Configuration

The backend allows all origins by default for local network access. For production, restrict origins in `main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/info` | GET | System information |
| `/api/config` | GET | Get configuration |
| `/api/config` | POST | Update configuration |
| `/api/ingest/upload` | POST | Upload file for ingestion |
| `/api/ingest/status/{job_id}` | GET | Get ingestion job status |
| `/api/retrieve` | POST | Execute retrieval query |
| `/api/database/stats` | GET | Get database statistics |
| `/api/database/documents` | GET | List documents |
| `/api/database/entities` | GET | List entities |
| `/api/metrics` | GET | Get pipeline metrics |

## Troubleshooting

### Cannot Access from Other Machines

1. **Firewall**: Ensure ports 5173 and 8009 are open
   ```bash
   # Linux (ufw)
   sudo ufw allow 5173/tcp
   sudo ufw allow 8009/tcp
   
   # Linux (firewalld)
   sudo firewall-cmd --add-port=5173/tcp --permanent
   sudo firewall-cmd --add-port=8009/tcp --permanent
   sudo firewall-cmd --reload
   ```

2. **Certificate Issues**: Accept certificates on both ports in your browser

3. **Network**: Ensure both machines are on the same network

### SSL Certificate Errors

If you see certificate errors, regenerate certificates with the correct IP:

```bash
cd dashboard/certs
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes \
  -subj "/C=US/ST=Local/L=Local/O=Dev/CN=localhost" \
  -addext "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:<YOUR_IP>"
```

### Backend Not Starting with HTTPS

Ensure certificates exist at `dashboard/certs/key.pem` and `dashboard/certs/cert.pem`. The backend will fall back to HTTP if certificates are not found.

## Development

### Running in HTTP Mode (No Certificates)

If you don't need network access, you can run without HTTPS:

1. Delete or rename the `dashboard/certs` directory
2. Update `useApi.ts` to use `http://localhost:8009/api` directly
3. Start backend and frontend normally

### Custom Ports

To change ports, update:
- Backend: `uvicorn.run(..., port=NEW_PORT)` in `main.py`
- Frontend: `server.port` in `vite.config.ts`
- Proxy target in `vite.config.ts`

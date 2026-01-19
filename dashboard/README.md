# Triple-Hybrid-RAG Dashboard

A comprehensive web dashboard for managing, configuring, and visualizing the Triple-Hybrid-RAG pipeline.

## Features

- **📊 Dashboard Overview**: Database statistics, feature status, ingestion job summary
- **⚙️ Configuration Management**: Toggle features, adjust parameters, save to `.env`
- **📁 File Ingestion**: Drag-and-drop upload with progress tracking
- **🔍 Query Interface**: Execute retrieval queries with full score breakdown
- **🗃️ Database Browser**: Explore documents and entities
- **🔗 Graph Visualization**: Embedded PuppyGraph Web UI

## Prerequisites

- Node.js 18+
- Python 3.11+
- Docker (for PostgreSQL, pgvector, PuppyGraph)

## Quick Start

### 1. Start Infrastructure

```bash
cd /home/matheus/repos/triple-hybrid-rag
docker compose up -d
```

### 2. Start Backend API

```bash
cd dashboard/backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 3. Start Frontend

```bash
cd dashboard/frontend
npm install
npm run dev
```

### 4. Open Dashboard

Navigate to **<http://localhost:5173>**

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/config` | GET | Get configuration |
| `/api/config` | POST | Update configuration |
| `/api/config/reload` | POST | Reload from .env |
| `/api/ingest/upload` | POST | Upload file for ingestion |
| `/api/ingest/status/{id}` | GET | Get ingestion job status |
| `/api/ingest/jobs` | GET | List all ingestion jobs |
| `/api/retrieve` | POST | Execute retrieval query |
| `/api/database/stats` | GET | Get database statistics |
| `/api/database/documents` | GET | List documents |
| `/api/database/entities` | GET | List entities |
| `/api/metrics` | GET | Get pipeline metrics |

## Architecture

```
dashboard/
├── backend/              # FastAPI backend
│   ├── main.py           # API endpoints
│   └── requirements.txt  # Python dependencies
└── frontend/             # React TypeScript frontend
    ├── src/
    │   ├── App.tsx       # Main application
    │   ├── index.css     # Design system
    │   ├── components/   # UI components
    │   ├── hooks/        # API hooks
    │   └── types/        # TypeScript types
    └── package.json      # Node dependencies
```

## Technology Stack

- **Frontend**: React, TypeScript, Vite
- **Backend**: Python, FastAPI, asyncpg
- **Database**: PostgreSQL + pgvector
- **Graph**: PuppyGraph (Bolt protocol)

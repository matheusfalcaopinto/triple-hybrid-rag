# Phase 1: Deploy PuppyGraph

> **Priority:** CRITICAL - Unblocks multiple downstream tasks  
> **Time:** 30 minutes  
> **Status:** [ ] Not Started

---

## Objective

Deploy PuppyGraph container and apply schema to enable the Graph channel (Cypher queries) for RAG 2.0 retrieval.

---

## Current State

- ✅ `infrastructure/puppygraph/docker-compose.yml` exists
- ✅ `infrastructure/puppygraph/schema.json` exists  
- ✅ `infrastructure/puppygraph/README.md` exists
- ✅ `graph_search.py` has SQL fallback working
- ❌ PuppyGraph container NOT running
- ❌ Schema NOT applied
- ❌ `RAG2_GRAPH_ENABLED=false` in .env

---

## Steps

### 1.1 Start PuppyGraph Container

```bash
cd /home/matheus/repos/voice-agent-v5/infrastructure/puppygraph
docker-compose up -d
```

**Expected Output:**
```
Creating puppygraph ... done
```

### 1.2 Wait for Startup

```bash
# Wait 30 seconds for container to initialize
sleep 30

# Check container is running
docker ps | grep puppygraph
```

**Expected:** Container running on ports 8081 (HTTP) and 8182 (Gremlin)

### 1.3 Verify HTTP Endpoint

```bash
curl http://localhost:8081/health
```

**Expected:** `{"status":"healthy"}` or similar

### 1.4 Apply Schema

```bash
curl -X POST http://localhost:8081/schema \
  -H "Content-Type: application/json" \
  -d @schema.json
```

**Expected:** Success response with schema applied

### 1.5 Verify Schema Applied

```bash
curl http://localhost:8081/schema
```

**Expected:** JSON showing `Entity` vertex and `RELATES_TO` edge definitions

### 1.6 Test Gremlin Endpoint

```bash
# Simple connectivity test
curl -X POST http://localhost:8182/gremlin \
  -H "Content-Type: application/json" \
  -d '{"gremlin": "g.V().count()"}'
```

**Expected:** `{"result":{"data":{"@type":"g:List","@value":[{"@type":"g:Int64","@value":0}]}}}`

### 1.7 Update .env

```bash
cd /home/matheus/repos/voice-agent-v5

# Add/update these lines
echo "RAG2_GRAPH_ENABLED=true" >> .env
echo "RAG2_PUPPYGRAPH_URL=http://localhost:8182" >> .env
```

### 1.8 Verify Python Connection

```python
# Quick test
cd /home/matheus/repos/voice-agent-v5
source venv/bin/activate
python -c "
from voice_agent.rag2.graph_search import get_graph_searcher
from voice_agent.supabase_client import get_supabase_client

supabase = get_supabase_client()
searcher = get_graph_searcher(supabase)
print(f'Graph searcher type: {type(searcher).__name__}')
print(f'PuppyGraph URL: {searcher.puppygraph_url if hasattr(searcher, \"puppygraph_url\") else \"N/A\"}')
"
```

---

## Verification Checklist

```
[ ] docker-compose up -d succeeds
[ ] Container running (docker ps shows puppygraph)
[ ] HTTP endpoint responds (curl :8081/health)
[ ] Schema applied (curl POST :8081/schema)
[ ] Gremlin endpoint responds (curl :8182/gremlin)
[ ] .env updated with RAG2_GRAPH_ENABLED=true
[ ] Python can import and connect
```

---

## Troubleshooting

### Container won't start
```bash
docker-compose logs puppygraph
```

### Port already in use
```bash
# Check what's using ports
lsof -i :8081
lsof -i :8182

# Kill if needed
kill -9 <PID>
```

### Schema apply fails
- Check schema.json syntax
- Verify container is fully initialized (wait longer)
- Check container logs: `docker logs puppygraph`

### Connection refused from Python
- Verify container is running
- Check firewall rules
- Ensure RAG2_PUPPYGRAPH_URL is correct

---

## Unblocks

After this phase completes:
- [x] Phase 2: Graph Channel E2E Test
- [x] Phase 2: Entity Extraction E2E Test (needs graph for full test)
- [x] Phase 4: Triple-Hybrid Integration Test

---

## Rollback

If PuppyGraph causes issues:
```bash
# Stop container
cd infrastructure/puppygraph
docker-compose down

# Disable in .env
sed -i 's/RAG2_GRAPH_ENABLED=true/RAG2_GRAPH_ENABLED=false/' .env
```

SQL fallback in `graph_search.py` will continue to work.

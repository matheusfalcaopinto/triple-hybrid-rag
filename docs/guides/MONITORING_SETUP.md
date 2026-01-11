# Monitoring Setup Guide

This guide covers how to run Prometheus and Grafana locally to monitor the Voice Agent v4 service.

---

## Quick Start

### 1. Start Voice Agent v4
```bash
# In terminal 1
python scripts/run_local.py
```

The service exposes metrics at `http://localhost:5050/metrics`.

### 2. Start Prometheus & Grafana
```bash
# In terminal 2
docker-compose up -d
```

This starts:
- **Prometheus** on http://localhost:9090
- **Grafana** on http://localhost:3000

### 3. Access Grafana

1. Open http://localhost:3000
2. Login with:
   - **Username**: `admin`
   - **Password**: `admin`
3. Navigate to **Dashboards** → **Voice Agent v4 - Performance & Health**

---

## What You'll See

### Pre-configured Dashboard

The Grafana dashboard includes:

#### **Error Monitoring**
- Cartesia TTS errors, rate limits, timeouts
- OpenAI stream errors and timeouts

#### **Latency Metrics**
- Time to first audio (p50, p90, p95, p99)
- Turn latency percentiles

#### **Application Health**
- Late event drops
- Duplicate final drops

### Prometheus Alerts

The following alerts are pre-configured in `monitoring/prometheus/alerts.yml`:

| Alert | Threshold | Severity |
|-------|-----------|----------|
| CartesiaTTSRateLimitHigh | >5/min for 2m | warning |
| CartesiaTTSRateLimitCritical | >20/min for 1m | critical |
| CartesiaTTSErrorRateHigh | >10/min for 3m | warning |
| OpenAIStreamErrorRateHigh | >5/min for 3m | warning |
| OpenAIStreamTimeoutHigh | >3/min for 3m | critical |
| TimeToFirstAudioSlow | p95 >2000ms for 5m | warning |

---

## Directory Structure

```
monitoring/
├── prometheus/
│   ├── prometheus.yml    # Scrape config (targets localhost:5050/metrics)
│   └── alerts.yml        # Alert rules
└── grafana/
    ├── provisioning/
    │   ├── datasources/
    │   │   └── prometheus.yml    # Auto-configure Prometheus datasource
    │   └── dashboards/
    │       └── default.yml       # Auto-load dashboards
    └── dashboards/
        └── voice-agent-v4.json   # Pre-built dashboard
```

---

## Customization

### Add More Dashboards

1. Create dashboards in Grafana UI
2. Export as JSON
3. Save to `monitoring/grafana/dashboards/`
4. Restart Grafana: `docker-compose restart grafana`

### Modify Alert Thresholds

Edit `monitoring/prometheus/alerts.yml`, then reload Prometheus:
```bash
curl -X POST http://localhost:9090/-/reload
```

### Change Scrape Interval

Edit `monitoring/prometheus/prometheus.yml`:
```yaml
scrape_configs:
  - job_name: 'voice_agent_v4'
    scrape_interval: 5s  # Change this
```

Then restart: `docker-compose restart prometheus`

---

## Troubleshooting

### Prometheus shows "DOWN" for voice_agent_v4 target

**Cause**: Voice agent isn't running or not accessible from Docker.

**Fix**: 
1. Verify voice agent is running: `curl http://localhost:5050/metrics`
2. Check Docker can reach host: `docker exec voice_agent_prometheus ping host.docker.internal`

### No metrics appearing in Grafana

**Cause**: No calls have been made yet, so metrics are empty.

**Fix**: Place a test call to generate metrics:
```bash
python scripts/make_call_v4.py
```

### Grafana shows "No data"

**Cause**: Prometheus not collecting data or wrong time range.

**Fix**:
1. Check Prometheus targets: http://localhost:9090/targets
2. Verify metrics exist: http://localhost:9090/graph (query: `cartesia_tts_error_total`)
3. Adjust Grafana time range to "Last 15 minutes"

---

## Stopping Monitoring Stack

```bash
# Stop containers but keep data
docker-compose stop

# Stop and remove containers (keeps volumes)
docker-compose down

# Stop and remove everything including data
docker-compose down -v
```

---

## Production Deployment

For production use:

1. **Change Grafana password**: Set `GF_SECURITY_ADMIN_PASSWORD` in `docker-compose.yml`
2. **Enable HTTPS**: Configure reverse proxy (nginx/Caddy) in front of Grafana
3. **Persistent storage**: Volumes are already configured (`prometheus_data`, `grafana_data`)
4. **Add Alertmanager**: Uncomment alerting section in `prometheus.yml` and add alertmanager service
5. **Update scrape target**: Change `host.docker.internal:5050` to your production endpoint

---

## Useful Commands

```bash
# View Prometheus logs
docker logs -f voice_agent_prometheus

# View Grafana logs
docker logs -f voice_agent_grafana

# Restart services
docker-compose restart

# Check metrics endpoint directly
curl http://localhost:5050/metrics

# Query Prometheus API
curl 'http://localhost:9090/api/v1/query?query=up'
```

---

## Next Steps

- Explore **Prometheus UI** at http://localhost:9090 to write custom queries
- Add **custom panels** to Grafana dashboards for your specific metrics
- Set up **Alertmanager** for email/Slack notifications
- Export metrics to external systems (Datadog, New Relic, etc.) via Prometheus remote write

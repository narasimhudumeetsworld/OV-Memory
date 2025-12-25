# 🚀 Deployment Guide

**Om Vinayaka 🙏**

Production deployment guide for OV-Memory across all platforms.

---

## 👋 Prerequisites

### System Requirements
- **CPU**: 2+ cores (4+ recommended)
- **RAM**: 4GB minimum (8GB+ for large graphs)
- **Storage**: SSD recommended
- **Network**: Low-latency connection

### Language Runtimes
- **Node.js**: 18.0+
- **Python**: 3.9+
- **C**: GCC 9.0+ or Clang 11.0+
- **Go**: 1.21+
- **Rust**: 1.70+

---

## 🚋 Docker Deployment

### Dockerfile (Production)
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy application
COPY python/ .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import ov_memory; print('ok')"

# Run application
CMD ["python", "ov_memory.py"]
```

### Docker Compose
```yaml
version: '3.8'

services:
  ov-memory:
    build: .
    container_name: ov-memory-prod
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - LOG_LEVEL=INFO
      - MAX_NODES=100000
      - MAX_SESSION_TIME=3600
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Building and Running
```bash
# Build image
docker build -t ov-memory:1.0.0 .

# Run container
docker run -d \
  --name ov-memory \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  ov-memory:1.0.0

# Docker Compose
docker-compose up -d
```

---

## ☁️ Cloud Deployment

### AWS (EC2)
```bash
#!/bin/bash

# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Python
sudo apt-get install -y python3.11 python3-pip

# Clone repository
git clone https://github.com/narasimhudumeetsworld/OV-Memory.git
cd OV-Memory/python

# Install dependencies
pip install -r requirements.txt

# Run application
python ov_memory.py
```

### Google Cloud Run
```bash
# Deploy to Cloud Run
gcloud run deploy ov-memory \
  --source . \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --timeout 3600
```

### Azure Container Instances
```bash
# Create container image
az acr build --registry myregistry \
  --image ov-memory:1.0.0 .

# Deploy
az container create \
  --resource-group mygroup \
  --name ov-memory \
  --image myregistry.azurecr.io/ov-memory:1.0.0 \
  --memory 2 \
  --cpu 1
```

### Kubernetes (K8s)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ov-memory
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ov-memory
  template:
    metadata:
      labels:
        app: ov-memory
    spec:
      containers:
      - name: ov-memory
        image: ov-memory:1.0.0
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
---
apiVersion: v1
kind: Service
metadata:
  name: ov-memory-service
spec:
  selector:
    app: ov-memory
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

```bash
# Deploy to Kubernetes
kubectl apply -f k8s-deployment.yaml
```

---

## 💾 Production Configuration

### Environment Variables
```bash
# Core Settings
export LOG_LEVEL=INFO
export MAX_NODES=100000
export MAX_EMBEDDING_DIM=768
export MAX_SESSION_TIME=3600

# Performance
export NUM_WORKERS=4
export CACHE_SIZE=10000
export BATCH_SIZE=32

# Security
export API_KEY=your-secret-key
export JWT_SECRET=your-jwt-secret
export ENABLE_SSL=true

# Monitoring
export PROMETHEUS_ENABLED=true
export SENTRY_DSN=your-sentry-dsn
```

### Configuration File
```yaml
# config.yaml
server:
  host: 0.0.0.0
  port: 8000
  workers: 4
  timeout: 30

memory:
  max_nodes: 100000
  max_embedding_dim: 768
  max_session_time: 3600

performance:
  cache_enabled: true
  cache_size: 10000
  batch_size: 32
  num_workers: 4

security:
  enable_authentication: true
  enable_ssl: true
  ssl_cert: /etc/ssl/certs/cert.pem
  ssl_key: /etc/ssl/private/key.pem

monitoring:
  prometheus_enabled: true
  sentry_enabled: true
  log_level: INFO
```

---

## 📊 Monitoring & Logging

### Logging Setup
```python
import logging
import logging.handlers

# Configure logging
logger = logging.getLogger('ov_memory')
logger.setLevel(logging.INFO)

# File handler
file_handler = logging.handlers.RotatingFileHandler(
    'ov_memory.log',
    maxBytes=10485760,  # 10MB
    backupCount=5
)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
```

### Prometheus Metrics
```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
node_add_total = Counter('ov_memory_nodes_added', 'Total nodes added')
edge_add_total = Counter('ov_memory_edges_added', 'Total edges added')
query_duration = Histogram('ov_memory_query_duration_seconds', 'Query duration')
graph_size = Gauge('ov_memory_graph_size', 'Current graph size')

# Use metrics
node_add_total.inc()
with query_duration.time():
    # Execute query
    pass
```

### Health Checks
```bash
# Health check endpoint
curl -X GET http://localhost:8000/health

# Response
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-12-25T10:00:00Z",
  "uptime": 3600,
  "nodes": 1000,
  "memory_mb": 128
}
```

---

## 🔐 Security Checklist

- [ ] Enable HTTPS/TLS
- [ ] Configure firewall rules
- [ ] Set up API authentication
- [ ] Enable rate limiting
- [ ] Configure CORS properly
- [ ] Sanitize inputs
- [ ] Use secrets management
- [ ] Enable audit logging
- [ ] Regular backups
- [ ] Security scanning

---

## 💪 Scaling Strategies

### Vertical Scaling
```bash
# Increase resources
# - CPU cores
# - RAM
# - Disk space

# Environment variables
export NUM_WORKERS=8  # Increase from 4
export CACHE_SIZE=50000  # Increase from 10000
```

### Horizontal Scaling
```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ov-memory-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ov-memory
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

---

## 🔁 Backup & Recovery

### Backup Strategy
```bash
# Daily backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
TARGET="backups/ov_memory_$DATE.tar.gz"

tar -czf $TARGET data/ logs/

# Upload to S3
aws s3 cp $TARGET s3://my-backup-bucket/
```

### Recovery
```bash
# Restore from backup
aws s3 cp s3://my-backup-bucket/ov_memory_20251225_100000.tar.gz .
tar -xzf ov_memory_20251225_100000.tar.gz

# Restart service
sudo systemctl restart ov-memory
```

---

## 💡 Performance Tuning

### System Optimization
```bash
# Increase file descriptors
ulimit -n 65536

# Optimize TCP
sysctl -w net.core.somaxconn=65535
sysctl -w net.ipv4.tcp_max_syn_backlog=65535

# Memory optimization
sysctl -w vm.swappiness=10
```

### Application Tuning
```python
# Enable connection pooling
from concurrent.futures import ThreadPoolExecutor
pool = ThreadPoolExecutor(max_workers=8)

# Enable caching
from functools import lru_cache
@lru_cache(maxsize=1000)
def get_node(node_id):
    pass
```

---

## 🗣️ Troubleshooting

### High Memory Usage
```bash
# Check current usage
free -h
ps aux | grep ov_memory

# Reduce max_nodes
export MAX_NODES=50000

# Enable memory profiling
python -m memory_profiler script.py
```

### Slow Queries
```bash
# Enable query logging
export LOG_LEVEL=DEBUG

# Check slow query log
tail -f logs/slow_queries.log

# Profile code
python -m cProfile -s cumulative script.py
```

### Connection Issues
```bash
# Test connectivity
curl -v http://localhost:8000/health

# Check firewall
sudo ufw status

# Check listening ports
ss -tlnp
```

---

## 📗 Maintenance

### Regular Tasks
- **Daily**: Monitor logs, check health
- **Weekly**: Backup data, analyze metrics
- **Monthly**: Update dependencies, security audit
- **Quarterly**: Performance review, capacity planning
- **Annually**: Major version upgrade, security assessment

### Update Procedures
```bash
# Download latest version
wget https://github.com/narasimhudumeetsworld/OV-Memory/releases/latest

# Backup current version
cp -r /opt/ov-memory /opt/ov-memory.backup

# Install new version
tar -xzf ov-memory-1.1.0.tar.gz -C /opt/

# Test
./ov-memory test

# Restart service
sudo systemctl restart ov-memory
```

---

## 🐧 Support

- **Documentation**: [README.md](README.md)
- **Issues**: [GitHub Issues](https://github.com/narasimhudumeetsworld/OV-Memory/issues)
- **Community**: [GitHub Discussions](https://github.com/narasimhudumeetsworld/OV-Memory/discussions)

---

**Om Vinayaka 🙏**

Version: 1.0.0  
Last Updated: December 25, 2025

//! OV-Memory Production Implementation in Rust
//! Complete with error handling, structured logging, metrics collection, and circuit breaker

use std::collections::HashMap;
use std::sync::{Arc, RwLock, Mutex};
use std::time::{SystemTime, Duration};
use serde_json::json;

// ========== STRUCTURED LOGGING ==========

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Debug = 0,
    Info = 1,
    Warning = 2,
    Error = 3,
    Critical = 4,
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogLevel::Debug => write!(f, "DEBUG"),
            LogLevel::Info => write!(f, "INFO"),
            LogLevel::Warning => write!(f, "WARNING"),
            LogLevel::Error => write!(f, "ERROR"),
            LogLevel::Critical => write!(f, "CRITICAL"),
        }
    }
}

pub struct StructuredLogger {
    log_level: LogLevel,
    logs: Arc<Mutex<Vec<String>>>,
}

impl StructuredLogger {
    pub fn new(log_level: LogLevel) -> Self {
        StructuredLogger {
            log_level,
            logs: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn log(&self, level: LogLevel, message: &str, fields: serde_json::Value) {
        if level < self.log_level {
            return;
        }

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let log_entry = json!({
            "timestamp": timestamp,
            "level": level.to_string(),
            "message": message,
            "fields": fields,
        });

        let json_str = log_entry.to_string();
        println!("{}", json_str);

        if let Ok(mut logs) = self.logs.lock() {
            logs.push(json_str);
        }
    }
}

// ========== CUSTOM ERRORS ==========

#[derive(Debug, Clone)]
pub enum OVMemoryError {
    InvalidData { message: String, context: HashMap<String, String> },
    MemoryCorruption { message: String, context: HashMap<String, String> },
    ResourceExhaustion { message: String, context: HashMap<String, String> },
    Timeout { message: String, context: HashMap<String, String> },
}

impl std::fmt::Display for OVMemoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OVMemoryError::InvalidData { message, .. } => write!(f, "[InvalidData] {}", message),
            OVMemoryError::MemoryCorruption { message, .. } => write!(f, "[MemoryCorruption] {}", message),
            OVMemoryError::ResourceExhaustion { message, .. } => write!(f, "[ResourceExhaustion] {}", message),
            OVMemoryError::Timeout { message, .. } => write!(f, "[Timeout] {}", message),
        }
    }
}

impl std::error::Error for OVMemoryError {}

// ========== METRICS COLLECTION ==========

pub struct MetricsCollector {
    queries_processed: Arc<RwLock<i64>>,
    total_latency: Arc<RwLock<f64>>,
    error_count: Arc<RwLock<HashMap<String, i64>>>,
    p50_latency: Arc<RwLock<f64>>,
    p95_latency: Arc<RwLock<f64>>,
    p99_latency: Arc<RwLock<f64>>,
    max_latency: Arc<RwLock<f64>>,
    start_time: SystemTime,
}

impl MetricsCollector {
    pub fn new() -> Self {
        MetricsCollector {
            queries_processed: Arc::new(RwLock::new(0)),
            total_latency: Arc::new(RwLock::new(0.0)),
            error_count: Arc::new(RwLock::new(HashMap::new())),
            p50_latency: Arc::new(RwLock::new(0.0)),
            p95_latency: Arc::new(RwLock::new(0.0)),
            p99_latency: Arc::new(RwLock::new(0.0)),
            max_latency: Arc::new(RwLock::new(0.0)),
            start_time: SystemTime::now(),
        }
    }

    pub fn record_latency(&self, latency_ms: f64) {
        if let Ok(mut queries) = self.queries_processed.write() {
            *queries += 1;
        }
        if let Ok(mut total) = self.total_latency.write() {
            *total += latency_ms;
        }
        if let Ok(mut max) = self.max_latency.write() {
            if latency_ms > *max {
                *max = latency_ms;
            }
        }
    }

    pub fn record_error(&self, error_type: &str) {
        if let Ok(mut errors) = self.error_count.write() {
            *errors.entry(error_type.to_string()).or_insert(0) += 1;
        }
    }

    pub fn get_metrics(&self) -> serde_json::Value {
        let queries = self.queries_processed.read().unwrap_or(0);
        let total_lat = self.total_latency.read().unwrap_or(0.0);
        let uptime = self.start_time.elapsed()
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        let qps = if uptime > 0.0 { queries as f64 / uptime } else { 0.0 };
        let avg_lat = if queries > 0 { total_lat / queries as f64 } else { 0.0 };
        let errors = self.error_count.read().unwrap_or_default();
        let total_errors: i64 = errors.values().sum();
        let error_rate = if queries > 0 { (total_errors as f64 / queries as f64) * 100.0 } else { 0.0 };

        json!({
            "queries_processed": queries,
            "qps": qps,
            "avg_latency_ms": avg_lat,
            "p50_latency_ms": self.p50_latency.read().unwrap_or(0.0),
            "p95_latency_ms": self.p95_latency.read().unwrap_or(0.0),
            "p99_latency_ms": self.p99_latency.read().unwrap_or(0.0),
            "max_latency_ms": self.max_latency.read().unwrap_or(0.0),
            "error_count": total_errors,
            "error_rate_pct": error_rate,
            "error_breakdown": errors,
            "uptime_seconds": uptime,
        })
    }
}

// ========== CIRCUIT BREAKER ==========

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitBreakerState {
    Closed,
    Open,
    HalfOpen,
}

pub struct CircuitBreaker {
    state: Arc<RwLock<CircuitBreakerState>>,
    failure_count: Arc<RwLock<i64>>,
    success_count: Arc<RwLock<i64>>,
    failure_threshold: i64,
    success_threshold: i64,
    timeout: Duration,
    last_failure_time: Arc<RwLock<SystemTime>>,
}

impl CircuitBreaker {
    pub fn new(failure_threshold: i64, success_threshold: i64, timeout_secs: u64) -> Self {
        CircuitBreaker {
            state: Arc::new(RwLock::new(CircuitBreakerState::Closed)),
            failure_count: Arc::new(RwLock::new(0)),
            success_count: Arc::new(RwLock::new(0)),
            failure_threshold,
            success_threshold,
            timeout: Duration::from_secs(timeout_secs),
            last_failure_time: Arc::new(RwLock::new(SystemTime::now())),
        }
    }

    pub fn call<F, T>(&self, f: F) -> Result<T, OVMemoryError>
    where
        F: FnOnce() -> Result<T, OVMemoryError>,
    {
        let state = *self.state.read().unwrap_or(&CircuitBreakerState::Closed);

        if state == CircuitBreakerState::Open {
            if let Ok(last_failure) = self.last_failure_time.read() {
                if last_failure.elapsed().unwrap_or(Duration::from_secs(0)) > self.timeout {
                    if let Ok(mut s) = self.state.write() {
                        *s = CircuitBreakerState::HalfOpen;
                    }
                } else {
                    let mut context = HashMap::new();
                    context.insert("state".to_string(), "OPEN".to_string());
                    return Err(OVMemoryError::ResourceExhaustion {
                        message: "Circuit breaker is OPEN".to_string(),
                        context,
                    });
                }
            }
        }

        match f() {
            Ok(result) => {
                if let Ok(mut count) = self.failure_count.write() {
                    *count = 0;
                }
                if let Ok(mut s) = self.state.read() {
                    if *s == CircuitBreakerState::HalfOpen {
                        if let Ok(mut success) = self.success_count.write() {
                            *success += 1;
                            if *success >= self.success_threshold {
                                if let Ok(mut state) = self.state.write() {
                                    *state = CircuitBreakerState::Closed;
                                }
                            }
                        }
                    }
                }
                Ok(result)
            }
            Err(e) => {
                if let Ok(mut count) = self.failure_count.write() {
                    *count += 1;
                    if *count >= self.failure_threshold {
                        if let Ok(mut state) = self.state.write() {
                            *state = CircuitBreakerState::Open;
                        }
                    }
                }
                if let Ok(mut last_time) = self.last_failure_time.write() {
                    *last_time = SystemTime::now();
                }
                Err(e)
            }
        }
    }

    pub fn get_state(&self) -> CircuitBreakerState {
        *self.state.read().unwrap_or(&CircuitBreakerState::Closed)
    }
}

// ========== MEMORY NODE ==========

#[derive(Debug, Clone)]
pub struct MemoryNode {
    pub id: String,
    pub embedding: Vec<f64>,
    pub text: String,
    pub centrality: f64,
    pub importance: f64,
    pub age: i64,
    pub created_at: SystemTime,
}

// ========== PRODUCTION OV-MEMORY ==========

pub struct OVMemoryProduction {
    embedding_dim: usize,
    max_nodes: usize,
    nodes: Arc<RwLock<HashMap<String, MemoryNode>>>,
    logger: StructuredLogger,
    metrics: MetricsCollector,
    circuit_breaker: CircuitBreaker,
    enable_monitoring: bool,
    error_logs: Arc<Mutex<Vec<String>>>,
    max_error_log_size: usize,
}

impl OVMemoryProduction {
    pub fn new(embedding_dim: usize, max_nodes: usize, enable_monitoring: bool) -> Self {
        OVMemoryProduction {
            embedding_dim,
            max_nodes,
            nodes: Arc::new(RwLock::new(HashMap::new())),
            logger: StructuredLogger::new(LogLevel::Info),
            metrics: MetricsCollector::new(),
            circuit_breaker: CircuitBreaker::new(5, 3, 30),
            enable_monitoring,
            error_logs: Arc::new(Mutex::new(Vec::new())),
            max_error_log_size: 1000,
        }
    }

    // ========== INPUT VALIDATION ==========

    fn validate_embedding(&self, embedding: &[f64]) -> Result<(), OVMemoryError> {
        if embedding.len() != self.embedding_dim {
            let mut context = HashMap::new();
            context.insert("expected".to_string(), self.embedding_dim.to_string());
            context.insert("got".to_string(), embedding.len().to_string());
            return Err(OVMemoryError::InvalidData {
                message: "Embedding dimension mismatch".to_string(),
                context,
            });
        }

        for (idx, &val) in embedding.iter().enumerate() {
            if !val.is_finite() {
                let mut context = HashMap::new();
                context.insert("index".to_string(), idx.to_string());
                context.insert("value".to_string(), val.to_string());
                return Err(OVMemoryError::InvalidData {
                    message: "Embedding contains NaN or Inf".to_string(),
                    context,
                });
            }
        }

        Ok(())
    }

    fn validate_text(&self, text: &str) -> Result<(), OVMemoryError> {
        if text.is_empty() {
            return Err(OVMemoryError::InvalidData {
                message: "Text cannot be empty".to_string(),
                context: HashMap::new(),
            });
        }
        if text.len() > 1_000_000 {
            let mut context = HashMap::new();
            context.insert("length".to_string(), text.len().to_string());
            return Err(OVMemoryError::InvalidData {
                message: "Text exceeds max length".to_string(),
                context,
            });
        }
        Ok(())
    }

    fn validate_resources(&self) -> Result<(), OVMemoryError> {
        if let Ok(nodes) = self.nodes.read() {
            if nodes.len() >= self.max_nodes {
                let mut context = HashMap::new();
                context.insert("current".to_string(), nodes.len().to_string());
                context.insert("max".to_string(), self.max_nodes.to_string());
                return Err(OVMemoryError::ResourceExhaustion {
                    message: "Max nodes reached".to_string(),
                    context,
                });
            }
        }
        Ok(())
    }

    // ========== CORE OPERATIONS ==========

    pub fn add_memory(
        &self,
        embedding: Vec<f64>,
        text: String,
        centrality: f64,
        node_id: Option<String>,
    ) -> Result<String, OVMemoryError> {
        let start = SystemTime::now();

        // Validation
        self.validate_embedding(&embedding)?;
        self.validate_text(&text)?;
        self.validate_resources()?;

        // Circuit breaker
        let id = self.circuit_breaker.call(|| {
            let mut nodes = self.nodes.write()
                .map_err(|_| OVMemoryError::MemoryCorruption {
                    message: "Lock poisoned".to_string(),
                    context: HashMap::new(),
                })?;

            let node_id = node_id.unwrap_or_else(|| {
                format!("node_{}_{}" ,
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_nanos())
                        .unwrap_or(0),
                    nodes.len()
                )
            });

            let node = MemoryNode {
                id: node_id.clone(),
                embedding,
                text,
                centrality,
                importance: 1.0,
                age: 0,
                created_at: SystemTime::now(),
            };

            nodes.insert(node_id.clone(), node);
            Ok(node_id)
        })?;

        let latency_ms = start.elapsed()
            .map(|d| d.as_secs_f64() * 1000.0)
            .unwrap_or(0.0);

        self.metrics.record_latency(latency_ms);
        self.logger.log(
            LogLevel::Info,
            "Memory added",
            json!({
                "node_id": &id,
                "latency_ms": latency_ms,
            }),
        );

        Ok(id)
    }

    pub fn get_memory(&self, node_id: &str) -> Result<MemoryNode, OVMemoryError> {
        let nodes = self.nodes.read()
            .map_err(|_| OVMemoryError::MemoryCorruption {
                message: "Lock poisoned".to_string(),
                context: HashMap::new(),
            })?;

        nodes.get(node_id)
            .cloned()
            .ok_or_else(|| {
                let mut context = HashMap::new();
                context.insert("node_id".to_string(), node_id.to_string());
                OVMemoryError::InvalidData {
                    message: "Node not found".to_string(),
                    context,
                }
            })
    }

    // ========== HEALTH & METRICS ==========

    pub fn get_health_status(&self) -> serde_json::Value {
        let metrics = self.metrics.get_metrics();
        let error_rate = metrics["error_rate_pct"].as_f64().unwrap_or(0.0);

        let status = if error_rate > 10.0 {
            "CRITICAL"
        } else if error_rate > 5.0 {
            "WARNING"
        } else {
            "HEALTHY"
        };

        json!({
            "status": status,
            "error_rate_pct": error_rate,
            "metrics": metrics,
            "timestamp": SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        })
    }

    pub fn get_metrics(&self) -> serde_json::Value {
        self.metrics.get_metrics()
    }
}

// ========== EXAMPLE USAGE ==========

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_memory() {
        let memory = OVMemoryProduction::new(768, 10000, true);
        let embedding = vec![0.5; 768];
        let result = memory.add_memory(embedding, "test".to_string(), 0.9, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_invalid_embedding() {
        let memory = OVMemoryProduction::new(768, 10000, true);
        let embedding = vec![0.5; 100]; // Wrong size
        let result = memory.add_memory(embedding, "test".to_string(), 0.9, None);
        assert!(result.is_err());
    }
}

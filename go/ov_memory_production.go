package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"sync"
	"time"
)

// ========== STRUCTURED LOGGING ==========

type LogLevel int

const (
	DEBUG LogLevel = iota
	INFO
	WARNING
	ERROR
	CRITICAL
)

func (l LogLevel) String() string {
	names := []string{"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
	return names[l]
}

type StructuredLogger struct {
	mu       sync.Mutex
	logLevel LogLevel
}

func (sl *StructuredLogger) Log(level LogLevel, message string, fields map[string]interface{}) {
	if level < sl.logLevel {
		return
	}

	sl.mu.Lock()
	defer sl.mu.Unlock()

	logEntry := map[string]interface{}{
		"timestamp": time.Now().Format(time.RFC3339),
		"level":     level.String(),
		"message":   message,
	}

	for k, v := range fields {
		logEntry[k] = v
	}

	jsonBytes, _ := json.Marshal(logEntry)
	fmt.Printf("%s\n", string(jsonBytes))
}

// ========== CUSTOM EXCEPTIONS ==========

type OVMemoryError struct {
	Type    string
	Message string
	Context map[string]interface{}
}

func (e OVMemoryError) Error() string {
	return fmt.Sprintf("[%s] %s", e.Type, e.Message)
}

func NewInvalidDataError(msg string, context map[string]interface{}) error {
	return OVMemoryError{
		Type:    "InvalidDataException",
		Message: msg,
		Context: context,
	}
}

func NewMemoryCorruptionError(msg string, context map[string]interface{}) error {
	return OVMemoryError{
		Type:    "MemoryCorruptionException",
		Message: msg,
		Context: context,
	}
}

func NewResourceExhaustionError(msg string, context map[string]interface{}) error {
	return OVMemoryError{
		Type:    "ResourceExhaustionException",
		Message: msg,
		Context: context,
	}
}

func NewTimeoutError(msg string, context map[string]interface{}) error {
	return OVMemoryError{
		Type:    "TimeoutException",
		Message: msg,
		Context: context,
	}
}

// ========== METRICS COLLECTION ==========

type MetricsCollector struct {
	mu                 sync.RWMutex
	queriesProcessed   int64
	totalLatency       float64
	errorCount         map[string]int64
	p50Latency         float64
	p95Latency         float64
	p99Latency         float64
	maxLatency         float64
	startTime          time.Time
	lastCollectionTime time.Time
}

func NewMetricsCollector() *MetricsCollector {
	return &MetricsCollector{
		errorCount:         make(map[string]int64),
		startTime:          time.Now(),
		lastCollectionTime: time.Now(),
	}
}

func (mc *MetricsCollector) RecordLatency(latencyMs float64) {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	mc.queriesProcessed++
	mc.totalLatency += latencyMs

	// Simple percentile tracking
	if latencyMs > mc.maxLatency {
		mc.maxLatency = latencyMs
	}
	if mc.p50Latency == 0 {
		mc.p50Latency = latencyMs
	}
	if mc.p95Latency == 0 {
		mc.p95Latency = latencyMs
	}
	if mc.p99Latency == 0 {
		mc.p99Latency = latencyMs
	}
}

func (mc *MetricsCollector) RecordError(errorType string) {
	mc.mu.Lock()
	defer mc.mu.Unlock()
	mc.errorCount[errorType]++
}

func (mc *MetricsCollector) GetMetrics() map[string]interface{} {
	mc.mu.RLock()
	defer mc.mu.RUnlock()

	uptime := time.Since(mc.startTime).Seconds()
	var qps float64
	if uptime > 0 {
		qps = float64(mc.queriesProcessed) / uptime
	}

	avgLatency := 0.0
	if mc.queriesProcessed > 0 {
		avgLatency = mc.totalLatency / float64(mc.queriesProcessed)
	}

	errorRate := 0.0
	totalErrors := int64(0)
	for _, count := range mc.errorCount {
		totalErrors += count
	}
	if mc.queriesProcessed > 0 {
		errorRate = float64(totalErrors) / float64(mc.queriesProcessed) * 100
	}

	return map[string]interface{}{
		"queries_processed":  mc.queriesProcessed,
		"qps":                qps,
		"avg_latency_ms":     avgLatency,
		"p50_latency_ms":     mc.p50Latency,
		"p95_latency_ms":     mc.p95Latency,
		"p99_latency_ms":     mc.p99Latency,
		"max_latency_ms":     mc.maxLatency,
		"error_count":        totalErrors,
		"error_rate_pct":     errorRate,
		"error_breakdown":    mc.errorCount,
		"uptime_seconds":     uptime,
	}
}

// ========== CIRCUIT BREAKER ==========

type CircuitBreakerState string

const (
	StateClosed CircuitBreakerState = "CLOSED"
	StateOpen   CircuitBreakerState = "OPEN"
	StateHalfOpen CircuitBreakerState = "HALF_OPEN"
)

type CircuitBreaker struct {
	mu              sync.RWMutex
	state           CircuitBreakerState
	failureCount    int64
	successCount    int64
	failureThreshold int64
	successThreshold int64
	timeout         time.Duration
	lastFailureTime  time.Time
}

func NewCircuitBreaker(failureThreshold, successThreshold int64, timeout time.Duration) *CircuitBreaker {
	return &CircuitBreaker{
		state:            StateClosed,
		failureThreshold: failureThreshold,
		successThreshold: successThreshold,
		timeout:          timeout,
	}
}

func (cb *CircuitBreaker) Call(fn func() error) error {
	cb.mu.RLock()
	state := cb.state
	cb.mu.RUnlock()

	if state == StateOpen {
		if time.Since(cb.lastFailureTime) > cb.timeout {
			cb.mu.Lock()
			cb.state = StateHalfOpen
			cb.successCount = 0
			cb.mu.Unlock()
		} else {
			return NewResourceExhaustionError("Circuit breaker is OPEN", map[string]interface{}{
				"state": StateOpen,
			})
		}
	}

	err := fn()

	cb.mu.Lock()
	defer cb.mu.Unlock()

	if err != nil {
		cb.failureCount++
		cb.lastFailureTime = time.Now()
		if cb.failureCount >= cb.failureThreshold {
			cb.state = StateOpen
		}
		return err
	}

	cb.failureCount = 0
	if cb.state == StateHalfOpen {
		cb.successCount++
		if cb.successCount >= cb.successThreshold {
			cb.state = StateClosed
			cb.failureCount = 0
		}
	}

	return nil
}

func (cb *CircuitBreaker) GetState() CircuitBreakerState {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	return cb.state
}

// ========== PRODUCTION OV-MEMORY ==========

type MemoryNode struct {
	ID         string
	Embedding  []float64
	Text       string
	Centrality float64
	Importance float64
	Age        int64 // seconds
	CreatedAt  time.Time
}

type OVMemoryProduction struct {
	mu                  sync.RWMutex
	embeddingDim        int
	maxNodes            int
	nodes               map[string]*MemoryNode
	centroids           []*MemoryNode
	logger              *StructuredLogger
	metrics             *MetricsCollector
	circuitBreaker      *CircuitBreaker
	enableMonitoring    bool
	queryTimeout        time.Duration
	maxErrorLogSize     int
	errorLogs           []map[string]interface{}
}

func NewOVMemoryProduction(embeddingDim, maxNodes int, enableMonitoring bool) *OVMemoryProduction {
	return &OVMemoryProduction{
		embeddingDim:     embeddingDim,
		maxNodes:         maxNodes,
		nodes:            make(map[string]*MemoryNode),
		centroids:        make([]*MemoryNode, 0),
		logger:           &StructuredLogger{logLevel: INFO},
		metrics:          NewMetricsCollector(),
		circuitBreaker:   NewCircuitBreaker(5, 3, 30*time.Second),
		enableMonitoring: enableMonitoring,
		queryTimeout:     30 * time.Second,
		maxErrorLogSize:  1000,
		errorLogs:        make([]map[string]interface{}, 0),
	}
}

// ========== INPUT VALIDATION ==========

func (om *OVMemoryProduction) validateEmbedding(embedding []float64) error {
	// Check dimension
	if len(embedding) != om.embeddingDim {
		return NewInvalidDataError(
			"Embedding dimension mismatch",
			map[string]interface{}{
				"expected": om.embeddingDim,
				"got":      len(embedding),
			},
		)
	}

	// Check for NaN/Inf
	for i, val := range embedding {
		if math.IsNaN(val) || math.IsInf(val, 0) {
			return NewInvalidDataError(
				"Embedding contains NaN or Inf",
				map[string]interface{}{
					"index": i,
					"value": val,
				},
			)
		}
	}

	return nil
}

func (om *OVMemoryProduction) validateText(text string) error {
	if text == "" {
		return NewInvalidDataError("Text cannot be empty", map[string]interface{}{})
	}
	if len(text) > 1000000 {
		return NewInvalidDataError("Text exceeds max length", map[string]interface{}{
			"length": len(text),
		})
	}
	return nil
}

func (om *OVMemoryProduction) validateResources() error {
	om.mu.RLock()
	defer om.mu.RUnlock()

	if len(om.nodes) >= om.maxNodes {
		return NewResourceExhaustionError(
			"Max nodes reached",
			map[string]interface{}{
				"current": len(om.nodes),
				"max":     om.maxNodes,
			},
		)
	}

	return nil
}

// ========== CORE OPERATIONS ==========

func (om *OVMemoryProduction) AddMemory(embedding []float64, text string, centrality float64, nodeID *string) (string, error) {
	start := time.Now()

	// Input validation
	if err := om.validateEmbedding(embedding); err != nil {
		om.metrics.RecordError("InvalidDataException")
		om.logError(err)
		return "", err
	}

	if err := om.validateText(text); err != nil {
		om.metrics.RecordError("InvalidDataException")
		om.logError(err)
		return "", err
	}

	if err := om.validateResources(); err != nil {
		om.metrics.RecordError("ResourceExhaustionException")
		om.logError(err)
		return "", err
	}

	// Circuit breaker check
	var id string
	err := om.circuitBreaker.Call(func() error {
		om.mu.Lock()
		defer om.mu.Unlock()

		if nodeID != nil {
			id = *nodeID
		} else {
			id = fmt.Sprintf("node_%d_%d", time.Now().UnixNano(), len(om.nodes))
		}

		node := &MemoryNode{
			ID:         id,
			Embedding:  embedding,
			Text:       text,
			Centrality: centrality,
			Importance: 1.0,
			Age:        0,
			CreatedAt:  time.Now(),
		}

		om.nodes[id] = node
		return nil
	})

	if err != nil {
		om.metrics.RecordError("CircuitBreakerOpen")
		return "", err
	}

	latencyMs := float64(time.Since(start).Milliseconds())
	om.metrics.RecordLatency(latencyMs)

	om.logger.Log(INFO, "Memory added", map[string]interface{}{
		"node_id":   id,
		"latency_ms": latencyMs,
	})

	return id, nil
}

func (om *OVMemoryProduction) GetMemory(nodeID string) (*MemoryNode, error) {
	om.mu.RLock()
	defer om.mu.RUnlock()

	node, exists := om.nodes[nodeID]
	if !exists {
		return nil, NewInvalidDataError("Node not found", map[string]interface{}{
			"node_id": nodeID,
		})
	}

	return node, nil
}

// ========== HEALTH & METRICS ==========

type HealthStatus struct {
	Status    string                 `json:"status"`
	ErrorRate float64                `json:"error_rate_pct"`
	Metrics   map[string]interface{} `json:"metrics"`
	Timestamp string                 `json:"timestamp"`
}

func (om *OVMemoryProduction) GetHealthStatus() HealthStatus {
	metrics := om.metrics.GetMetrics()
	errorRate := metrics["error_rate_pct"].(float64)

	status := "HEALTHY"
	if errorRate > 10 {
		status = "CRITICAL"
	} else if errorRate > 5 {
		status = "WARNING"
	}

	return HealthStatus{
		Status:    status,
		ErrorRate: errorRate,
		Metrics:   metrics,
		Timestamp: time.Now().Format(time.RFC3339),
	}
}

func (om *OVMemoryProduction) GetMetrics() map[string]interface{} {
	return om.metrics.GetMetrics()
}

// ========== ERROR TRACKING ==========

func (om *OVMemoryProduction) logError(err error) {
	om.mu.Lock()
	defer om.mu.Unlock()

	errorEntry := map[string]interface{}{
		"error":     err.Error(),
		"timestamp": time.Now().Format(time.RFC3339),
	}

	om.errorLogs = append(om.errorLogs, errorEntry)

	// Keep only recent errors
	if len(om.errorLogs) > om.maxErrorLogSize {
		om.errorLogs = om.errorLogs[1:]
	}

	om.logger.Log(ERROR, "Error logged", errorEntry)
}

// ========== EXAMPLE USAGE ==========

func ExampleUsage() {
	memory := NewOVMemoryProduction(768, 10000, true)

	// Add memory
	embedding := make([]float64, 768)
	for i := range embedding {
		embedding[i] = 0.5
	}

	nodeID, err := memory.AddMemory(embedding, "Sample text", 0.9, nil)
	if err != nil {
		log.Printf("Error adding memory: %v", err)
		return
	}

	// Retrieve memory
	node, err := memory.GetMemory(nodeID)
	if err != nil {
		log.Printf("Error retrieving memory: %v", err)
		return
	}

	fmt.Printf("Retrieved node: %s\n", node.ID)

	// Get health status
	health := memory.GetHealthStatus()
	fmt.Printf("Health status: %s\n", health.Status)

	// Get metrics
	metrics := memory.GetMetrics()
	fmt.Printf("Queries processed: %d\n", metrics["queries_processed"])
}

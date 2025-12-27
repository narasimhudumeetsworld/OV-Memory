#!/usr/bin/env python3
"""
OV-Memory v1.1: Production-Hardened Implementation
Comprehensive error handling, structured logging, and monitoring

Author: Vaibhav Prayaga
Date: December 27, 2025
"""

import json
import logging
import time
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
from enum import Enum


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

class LogLevel(Enum):
    """Log level enumeration"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class StructuredLogger:
    """Structured logging with context and metrics"""
    
    def __init__(self, name: str, level: LogLevel = LogLevel.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level.value)
        
        # Console handler with structured format
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.context = {}
    
    def set_context(self, **kwargs):
        """Set logging context"""
        self.context.update(kwargs)
    
    def _format_message(self, msg: str, **kwargs) -> str:
        """Format message with context"""
        context_str = " | ".join(f"{k}={v}" for k, v in {**self.context, **kwargs}.items())
        return f"{msg} | {context_str}" if context_str else msg
    
    def debug(self, msg: str, **kwargs):
        self.logger.debug(self._format_message(msg, **kwargs))
    
    def info(self, msg: str, **kwargs):
        self.logger.info(self._format_message(msg, **kwargs))
    
    def warning(self, msg: str, **kwargs):
        self.logger.warning(self._format_message(msg, **kwargs))
    
    def error(self, msg: str, exc_info: bool = False, **kwargs):
        self.logger.error(self._format_message(msg, **kwargs), exc_info=exc_info)
    
    def critical(self, msg: str, exc_info: bool = False, **kwargs):
        self.logger.critical(self._format_message(msg, **kwargs), exc_info=exc_info)


# ============================================================================
# MONITORING & METRICS
# ============================================================================

@dataclass
class MetricSnapshot:
    """Snapshot of system metrics"""
    timestamp: datetime
    queries_processed: int = 0
    total_memory_nodes: int = 0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    drift_detections: int = 0
    loop_preventions: int = 0
    redundancy_filters: int = 0
    errors_encountered: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'queries_processed': self.queries_processed,
            'total_memory_nodes': self.total_memory_nodes,
            'avg_latency_ms': round(self.avg_latency_ms, 2),
            'max_latency_ms': round(self.max_latency_ms, 2),
            'drift_detections': self.drift_detections,
            'loop_preventions': self.loop_preventions,
            'redundancy_filters': self.redundancy_filters,
            'errors_encountered': self.errors_encountered,
        }


class MetricsCollector:
    """Collect and track system metrics"""
    
    def __init__(self):
        self.queries_processed = 0
        self.latencies: List[float] = []
        self.drift_detections = 0
        self.loop_preventions = 0
        self.redundancy_filters = 0
        self.errors_encountered = 0
        self.snapshots: List[MetricSnapshot] = []
        self.lock_time = 0.0
        self.logger = StructuredLogger('MetricsCollector')
    
    def record_query(self, latency_ms: float):
        """Record a query completion"""
        self.queries_processed += 1
        self.latencies.append(latency_ms)
    
    def record_drift_detection(self):
        """Record a drift detection event"""
        self.drift_detections += 1
    
    def record_loop_prevention(self):
        """Record a loop prevention event"""
        self.loop_preventions += 1
    
    def record_redundancy_filter(self):
        """Record a redundancy filter event"""
        self.redundancy_filters += 1
    
    def record_error(self):
        """Record an error event"""
        self.errors_encountered += 1
    
    def take_snapshot(self, total_nodes: int) -> MetricSnapshot:
        """Take a metrics snapshot"""
        snapshot = MetricSnapshot(
            timestamp=datetime.now(),
            queries_processed=self.queries_processed,
            total_memory_nodes=total_nodes,
            avg_latency_ms=np.mean(self.latencies) if self.latencies else 0.0,
            max_latency_ms=np.max(self.latencies) if self.latencies else 0.0,
            drift_detections=self.drift_detections,
            loop_preventions=self.loop_preventions,
            redundancy_filters=self.redundancy_filters,
            errors_encountered=self.errors_encountered,
        )
        self.snapshots.append(snapshot)
        return snapshot
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        if self.queries_processed == 0:
            return {'status': 'UNKNOWN', 'message': 'No queries processed yet'}
        
        error_rate = self.errors_encountered / self.queries_processed
        
        if error_rate > 0.1:  # >10% error rate
            status = 'CRITICAL'
        elif error_rate > 0.05:  # >5% error rate
            status = 'WARNING'
        else:
            status = 'HEALTHY'
        
        return {
            'status': status,
            'error_rate': round(error_rate * 100, 2),
            'queries_processed': self.queries_processed,
            'errors': self.errors_encountered,
        }


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class OVMemoryException(Exception):
    """Base exception for OV-Memory"""
    def __init__(self, message: str, code: str = "UNKNOWN"):
        self.message = message
        self.code = code
        self.timestamp = datetime.now()
        super().__init__(self.message)


class InvalidDataException(OVMemoryException):
    """Exception for invalid input data"""
    def __init__(self, message: str):
        super().__init__(message, "INVALID_DATA")


class MemoryCorruptionException(OVMemoryException):
    """Exception for memory corruption detection"""
    def __init__(self, message: str):
        super().__init__(message, "MEMORY_CORRUPTION")


class ResourceExhaustionException(OVMemoryException):
    """Exception for resource exhaustion"""
    def __init__(self, message: str):
        super().__init__(message, "RESOURCE_EXHAUSTION")


class TimeoutException(OVMemoryException):
    """Exception for operation timeout"""
    def __init__(self, message: str):
        super().__init__(message, "TIMEOUT")


# ============================================================================
# CIRCUIT BREAKER
# ============================================================================

class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """Circuit breaker for fault tolerance"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        success_threshold: int = 2
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.logger = StructuredLogger('CircuitBreaker')
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                self.logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise OVMemoryException(
                    "Circuit breaker is OPEN. Service temporarily unavailable.",
                    "CIRCUIT_BREAKER_OPEN"
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed for recovery attempt"""
        if self.last_failure_time is None:
            return False
        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful operation"""
        self.failure_count = 0
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.success_count = 0
                self.logger.info("Circuit breaker CLOSED after recovery")
    
    def _on_failure(self):
        """Handle failed operation"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        self.success_count = 0
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            self.logger.warning(
                "Circuit breaker OPEN after failures",
                failure_count=self.failure_count
            )


# ============================================================================
# PRODUCTION-HARDENED OV-MEMORY
# ============================================================================

@dataclass
class MemoryNode:
    """Enhanced memory node with validation"""
    embedding: np.ndarray
    text: str
    timestamp: datetime
    centrality: float = 0.0
    recency: float = 0.0
    intrinsic_weight: float = 1.0
    access_count: int = 0
    last_access: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Validate node integrity"""
        if self.embedding is None or len(self.embedding) == 0:
            return False
        if not isinstance(self.text, str) or len(self.text) == 0:
            return False
        if not (0.0 <= self.centrality <= 1.0):
            return False
        if not (0.0 <= self.recency <= 1.0):
            return False
        if self.intrinsic_weight <= 0:
            return False
        return True


class OVMemoryProduction:
    """Production-hardened OV-Memory with comprehensive error handling"""
    
    def __init__(
        self,
        embedding_dim: int = 768,
        max_nodes: int = 1_000_000,
        enable_monitoring: bool = True,
        log_level: LogLevel = LogLevel.INFO
    ):
        self.embedding_dim = embedding_dim
        self.max_nodes = max_nodes
        self.enable_monitoring = enable_monitoring
        
        # Core storage
        self.memory: Dict[int, MemoryNode] = {}
        self.node_counter = 0
        
        # Monitoring
        self.metrics = MetricsCollector() if enable_monitoring else None
        self.logger = StructuredLogger('OVMemoryProduction', log_level)
        self.circuit_breaker = CircuitBreaker()
        self.error_log: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.operation_times: Dict[str, List[float]] = defaultdict(list)
        
        self.logger.info(
            "OV-Memory initialized",
            embedding_dim=embedding_dim,
            max_nodes=max_nodes,
            monitoring_enabled=enable_monitoring
        )
    
    def _validate_input(self, embedding: np.ndarray, text: str) -> None:
        """Validate input data"""
        if embedding is None:
            raise InvalidDataException("Embedding cannot be None")
        
        if not isinstance(embedding, np.ndarray):
            raise InvalidDataException(f"Expected np.ndarray, got {type(embedding)}")
        
        if len(embedding) != self.embedding_dim:
            raise InvalidDataException(
                f"Embedding dimension mismatch. Expected {self.embedding_dim}, "
                f"got {len(embedding)}"
            )
        
        if not np.isfinite(embedding).all():
            raise InvalidDataException("Embedding contains NaN or Inf values")
        
        if not isinstance(text, str):
            raise InvalidDataException(f"Text must be string, got {type(text)}")
        
        if len(text.strip()) == 0:
            raise InvalidDataException("Text cannot be empty")
    
    def _check_resource_limits(self) -> None:
        """Check if system resources are exhausted"""
        if len(self.memory) >= self.max_nodes:
            raise ResourceExhaustionException(
                f"Memory limit reached: {len(self.memory)} / {self.max_nodes}"
            )
    
    def _record_error(self, operation: str, error: Exception) -> None:
        """Record error for debugging"""
        error_record = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
        }
        self.error_log.append(error_record)
        
        if len(self.error_log) > 1000:  # Keep last 1000 errors
            self.error_log = self.error_log[-1000:]
        
        if self.metrics:
            self.metrics.record_error()
    
    def add_memory(
        self,
        embedding: np.ndarray,
        text: str,
        intrinsic_weight: float = 1.0,
        metadata: Optional[Dict] = None
    ) -> int:
        """Add memory node with error handling"""
        operation = "add_memory"
        start_time = time.time()
        
        try:
            # Validate inputs
            self._validate_input(embedding, text)
            if not (intrinsic_weight > 0):
                raise InvalidDataException("Intrinsic weight must be positive")
            
            # Check resources
            self._check_resource_limits()
            
            # Create node
            node_id = self.node_counter
            self.node_counter += 1
            
            node = MemoryNode(
                embedding=embedding.copy(),  # Always copy to prevent external modification
                text=text,
                timestamp=datetime.now(),
                intrinsic_weight=intrinsic_weight,
                metadata=metadata or {}
            )
            
            # Validate node
            if not node.validate():
                raise MemoryCorruptionException("Node validation failed")
            
            self.memory[node_id] = node
            
            # Record metrics
            elapsed = (time.time() - start_time) * 1000
            self.operation_times[operation].append(elapsed)
            if self.metrics:
                self.metrics.record_query(elapsed)
            
            self.logger.debug(
                "Memory added",
                node_id=node_id,
                text_length=len(text),
                latency_ms=round(elapsed, 2)
            )
            
            return node_id
        
        except OVMemoryException as e:
            self._record_error(operation, e)
            self.logger.error(f"OV-Memory error in {operation}: {e.message}")
            raise
        
        except Exception as e:
            self._record_error(operation, e)
            self.logger.error(f"Unexpected error in {operation}", exc_info=True)
            raise OVMemoryException(f"Failed to add memory: {str(e)}", "ADD_MEMORY_FAILED")
    
    def retrieve_memories(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        alpha: float = 0.75,
        timeout_seconds: float = 30.0
    ) -> List[Tuple[int, float, str]]:
        """Retrieve memories with error handling and timeout"""
        operation = "retrieve_memories"
        start_time = time.time()
        
        try:
            # Use circuit breaker
            return self.circuit_breaker.call(
                self._retrieve_memories_internal,
                query_embedding,
                top_k,
                alpha,
                timeout_seconds,
                start_time
            )
        
        except TimeoutException as e:
            self._record_error(operation, e)
            self.logger.error(f"Timeout in {operation}: {e.message}")
            raise
        
        except Exception as e:
            self._record_error(operation, e)
            self.logger.error(f"Error in {operation}", exc_info=True)
            raise OVMemoryException(
                f"Failed to retrieve memories: {str(e)}",
                "RETRIEVE_FAILED"
            )
    
    def _retrieve_memories_internal(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        alpha: float,
        timeout_seconds: float,
        start_time: float
    ) -> List[Tuple[int, float, str]]:
        """Internal retrieval with timeout checking"""
        try:
            self._validate_input(query_embedding, "query")
        except InvalidDataException as e:
            raise InvalidDataException(f"Invalid query embedding: {str(e)}")
        
        if not (1 <= top_k <= len(self.memory)):
            self.logger.warning(
                "top_k out of range, clamping",
                requested=top_k,
                available=len(self.memory)
            )
            top_k = min(top_k, len(self.memory))
        
        if not (0.0 <= alpha <= 1.0):
            raise InvalidDataException("Alpha must be between 0 and 1")
        
        results = []
        visited = set()
        
        for node_id, node in self.memory.items():
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                raise TimeoutException(
                    f"Retrieval timeout after {elapsed:.2f}s"
                )
            
            # Calculate similarity
            similarity = np.dot(query_embedding, node.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(node.embedding) + 1e-8
            )
            
            # Calculate priority
            recency = np.exp(-0.1 * (time.time() - node.timestamp.timestamp()))
            priority = (
                similarity ** alpha *
                max(0.1, node.centrality) ** (1 - alpha) *
                max(0.1, recency) ** (1 - alpha) *
                node.intrinsic_weight
            )
            
            results.append((node_id, priority, node.text))
        
        # Sort by priority and return top-k
        results.sort(key=lambda x: x[1], reverse=True)
        top_results = results[:top_k]
        
        # Record metrics
        elapsed = (time.time() - start_time) * 1000
        if self.metrics:
            self.metrics.record_query(elapsed)
        
        self.logger.debug(
            "Memories retrieved",
            count=len(top_results),
            latency_ms=round(elapsed, 2)
        )
        
        return top_results
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        if not self.metrics:
            return {'status': 'Monitoring disabled'}
        
        snapshot = self.metrics.take_snapshot(len(self.memory))
        health = self.metrics.get_health_status()
        
        return {
            'metrics': snapshot.to_dict(),
            'health': health,
            'operation_times': {
                op: {
                    'count': len(times),
                    'avg_ms': round(np.mean(times), 2),
                    'min_ms': round(np.min(times), 2),
                    'max_ms': round(np.max(times), 2),
                }
                for op, times in self.operation_times.items()
                if times
            },
            'memory_size': len(self.memory),
            'node_limit': self.max_nodes,
        }
    
    def get_error_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent errors"""
        return self.error_log[-limit:]
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status"""
        if not self.metrics:
            return {'status': 'UNKNOWN'}
        
        return self.metrics.get_health_status()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("OV-MEMORY v1.1: Production-Hardened Implementation")
    print("="*80 + "\n")
    
    try:
        # Initialize with monitoring
        memory = OVMemoryProduction(
            embedding_dim=768,
            max_nodes=10000,
            enable_monitoring=True,
            log_level=LogLevel.INFO
        )
        
        print("✅ OV-Memory initialized with production hardening")
        print("   - Error handling: Enabled")
        print("   - Monitoring: Enabled")
        print("   - Circuit breaker: Enabled")
        print("   - Logging: Enabled\n")
        
        # Add some test memories
        print("📝 Adding test memories...")
        embeddings = np.random.randn(5, 768).astype(np.float32)
        texts = [
            "The quick brown fox jumps over the lazy dog",
            "Machine learning is transforming AI",
            "Neural networks learn hierarchical representations",
            "Deep learning requires lots of data",
            "Natural language processing is challenging"
        ]
        
        for i, (embedding, text) in enumerate(zip(embeddings, texts)):
            node_id = memory.add_memory(
                embedding=embedding,
                text=text,
                intrinsic_weight=1.0 + i * 0.1
            )
            print(f"   Added memory {node_id}: {text[:50]}...")
        
        # Retrieve memories
        print("\n🔍 Retrieving memories...")
        query = np.random.randn(768).astype(np.float32)
        results = memory.retrieve_memories(query, top_k=3)
        
        for node_id, priority, text in results:
            print(f"   ID {node_id}: Priority {priority:.4f} - {text[:50]}...")
        
        # Get metrics
        print("\n📊 System Metrics:")
        metrics = memory.get_metrics()
        print(json.dumps(metrics, indent=2, default=str))
        
        # Get health
        print("\n💚 Health Status:")
        health = memory.get_health_status()
        print(json.dumps(health, indent=2, default=str))
        
        print("\n" + "="*80)
        print("✅ Production-hardened OV-Memory working correctly!")
        print("="*80 + "\n")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n" + "="*80 + "\n")

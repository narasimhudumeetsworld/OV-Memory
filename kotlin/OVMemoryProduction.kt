package com.ov.memory

import java.util.concurrent.*
import java.util.*
import kotlin.math.*
import java.time.Instant

// ========== LOGGING ==========

enum class LogLevel {
    DEBUG, INFO, WARNING, ERROR, CRITICAL
}

data class LogEntry(
    val timestamp: String = Instant.now().toString(),
    val level: LogLevel,
    val message: String,
    val fields: Map<String, Any?> = emptyMap()
) {
    fun toJson(): String {
        return """
            {
                "timestamp": "$timestamp",
                "level": "$level",
                "message": "$message",
                "fields": ${fieldsToJson()}
            }
        """.trimIndent()
    }

    private fun fieldsToJson(): String {
        val entries = fields.map { (k, v) -> "\"$k\": $v" }.joinToString(",")
        return "{$entries}"
    }
}

class StructuredLogger(private var logLevel: LogLevel = LogLevel.INFO) {
    private val queue = ConcurrentLinkedQueue<LogEntry>()

    fun log(level: LogLevel, message: String, fields: Map<String, Any?> = emptyMap()) {
        if (level.ordinal < logLevel.ordinal) return

        val entry = LogEntry(level = level, message = message, fields = fields)
        queue.offer(entry)
        println(entry.toJson())
    }

    fun getLogs(): List<LogEntry> = queue.toList()
}

// ========== EXCEPTIONS ==========

open class OVMemoryException(
    val type: String,
    message: String,
    val context: Map<String, Any?> = emptyMap()
) : RuntimeException("[$type] $message")

class InvalidDataException(
    message: String,
    context: Map<String, Any?> = emptyMap()
) : OVMemoryException("InvalidDataException", message, context)

class MemoryCorruptionException(
    message: String,
    context: Map<String, Any?> = emptyMap()
) : OVMemoryException("MemoryCorruptionException", message, context)

class ResourceExhaustionException(
    message: String,
    context: Map<String, Any?> = emptyMap()
) : OVMemoryException("ResourceExhaustionException", message, context)

class TimeoutException(
    message: String,
    context: Map<String, Any?> = emptyMap()
) : OVMemoryException("TimeoutException", message, context)

// ========== METRICS ==========

data class MetricsSnapshot(
    val queriesProcessed: Long,
    val qps: Double,
    val avgLatencyMs: Double,
    val p50LatencyMs: Double,
    val p95LatencyMs: Double,
    val p99LatencyMs: Double,
    val maxLatencyMs: Double,
    val errorCount: Long,
    val errorRatePct: Double,
    val errorBreakdown: Map<String, Long>,
    val uptimeSeconds: Double
)

class MetricsCollector {
    private val lock = ReadWriteLock()
    private var queriesProcessed: Long = 0
    private var totalLatency: Double = 0.0
    private val errorCount: ConcurrentHashMap<String, Long> = ConcurrentHashMap()
    private var p50LatencyMs: Double = 0.0
    private var p95LatencyMs: Double = 0.0
    private var p99LatencyMs: Double = 0.0
    private var maxLatencyMs: Double = 0.0
    private val startTime: Long = System.currentTimeMillis()

    fun recordLatency(latencyMs: Double) {
        lock.writeLock().lock()
        try {
            queriesProcessed++
            totalLatency += latencyMs
            if (latencyMs > maxLatencyMs) maxLatencyMs = latencyMs
            if (p50LatencyMs == 0.0) p50LatencyMs = latencyMs
            if (p95LatencyMs == 0.0) p95LatencyMs = latencyMs
            if (p99LatencyMs == 0.0) p99LatencyMs = latencyMs
        } finally {
            lock.writeLock().unlock()
        }
    }

    fun recordError(errorType: String) {
        errorCount.merge(errorType, 1L) { old, new -> old + new }
    }

    fun getMetrics(): MetricsSnapshot {
        lock.readLock().lock()
        try {
            val uptime = (System.currentTimeMillis() - startTime) / 1000.0
            val qps = if (uptime > 0) queriesProcessed / uptime else 0.0
            val avgLatency = if (queriesProcessed > 0) totalLatency / queriesProcessed else 0.0
            val totalErrors = errorCount.values.sum()
            val errorRate = if (queriesProcessed > 0) (totalErrors.toDouble() / queriesProcessed) * 100 else 0.0

            return MetricsSnapshot(
                queriesProcessed = queriesProcessed,
                qps = qps,
                avgLatencyMs = avgLatency,
                p50LatencyMs = p50LatencyMs,
                p95LatencyMs = p95LatencyMs,
                p99LatencyMs = p99LatencyMs,
                maxLatencyMs = maxLatencyMs,
                errorCount = totalErrors,
                errorRatePct = errorRate,
                errorBreakdown = errorCount.toMap(),
                uptimeSeconds = uptime
            )
        } finally {
            lock.readLock().unlock()
        }
    }
}

// ========== CIRCUIT BREAKER ==========

enum class CircuitBreakerState {
    CLOSED, OPEN, HALF_OPEN
}

class CircuitBreaker(
    private val failureThreshold: Long = 5,
    private val successThreshold: Long = 3,
    private val timeout: Long = 30000 // ms
) {
    private val lock = ReentrantReadWriteLock()
    private var state: CircuitBreakerState = CircuitBreakerState.CLOSED
    private var failureCount: Long = 0
    private var successCount: Long = 0
    private var lastFailureTime: Long = System.currentTimeMillis()

    fun <T> call(fn: () -> T): T {
        lock.readLock().lock()
        val currentState = state
        lock.readLock().unlock()

        if (currentState == CircuitBreakerState.OPEN) {
            if (System.currentTimeMillis() - lastFailureTime > timeout) {
                lock.writeLock().lock()
                try {
                    state = CircuitBreakerState.HALF_OPEN
                    successCount = 0
                } finally {
                    lock.writeLock().unlock()
                }
            } else {
                throw ResourceExhaustionException(
                    "Circuit breaker is OPEN",
                    mapOf("state" to CircuitBreakerState.OPEN)
                )
            }
        }

        return try {
            val result = fn()
            lock.writeLock().lock()
            try {
                failureCount = 0
                if (state == CircuitBreakerState.HALF_OPEN) {
                    successCount++
                    if (successCount >= successThreshold) {
                        state = CircuitBreakerState.CLOSED
                    }
                }
            } finally {
                lock.writeLock().unlock()
            }
            result
        } catch (e: Exception) {
            lock.writeLock().lock()
            try {
                failureCount++
                lastFailureTime = System.currentTimeMillis()
                if (failureCount >= failureThreshold) {
                    state = CircuitBreakerState.OPEN
                }
            } finally {
                lock.writeLock().unlock()
            }
            throw e
        }
    }

    fun getState(): CircuitBreakerState {
        lock.readLock().lock()
        try {
            return state
        } finally {
            lock.readLock().unlock()
        }
    }
}

// ========== MEMORY NODE ==========

data class MemoryNode(
    val id: String,
    val embedding: List<Double>,
    val text: String,
    val centrality: Double = 0.5,
    val importance: Double = 1.0,
    var age: Long = 0, // seconds
    val createdAt: Long = System.currentTimeMillis()
)

// ========== PRODUCTION OV-MEMORY ==========

data class HealthStatus(
    val status: String,
    val errorRatePct: Double,
    val metrics: MetricsSnapshot,
    val timestamp: String = Instant.now().toString()
)

class OVMemoryProduction(
    private val embeddingDim: Int,
    private val maxNodes: Int,
    private val enableMonitoring: Boolean = true
) {
    private val lock = ReentrantReadWriteLock()
    private val nodes: ConcurrentHashMap<String, MemoryNode> = ConcurrentHashMap()
    private val logger = StructuredLogger(LogLevel.INFO)
    private val metrics = MetricsCollector()
    private val circuitBreaker = CircuitBreaker(5, 3, 30000)
    private val errorLogs = ConcurrentLinkedQueue<Map<String, Any?>>()
    private val maxErrorLogSize = 1000

    // ========== INPUT VALIDATION ==========

    private fun validateEmbedding(embedding: List<Double>): Boolean {
        if (embedding.size != embeddingDim) {
            throw InvalidDataException(
                "Embedding dimension mismatch",
                mapOf("expected" to embeddingDim, "got" to embedding.size)
            )
        }

        embedding.forEachIndexed { idx, value ->
            if (value.isNaN() || value.isInfinite()) {
                throw InvalidDataException(
                    "Embedding contains NaN or Inf",
                    mapOf("index" to idx, "value" to value)
                )
            }
        }

        return true
    }

    private fun validateText(text: String): Boolean {
        if (text.isEmpty()) {
            throw InvalidDataException("Text cannot be empty", emptyMap())
        }
        if (text.length > 1_000_000) {
            throw InvalidDataException(
                "Text exceeds max length",
                mapOf("length" to text.length)
            )
        }
        return true
    }

    private fun validateResources(): Boolean {
        lock.readLock().lock()
        try {
            if (nodes.size >= maxNodes) {
                throw ResourceExhaustionException(
                    "Max nodes reached",
                    mapOf("current" to nodes.size, "max" to maxNodes)
                )
            }
        } finally {
            lock.readLock().unlock()
        }
        return true
    }

    // ========== CORE OPERATIONS ==========

    fun addMemory(
        embedding: List<Double>,
        text: String,
        centrality: Double = 0.5,
        nodeId: String? = null
    ): String {
        val startTime = System.nanoTime()

        // Validation
        validateEmbedding(embedding)
        validateText(text)
        validateResources()

        // Circuit breaker
        val id = circuitBreaker.call {
            lock.writeLock().lock()
            try {
                val nodeId = nodeId ?: "node_${System.nanoTime()}_${nodes.size}"
                val node = MemoryNode(
                    id = nodeId,
                    embedding = embedding,
                    text = text,
                    centrality = centrality,
                    importance = 1.0
                )
                nodes[nodeId] = node
                nodeId
            } finally {
                lock.writeLock().unlock()
            }
        }

        val latencyMs = (System.nanoTime() - startTime) / 1_000_000.0
        metrics.recordLatency(latencyMs)

        logger.log(LogLevel.INFO, "Memory added", mapOf(
            "node_id" to id,
            "latency_ms" to latencyMs
        ))

        return id
    }

    fun getMemory(nodeId: String): MemoryNode {
        lock.readLock().lock()
        try {
            return nodes[nodeId] ?: throw InvalidDataException(
                "Node not found",
                mapOf("node_id" to nodeId)
            )
        } finally {
            lock.readLock().unlock()
        }
    }

    fun getNodeCount(): Int {
        lock.readLock().lock()
        try {
            return nodes.size
        } finally {
            lock.readLock().unlock()
        }
    }

    // ========== HEALTH & METRICS ==========

    fun getHealthStatus(): HealthStatus {
        val metrics = metrics.getMetrics()
        val status = when {
            metrics.errorRatePct > 10 -> "CRITICAL"
            metrics.errorRatePct > 5 -> "WARNING"
            else -> "HEALTHY"
        }

        return HealthStatus(
            status = status,
            errorRatePct = metrics.errorRatePct,
            metrics = metrics
        )
    }

    fun getMetrics(): MetricsSnapshot = metrics.getMetrics()

    fun getCircuitBreakerState(): CircuitBreakerState = circuitBreaker.getState()

    // ========== ERROR HANDLING ==========

    private fun logError(error: Exception) {
        lock.writeLock().lock()
        try {
            val errorEntry = mapOf(
                "error" to error.message,
                "type" to error::class.simpleName,
                "timestamp" to Instant.now().toString()
            )
            errorLogs.offer(errorEntry)

            if (errorLogs.size > maxErrorLogSize) {
                errorLogs.poll()
            }

            logger.log(LogLevel.ERROR, "Error logged", errorEntry)
        } finally {
            lock.writeLock().unlock()
        }
    }

    fun getErrorLogs(): List<Map<String, Any?>> = errorLogs.toList()
}

// ========== HELPER CLASSES ==========

class ReadWriteLock {
    private val lock = ReentrantReadWriteLock()

    fun <T> read(block: () -> T): T {
        lock.readLock().lock()
        try {
            return block()
        } finally {
            lock.readLock().unlock()
        }
    }

    fun <T> write(block: () -> T): T {
        lock.writeLock().lock()
        try {
            return block()
        } finally {
            lock.writeLock().unlock()
        }
    }

    fun writeLock() = lock.writeLock()
    fun readLock() = lock.readLock()
}

// ========== EXAMPLE USAGE ==========

fun main() {
    val memory = OVMemoryProduction(embeddingDim = 768, maxNodes = 10000, enableMonitoring = true)

    try {
        // Add memory
        val embedding = List(768) { 0.5 }
        val nodeId = memory.addMemory(embedding, "Sample memory text", 0.9)
        println("Added node: $nodeId")

        // Retrieve memory
        val node = memory.getMemory(nodeId)
        println("Retrieved node: ${node.id}")

        // Get health
        val health = memory.getHealthStatus()
        println("Health status: ${health.status}")

        // Get metrics
        val metrics = memory.getMetrics()
        println("Queries processed: ${metrics.queriesProcessed}")
    } catch (e: OVMemoryException) {
        println("Error: ${e.message}")
    }
}

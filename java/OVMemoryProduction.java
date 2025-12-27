import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.*;
import java.util.concurrent.*;
import java.util.logging.*;
import com.google.common.collect.EvictingQueue;

/**
 * OV-Memory v1.1: Production-Hardened Java Implementation
 * Comprehensive error handling, structured logging, and monitoring
 *
 * @author Vaibhav Prayaga
 * @date December 27, 2025
 */
public class OVMemoryProduction {

    // ========================================================================
    // LOGGING
    // ========================================================================

    private static class StructuredLogger {
        private final Logger logger;
        private final Map<String, String> context = new ConcurrentHashMap<>();

        public StructuredLogger(String name) {
            this.logger = Logger.getLogger(name);
            logger.setLevel(Level.INFO);
            ConsoleHandler handler = new ConsoleHandler();
            handler.setFormatter(new SimpleFormatter() {
                @Override
                public synchronized String format(LogRecord record) {
                    return String.format(
                        "%s | %-8s | %s | %s%n",
                        LocalDateTime.now(),
                        record.getLevel().getLocalizedName(),
                        record.getLoggerName(),
                        record.getMessage()
                    );
                }
            });
            logger.addHandler(handler);
        }

        public void setContext(String key, String value) {
            context.put(key, value);
        }

        private String formatMessage(String msg, Object... args) {
            StringBuilder sb = new StringBuilder(msg);
            if (!context.isEmpty()) {
                sb.append(" | ");
                context.forEach((k, v) -> sb.append(k).append("=").append(v).append(" "));
            }
            return sb.toString();
        }

        public void debug(String msg) {
            logger.log(Level.FINE, formatMessage(msg));
        }

        public void info(String msg) {
            logger.log(Level.INFO, formatMessage(msg));
        }

        public void warning(String msg) {
            logger.log(Level.WARNING, formatMessage(msg));
        }

        public void error(String msg, Throwable t) {
            logger.log(Level.SEVERE, formatMessage(msg), t);
        }
    }

    // ========================================================================
    // CUSTOM EXCEPTIONS
    // ========================================================================

    static class OVMemoryException extends Exception {
        private final String code;
        private final LocalDateTime timestamp;

        public OVMemoryException(String message, String code) {
            super(message);
            this.code = code;
            this.timestamp = LocalDateTime.now();
        }

        public String getCode() {
            return code;
        }
    }

    static class InvalidDataException extends OVMemoryException {
        public InvalidDataException(String message) {
            super(message, "INVALID_DATA");
        }
    }

    static class ResourceExhaustionException extends OVMemoryException {
        public ResourceExhaustionException(String message) {
            super(message, "RESOURCE_EXHAUSTION");
        }
    }

    static class TimeoutException extends OVMemoryException {
        public TimeoutException(String message) {
            super(message, "TIMEOUT");
        }
    }

    // ========================================================================
    // CIRCUIT BREAKER
    // ========================================================================

    enum CircuitBreakerState {
        CLOSED, OPEN, HALF_OPEN
    }

    static class CircuitBreaker {
        private CircuitBreakerState state = CircuitBreakerState.CLOSED;
        private int failureCount = 0;
        private int successCount = 0;
        private LocalDateTime lastFailureTime;
        private final int failureThreshold;
        private final int recoveryTimeoutSeconds;
        private final int successThreshold;
        private final StructuredLogger logger = new StructuredLogger("CircuitBreaker");

        public CircuitBreaker(int failureThreshold, int recoveryTimeout, int successThreshold) {
            this.failureThreshold = failureThreshold;
            this.recoveryTimeoutSeconds = recoveryTimeout;
            this.successThreshold = successThreshold;
        }

        public synchronized <T> T execute(Callable<T> action) throws Exception {
            if (state == CircuitBreakerState.OPEN) {
                if (shouldAttemptReset()) {
                    state = CircuitBreakerState.HALF_OPEN;
                    logger.info("Circuit breaker entering HALF_OPEN state");
                } else {
                    throw new OVMemoryException(
                        "Circuit breaker is OPEN",
                        "CIRCUIT_BREAKER_OPEN"
                    );
                }
            }

            try {
                T result = action.call();
                onSuccess();
                return result;
            } catch (Exception e) {
                onFailure();
                throw e;
            }
        }

        private boolean shouldAttemptReset() {
            if (lastFailureTime == null) return false;
            long elapsed = ChronoUnit.SECONDS.between(lastFailureTime, LocalDateTime.now());
            return elapsed >= recoveryTimeoutSeconds;
        }

        private synchronized void onSuccess() {
            failureCount = 0;
            if (state == CircuitBreakerState.HALF_OPEN) {
                successCount++;
                if (successCount >= successThreshold) {
                    state = CircuitBreakerState.CLOSED;
                    successCount = 0;
                    logger.info("Circuit breaker CLOSED after recovery");
                }
            }
        }

        private synchronized void onFailure() {
            failureCount++;
            lastFailureTime = LocalDateTime.now();
            successCount = 0;
            if (failureCount >= failureThreshold) {
                state = CircuitBreakerState.OPEN;
                logger.warning("Circuit breaker OPEN after failures");
            }
        }
    }

    // ========================================================================
    // METRICS
    // ========================================================================

    static class MetricsCollector {
        private int queriesProcessed = 0;
        private final Queue<Double> latencies = new ConcurrentLinkedQueue<>();
        private int driftDetections = 0;
        private int loopPreventions = 0;
        private int redundancyFilters = 0;
        private int errorsEncountered = 0;
        private final int maxLatencyHistorySize;

        public MetricsCollector(int maxSize) {
            this.maxLatencyHistorySize = maxSize;
        }

        public synchronized void recordQuery(double latencyMs) {
            queriesProcessed++;
            latencies.add(latencyMs);
            if (latencies.size() > maxLatencyHistorySize) {
                latencies.poll();
            }
        }

        public synchronized void recordDriftDetection() {
            driftDetections++;
        }

        public synchronized void recordLoopPrevention() {
            loopPreventions++;
        }

        public synchronized void recordRedundancyFilter() {
            redundancyFilters++;
        }

        public synchronized void recordError() {
            errorsEncountered++;
        }

        public synchronized Map<String, Object> getMetrics() {
            Map<String, Object> metrics = new LinkedHashMap<>();
            metrics.put("queries_processed", queriesProcessed);
            metrics.put("avg_latency_ms",
                latencies.isEmpty() ? 0 :
                latencies.stream().mapToDouble(Double::doubleValue).average().orElse(0)
            );
            metrics.put("drift_detections", driftDetections);
            metrics.put("loop_preventions", loopPreventions);
            metrics.put("redundancy_filters", redundancyFilters);
            metrics.put("errors", errorsEncountered);
            return metrics;
        }

        public synchronized Map<String, Object> getHealthStatus() {
            Map<String, Object> health = new LinkedHashMap<>();
            if (queriesProcessed == 0) {
                health.put("status", "UNKNOWN");
                return health;
            }

            double errorRate = (double) errorsEncountered / queriesProcessed;
            String status = errorRate > 0.1 ? "CRITICAL" :
                           errorRate > 0.05 ? "WARNING" :
                           "HEALTHY";
            health.put("status", status);
            health.put("error_rate_percent", errorRate * 100);
            health.put("queries", queriesProcessed);
            health.put("errors", errorsEncountered);
            return health;
        }
    }

    // ========================================================================
    // MEMORY NODE
    // ========================================================================

    static class MemoryNode {
        public final double[] embedding;
        public final String text;
        public final LocalDateTime timestamp;
        public double centrality = 0.0;
        public double recency = 0.0;
        public double intrinsicWeight = 1.0;
        public int accessCount = 0;
        public LocalDateTime lastAccess;
        public final Map<String, Object> metadata;

        public MemoryNode(double[] embedding, String text, Map<String, Object> metadata) {
            this.embedding = embedding.clone();
            this.text = text;
            this.timestamp = LocalDateTime.now();
            this.metadata = metadata != null ? new HashMap<>(metadata) : new HashMap<>();
        }

        public boolean validate() {
            if (embedding == null || embedding.length == 0) return false;
            if (text == null || text.trim().isEmpty()) return false;
            if (centrality < 0 || centrality > 1) return false;
            if (recency < 0 || recency > 1) return false;
            if (intrinsicWeight <= 0) return false;
            for (double val : embedding) {
                if (!Double.isFinite(val)) return false;
            }
            return true;
        }
    }

    // ========================================================================
    // MAIN SYSTEM
    // ========================================================================

    private final int embeddingDim;
    private final int maxNodes;
    private final Map<Integer, MemoryNode> memory = new ConcurrentHashMap<>();
    private final MetricsCollector metrics;
    private final CircuitBreaker circuitBreaker;
    private final StructuredLogger logger;
    private final Queue<Map<String, String>> errorLog;
    private int nodeCounter = 0;

    public OVMemoryProduction(int embeddingDim, int maxNodes, boolean enableMonitoring) {
        this.embeddingDim = embeddingDim;
        this.maxNodes = maxNodes;
        this.metrics = enableMonitoring ? new MetricsCollector(10000) : null;
        this.circuitBreaker = new CircuitBreaker(5, 60, 2);
        this.logger = new StructuredLogger("OVMemoryProduction");
        this.errorLog = EvictingQueue.create(1000);

        logger.info("OV-Memory initialized: embeddingDim=" + embeddingDim +
                   ", maxNodes=" + maxNodes +
                   ", monitoring=" + enableMonitoring);
    }

    public synchronized int addMemory(
        double[] embedding,
        String text,
        double intrinsicWeight,
        Map<String, Object> metadata
    ) throws OVMemoryException {
        long startTime = System.currentTimeMillis();

        try {
            validateInput(embedding, text);
            if (intrinsicWeight <= 0) {
                throw new InvalidDataException("Intrinsic weight must be positive");
            }
            if (memory.size() >= maxNodes) {
                throw new ResourceExhaustionException(
                    "Memory limit reached: " + memory.size() + " / " + maxNodes
                );
            }

            int nodeId = nodeCounter++;
            MemoryNode node = new MemoryNode(embedding, text, metadata);
            node.intrinsicWeight = intrinsicWeight;

            if (!node.validate()) {
                throw new OVMemoryException("Node validation failed", "VALIDATION_FAILED");
            }

            memory.put(nodeId, node);

            long elapsed = System.currentTimeMillis() - startTime;
            if (metrics != null) {
                metrics.recordQuery(elapsed);
            }

            logger.debug("Memory added: id=" + nodeId + ", latency=" + elapsed + "ms");
            return nodeId;

        } catch (OVMemoryException e) {
            recordError("add_memory", e);
            logger.error("OV-Memory error: " + e.getMessage(), e);
            throw e;
        }
    }

    public List<Object[]> retrieveMemories(
        double[] queryEmbedding,
        int topK,
        double alpha,
        long timeoutMs
    ) throws OVMemoryException {
        try {
            return circuitBreaker.execute(() ->
                retrieveMemoriesInternal(queryEmbedding, topK, alpha, timeoutMs)
            );
        } catch (Exception e) {
            if (e instanceof OVMemoryException) {
                throw (OVMemoryException) e;
            }
            recordError("retrieve_memories", new OVMemoryException(e.getMessage(), "RETRIEVE_FAILED"));
            throw new OVMemoryException("Retrieval failed: " + e.getMessage(), "RETRIEVE_FAILED");
        }
    }

    private List<Object[]> retrieveMemoriesInternal(
        double[] queryEmbedding,
        int topK,
        double alpha,
        long timeoutMs
    ) throws OVMemoryException {
        long startTime = System.currentTimeMillis();

        validateInput(queryEmbedding, "query");
        topK = Math.min(topK, memory.size());

        List<Object[]> results = new ArrayList<>();

        for (Map.Entry<Integer, MemoryNode> entry : memory.entrySet()) {
            if (System.currentTimeMillis() - startTime > timeoutMs) {
                throw new TimeoutException("Retrieval timeout");
            }

            int nodeId = entry.getKey();
            MemoryNode node = entry.getValue();

            double similarity = cosineSimilarity(queryEmbedding, node.embedding);
            double recency = Math.exp(-0.1 *
                ChronoUnit.SECONDS.between(node.timestamp, LocalDateTime.now()));
            double priority = Math.pow(similarity, alpha) *
                            Math.pow(Math.max(0.1, node.centrality), 1 - alpha) *
                            Math.pow(Math.max(0.1, recency), 1 - alpha) *
                            node.intrinsicWeight;

            results.add(new Object[]{nodeId, priority, node.text});
        }

        results.sort((a, b) -> Double.compare((Double)b[1], (Double)a[1]));

        long elapsed = System.currentTimeMillis() - startTime;
        if (metrics != null) {
            metrics.recordQuery(elapsed);
        }

        logger.debug("Retrieved " + Math.min(topK, results.size()) + " memories in " + elapsed + "ms");
        return results.subList(0, Math.min(topK, results.size()));
    }

    private double cosineSimilarity(double[] a, double[] b) {
        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-8);
    }

    private void validateInput(double[] embedding, String text) throws InvalidDataException {
        if (embedding == null) {
            throw new InvalidDataException("Embedding cannot be null");
        }
        if (embedding.length != embeddingDim) {
            throw new InvalidDataException(
                "Dimension mismatch: expected " + embeddingDim + ", got " + embedding.length
            );
        }
        for (double val : embedding) {
            if (!Double.isFinite(val)) {
                throw new InvalidDataException("Embedding contains NaN or Inf");
            }
        }
        if (text == null || text.trim().isEmpty()) {
            throw new InvalidDataException("Text cannot be null or empty");
        }
    }

    private void recordError(String operation, OVMemoryException e) {
        Map<String, String> error = new LinkedHashMap<>();
        error.put("timestamp", LocalDateTime.now().toString());
        error.put("operation", operation);
        error.put("code", e.getCode());
        error.put("message", e.getMessage());
        errorLog.add(error);
        if (metrics != null) {
            metrics.recordError();
        }
    }

    public Map<String, Object> getMetrics() {
        if (metrics == null) return Collections.singletonMap("status", "Monitoring disabled");
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("metrics", metrics.getMetrics());
        result.put("health", metrics.getHealthStatus());
        result.put("memory_size", memory.size());
        result.put("node_limit", maxNodes);
        return result;
    }

    public Map<String, Object> getHealthStatus() {
        if (metrics == null) return Collections.singletonMap("status", "UNKNOWN");
        return metrics.getHealthStatus();
    }

    // ========================================================================
    // EXAMPLE USAGE
    // ========================================================================

    public static void main(String[] args) throws OVMemoryException {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("OV-MEMORY v1.1: Production-Hardened Java Implementation");
        System.out.println("=".repeat(80) + "\n");

        OVMemoryProduction memory = new OVMemoryProduction(768, 10000, true);

        System.out.println("✅ OV-Memory initialized with production hardening");
        System.out.println("   - Error handling: Enabled");
        System.out.println("   - Monitoring: Enabled");
        System.out.println("   - Circuit breaker: Enabled");
        System.out.println("   - Logging: Enabled\n");

        // Add test memories
        System.out.println("📝 Adding test memories...");
        String[] texts = {
            "The quick brown fox jumps over the lazy dog",
            "Machine learning is transforming AI",
            "Neural networks learn hierarchical representations",
            "Deep learning requires lots of data",
            "Natural language processing is challenging"
        };

        for (String text : texts) {
            double[] embedding = new double[768];
            for (int i = 0; i < 768; i++) {
                embedding[i] = Math.random() - 0.5;
            }
            int nodeId = memory.addMemory(embedding, text, 1.0, null);
            System.out.println("   Added memory " + nodeId + ": " + text.substring(0, 40) + "...");
        }

        // Retrieve memories
        System.out.println("\n🔍 Retrieving memories...");
        double[] query = new double[768];
        for (int i = 0; i < 768; i++) {
            query[i] = Math.random() - 0.5;
        }
        List<Object[]> results = memory.retrieveMemories(query, 3, 0.75, 30000);
        for (Object[] result : results) {
            System.out.println("   ID " + result[0] + ": Priority " +
                String.format("%.4f", result[1]) + " - " +
                ((String)result[2]).substring(0, 40) + "...");
        }

        // Print metrics
        System.out.println("\n📊 System Metrics:");
        Map<String, Object> metrics = memory.getMetrics();
        metrics.forEach((k, v) -> System.out.println("   " + k + ": " + v));

        // Print health
        System.out.println("\n💚 Health Status:");
        Map<String, Object> health = memory.getHealthStatus();
        health.forEach((k, v) -> System.out.println("   " + k + ": " + v));

        System.out.println("\n" + "=".repeat(80));
        System.out.println("✅ Production-hardened OV-Memory working correctly!");
        System.out.println("=".repeat(80) + "\n");
    }
}

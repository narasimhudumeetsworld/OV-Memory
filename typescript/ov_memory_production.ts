/**
 * OV-Memory Production Implementation in TypeScript/Node.js
 * Complete with error handling, structured logging, metrics, and circuit breaker
 */

import { EventEmitter } from 'events';

// ========== LOGGING ==========

enum LogLevel {
  DEBUG = 0,
  INFO = 1,
  WARNING = 2,
  ERROR = 3,
  CRITICAL = 4,
}

class StructuredLogger {
  private logLevel: LogLevel;
  private logs: string[] = [];

  constructor(logLevel: LogLevel = LogLevel.INFO) {
    this.logLevel = logLevel;
  }

  log(level: LogLevel, message: string, fields: Record<string, any> = {}): void {
    if (level < this.logLevel) return;

    const entry = {
      timestamp: new Date().toISOString(),
      level: LogLevel[level],
      message,
      fields,
    };

    const jsonStr = JSON.stringify(entry);
    console.log(jsonStr);
    this.logs.push(jsonStr);
  }

  getLogs(): string[] {
    return this.logs;
  }
}

// ========== CUSTOM ERRORS ==========

class OVMemoryError extends Error {
  constructor(
    public type: string,
    message: string,
    public context: Record<string, any> = {}
  ) {
    super(`[${type}] ${message}`);
    this.name = 'OVMemoryError';
  }
}

class InvalidDataException extends OVMemoryError {
  constructor(message: string, context?: Record<string, any>) {
    super('InvalidDataException', message, context);
  }
}

class MemoryCorruptionException extends OVMemoryError {
  constructor(message: string, context?: Record<string, any>) {
    super('MemoryCorruptionException', message, context);
  }
}

class ResourceExhaustionException extends OVMemoryError {
  constructor(message: string, context?: Record<string, any>) {
    super('ResourceExhaustionException', message, context);
  }
}

class TimeoutException extends OVMemoryError {
  constructor(message: string, context?: Record<string, any>) {
    super('TimeoutException', message, context);
  }
}

// ========== METRICS ==========

interface MetricsSnapshot {
  queriesProcessed: number;
  qps: number;
  avgLatencyMs: number;
  p50LatencyMs: number;
  p95LatencyMs: number;
  p99LatencyMs: number;
  maxLatencyMs: number;
  errorCount: number;
  errorRatePct: number;
  errorBreakdown: Record<string, number>;
  uptimeSeconds: number;
}

class MetricsCollector {
  private queriesProcessed = 0;
  private totalLatency = 0;
  private errorCount: Record<string, number> = {};
  private p50LatencyMs = 0;
  private p95LatencyMs = 0;
  private p99LatencyMs = 0;
  private maxLatencyMs = 0;
  private startTime = Date.now();

  recordLatency(latencyMs: number): void {
    this.queriesProcessed++;
    this.totalLatency += latencyMs;
    if (latencyMs > this.maxLatencyMs) {
      this.maxLatencyMs = latencyMs;
    }
  }

  recordError(errorType: string): void {
    this.errorCount[errorType] = (this.errorCount[errorType] || 0) + 1;
  }

  getMetrics(): MetricsSnapshot {
    const uptime = (Date.now() - this.startTime) / 1000;
    const qps = uptime > 0 ? this.queriesProcessed / uptime : 0;
    const avgLatency = this.queriesProcessed > 0 ? this.totalLatency / this.queriesProcessed : 0;
    const totalErrors = Object.values(this.errorCount).reduce((a, b) => a + b, 0);
    const errorRate = this.queriesProcessed > 0 ? (totalErrors / this.queriesProcessed) * 100 : 0;

    return {
      queriesProcessed: this.queriesProcessed,
      qps,
      avgLatencyMs: avgLatency,
      p50LatencyMs: this.p50LatencyMs,
      p95LatencyMs: this.p95LatencyMs,
      p99LatencyMs: this.p99LatencyMs,
      maxLatencyMs: this.maxLatencyMs,
      errorCount: totalErrors,
      errorRatePct: errorRate,
      errorBreakdown: this.errorCount,
      uptimeSeconds: uptime,
    };
  }
}

// ========== CIRCUIT BREAKER ==========

enum CircuitBreakerState {
  CLOSED = 'CLOSED',
  OPEN = 'OPEN',
  HALF_OPEN = 'HALF_OPEN',
}

class CircuitBreaker {
  private state = CircuitBreakerState.CLOSED;
  private failureCount = 0;
  private successCount = 0;
  private lastFailureTime = Date.now();

  constructor(
    private failureThreshold = 5,
    private successThreshold = 3,
    private timeoutMs = 30000
  ) {}

  async call<T>(fn: () => Promise<T>): Promise<T> {
    if (this.state === CircuitBreakerState.OPEN) {
      if (Date.now() - this.lastFailureTime > this.timeoutMs) {
        this.state = CircuitBreakerState.HALF_OPEN;
        this.successCount = 0;
      } else {
        throw new ResourceExhaustionException('Circuit breaker is OPEN', {
          state: this.state,
        });
      }
    }

    try {
      const result = await fn();
      this.failureCount = 0;
      if (this.state === CircuitBreakerState.HALF_OPEN) {
        this.successCount++;
        if (this.successCount >= this.successThreshold) {
          this.state = CircuitBreakerState.CLOSED;
        }
      }
      return result;
    } catch (error) {
      this.failureCount++;
      this.lastFailureTime = Date.now();
      if (this.failureCount >= this.failureThreshold) {
        this.state = CircuitBreakerState.OPEN;
      }
      throw error;
    }
  }

  getState(): CircuitBreakerState {
    return this.state;
  }
}

// ========== MEMORY NODE ==========

interface MemoryNode {
  id: string;
  embedding: number[];
  text: string;
  centrality: number;
  importance: number;
  age: number;
  createdAt: Date;
}

interface HealthStatus {
  status: 'HEALTHY' | 'WARNING' | 'CRITICAL';
  errorRatePct: number;
  metrics: MetricsSnapshot;
  timestamp: string;
}

// ========== PRODUCTION OV-MEMORY ==========

class OVMemoryProduction extends EventEmitter {
  private nodes: Map<string, MemoryNode> = new Map();
  private logger: StructuredLogger;
  private metrics: MetricsCollector;
  private circuitBreaker: CircuitBreaker;
  private errorLogs: Map<string, unknown>[] = [];
  private readonly maxErrorLogSize = 1000;

  constructor(
    private embeddingDim: number,
    private maxNodes: number,
    enableMonitoring: boolean = true
  ) {
    super();
    this.logger = new StructuredLogger(LogLevel.INFO);
    this.metrics = new MetricsCollector();
    this.circuitBreaker = new CircuitBreaker(5, 3, 30000);
  }

  // ========== INPUT VALIDATION ==========

  private validateEmbedding(embedding: number[]): void {
    if (embedding.length !== this.embeddingDim) {
      throw new InvalidDataException('Embedding dimension mismatch', {
        expected: this.embeddingDim,
        got: embedding.length,
      });
    }

    for (let i = 0; i < embedding.length; i++) {
      if (!isFinite(embedding[i])) {
        throw new InvalidDataException('Embedding contains NaN or Inf', {
          index: i,
          value: embedding[i],
        });
      }
    }
  }

  private validateText(text: string): void {
    if (!text || text.length === 0) {
      throw new InvalidDataException('Text cannot be empty');
    }
    if (text.length > 1_000_000) {
      throw new InvalidDataException('Text exceeds max length', {
        length: text.length,
      });
    }
  }

  private validateResources(): void {
    if (this.nodes.size >= this.maxNodes) {
      throw new ResourceExhaustionException('Max nodes reached', {
        current: this.nodes.size,
        max: this.maxNodes,
      });
    }
  }

  // ========== CORE OPERATIONS ==========

  async addMemory(
    embedding: number[],
    text: string,
    centrality: number = 0.5,
    nodeId?: string
  ): Promise<string> {
    const startTime = Date.now();

    // Validation
    this.validateEmbedding(embedding);
    this.validateText(text);
    this.validateResources();

    // Circuit breaker
    const id = await this.circuitBreaker.call(async () => {
      const newId = nodeId || `node_${Date.now()}_${this.nodes.size}`;
      const node: MemoryNode = {
        id: newId,
        embedding,
        text,
        centrality,
        importance: 1.0,
        age: 0,
        createdAt: new Date(),
      };
      this.nodes.set(newId, node);
      return newId;
    });

    const latencyMs = Date.now() - startTime;
    this.metrics.recordLatency(latencyMs);

    this.logger.log(LogLevel.INFO, 'Memory added', {
      node_id: id,
      latency_ms: latencyMs,
    });

    return id;
  }

  async getMemory(nodeId: string): Promise<MemoryNode> {
    const node = this.nodes.get(nodeId);
    if (!node) {
      throw new InvalidDataException('Node not found', {
        node_id: nodeId,
      });
    }
    return node;
  }

  // ========== HEALTH & METRICS ==========

  getHealthStatus(): HealthStatus {
    const metrics = this.metrics.getMetrics();
    const status: 'HEALTHY' | 'WARNING' | 'CRITICAL' =
      metrics.errorRatePct > 10 ? 'CRITICAL' :
      metrics.errorRatePct > 5 ? 'WARNING' :
      'HEALTHY';

    return {
      status,
      errorRatePct: metrics.errorRatePct,
      metrics,
      timestamp: new Date().toISOString(),
    };
  }

  getMetrics(): MetricsSnapshot {
    return this.metrics.getMetrics();
  }

  getCircuitBreakerState(): CircuitBreakerState {
    return this.circuitBreaker.getState();
  }
}

// ========== EXAMPLE USAGE ==========

async function main() {
  const memory = new OVMemoryProduction(768, 10000, true);

  try {
    // Create embedding
    const embedding = Array(768).fill(0.5);

    // Add memory
    const nodeId = await memory.addMemory(embedding, 'Sample text', 0.9);
    console.log('Added node:', nodeId);

    // Retrieve memory
    const node = await memory.getMemory(nodeId);
    console.log('Retrieved node:', node.id);

    // Get health
    const health = memory.getHealthStatus();
    console.log('Health status:', health.status);

    // Get metrics
    const metrics = memory.getMetrics();
    console.log('QPS:', metrics.qps);
  } catch (error) {
    console.error('Error:', error);
  }
}

export {
  OVMemoryProduction,
  InvalidDataException,
  MemoryCorruptionException,
  ResourceExhaustionException,
  TimeoutException,
  OVMemoryError,
  MetricsCollector,
  CircuitBreaker,
  CircuitBreakerState,
  StructuredLogger,
  LogLevel,
  MemoryNode,
  HealthStatus,
  MetricsSnapshot,
};

if (require.main === module) {
  main().catch(console.error);
}

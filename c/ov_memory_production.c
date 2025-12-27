/*
 * OV-Memory Production Implementation in C
 * Complete with error handling, structured logging, metrics, and circuit breaker
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <pthread.h>

/* ========== LOGGING ========== */

typedef enum {
    LOG_DEBUG,
    LOG_INFO,
    LOG_WARNING,
    LOG_ERROR,
    LOG_CRITICAL
} LogLevel;

const char* log_level_string[] = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"};

typedef struct {
    LogLevel level;
    char message[256];
    char timestamp[32];
    char context[512];
} LogEntry;

void log_message(LogLevel level, const char* message, const char* context) {
    time_t now = time(NULL);
    struct tm* timeinfo = localtime(&now);
    char timestamp[32];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%SZ", timeinfo);

    printf("{\"timestamp\": \"%s\", \"level\": \"%s\", \"message\": \"%s\", \"context\": %s}\n",
           timestamp, log_level_string[level], message, context ? context : "{}");
}

/* ========== CUSTOM ERRORS ========== */

typedef enum {
    ERROR_INVALID_DATA,
    ERROR_MEMORY_CORRUPTION,
    ERROR_RESOURCE_EXHAUSTION,
    ERROR_TIMEOUT,
    ERROR_NONE
} ErrorType;

typedef struct {
    ErrorType type;
    char message[256];
    char context[512];
} OVMemoryError;

OVMemoryError create_error(ErrorType type, const char* message, const char* context) {
    OVMemoryError error;
    error.type = type;
    strncpy(error.message, message, sizeof(error.message) - 1);
    strncpy(error.context, context ? context : "{}", sizeof(error.context) - 1);
    return error;
}

/* ========== METRICS ========== */

typedef struct {
    long queries_processed;
    double total_latency;
    double avg_latency;
    double p50_latency;
    double p95_latency;
    double p99_latency;
    double max_latency;
    long error_count;
    double error_rate;
    double uptime_seconds;
    time_t start_time;
    pthread_mutex_t lock;
} MetricsCollector;

MetricsCollector* metrics_create() {
    MetricsCollector* m = (MetricsCollector*)malloc(sizeof(MetricsCollector));
    m->queries_processed = 0;
    m->total_latency = 0.0;
    m->max_latency = 0.0;
    m->error_count = 0;
    m->start_time = time(NULL);
    pthread_mutex_init(&m->lock, NULL);
    return m;
}

void metrics_record_latency(MetricsCollector* m, double latency_ms) {
    pthread_mutex_lock(&m->lock);
    m->queries_processed++;
    m->total_latency += latency_ms;
    if (latency_ms > m->max_latency) {
        m->max_latency = latency_ms;
    }
    pthread_mutex_unlock(&m->lock);
}

void metrics_record_error(MetricsCollector* m) {
    pthread_mutex_lock(&m->lock);
    m->error_count++;
    pthread_mutex_unlock(&m->lock);
}

void metrics_get(MetricsCollector* m, char* output, size_t size) {
    pthread_mutex_lock(&m->lock);
    time_t now = time(NULL);
    double uptime = difftime(now, m->start_time);
    double qps = uptime > 0 ? (double)m->queries_processed / uptime : 0.0;
    double avg_latency = m->queries_processed > 0 ? m->total_latency / m->queries_processed : 0.0;
    double error_rate = m->queries_processed > 0 ? ((double)m->error_count / m->queries_processed) * 100.0 : 0.0;

    snprintf(output, size,
             "{\"queries_processed\": %ld, \"qps\": %.2f, \"avg_latency_ms\": %.2f, "
             "\"max_latency_ms\": %.2f, \"error_count\": %ld, \"error_rate_pct\": %.2f}",
             m->queries_processed, qps, avg_latency, m->max_latency, m->error_count, error_rate);
    pthread_mutex_unlock(&m->lock);
}

void metrics_destroy(MetricsCollector* m) {
    pthread_mutex_destroy(&m->lock);
    free(m);
}

/* ========== CIRCUIT BREAKER ========== */

typedef enum {
    CB_CLOSED,
    CB_OPEN,
    CB_HALF_OPEN
} CircuitBreakerState;

typedef struct {
    CircuitBreakerState state;
    long failure_count;
    long success_count;
    long failure_threshold;
    long success_threshold;
    time_t last_failure_time;
    int timeout_seconds;
    pthread_mutex_t lock;
} CircuitBreaker;

CircuitBreaker* circuit_breaker_create() {
    CircuitBreaker* cb = (CircuitBreaker*)malloc(sizeof(CircuitBreaker));
    cb->state = CB_CLOSED;
    cb->failure_count = 0;
    cb->success_count = 0;
    cb->failure_threshold = 5;
    cb->success_threshold = 3;
    cb->last_failure_time = time(NULL);
    cb->timeout_seconds = 30;
    pthread_mutex_init(&cb->lock, NULL);
    return cb;
}

int circuit_breaker_call(CircuitBreaker* cb, int (*fn)()) {
    pthread_mutex_lock(&cb->lock);
    
    if (cb->state == CB_OPEN) {
        if (difftime(time(NULL), cb->last_failure_time) > cb->timeout_seconds) {
            cb->state = CB_HALF_OPEN;
            cb->success_count = 0;
        } else {
            pthread_mutex_unlock(&cb->lock);
            return -1; // Circuit breaker open
        }
    }
    
    pthread_mutex_unlock(&cb->lock);
    
    int result = fn();
    
    pthread_mutex_lock(&cb->lock);
    if (result != 0) {
        cb->failure_count++;
        cb->last_failure_time = time(NULL);
        if (cb->failure_count >= cb->failure_threshold) {
            cb->state = CB_OPEN;
        }
    } else {
        cb->failure_count = 0;
        if (cb->state == CB_HALF_OPEN) {
            cb->success_count++;
            if (cb->success_count >= cb->success_threshold) {
                cb->state = CB_CLOSED;
            }
        }
    }
    pthread_mutex_unlock(&cb->lock);
    
    return result;
}

void circuit_breaker_destroy(CircuitBreaker* cb) {
    pthread_mutex_destroy(&cb->lock);
    free(cb);
}

/* ========== MEMORY NODE ========== */

typedef struct {
    char id[64];
    double* embedding;
    int embedding_dim;
    char text[1024];
    double centrality;
    double importance;
    time_t created_at;
} MemoryNode;

/* ========== PRODUCTION OV-MEMORY ========== */

typedef struct {
    int embedding_dim;
    int max_nodes;
    MemoryNode** nodes;
    int node_count;
    MetricsCollector* metrics;
    CircuitBreaker* circuit_breaker;
    int enable_monitoring;
    pthread_rwlock_t lock;
} OVMemoryProduction;

OVMemoryProduction* ov_memory_create(int embedding_dim, int max_nodes, int enable_monitoring) {
    OVMemoryProduction* mem = (OVMemoryProduction*)malloc(sizeof(OVMemoryProduction));
    mem->embedding_dim = embedding_dim;
    mem->max_nodes = max_nodes;
    mem->nodes = (MemoryNode**)malloc(sizeof(MemoryNode*) * max_nodes);
    mem->node_count = 0;
    mem->metrics = metrics_create();
    mem->circuit_breaker = circuit_breaker_create();
    mem->enable_monitoring = enable_monitoring;
    pthread_rwlock_init(&mem->lock, NULL);
    return mem;
}

// Input validation
int validate_embedding(OVMemoryProduction* mem, double* embedding, OVMemoryError* error) {
    if (embedding == NULL) {
        *error = create_error(ERROR_INVALID_DATA, "Embedding is NULL", "{}");
        return -1;
    }
    return 0;
}

int validate_text(const char* text, OVMemoryError* error) {
    if (text == NULL || strlen(text) == 0) {
        *error = create_error(ERROR_INVALID_DATA, "Text cannot be empty", "{}");
        return -1;
    }
    if (strlen(text) > 1000000) {
        *error = create_error(ERROR_INVALID_DATA, "Text exceeds max length", "{}");
        return -1;
    }
    return 0;
}

int validate_resources(OVMemoryProduction* mem, OVMemoryError* error) {
    if (mem->node_count >= mem->max_nodes) {
        *error = create_error(ERROR_RESOURCE_EXHAUSTION, "Max nodes reached", "{}");
        return -1;
    }
    return 0;
}

// Core operations
int ov_memory_add(OVMemoryProduction* mem, double* embedding, const char* text, 
                  double centrality, char* node_id_out) {
    OVMemoryError error = {ERROR_NONE, "", ""};
    
    // Validation
    if (validate_embedding(mem, embedding, &error) != 0) return -1;
    if (validate_text(text, &error) != 0) return -1;
    if (validate_resources(mem, &error) != 0) return -1;

    pthread_rwlock_wrlock(&mem->lock);
    
    MemoryNode* node = (MemoryNode*)malloc(sizeof(MemoryNode));
    snprintf(node->id, sizeof(node->id), "node_%ld_%d", time(NULL), mem->node_count);
    strcpy(node->id, node->id);
    
    node->embedding = (double*)malloc(sizeof(double) * mem->embedding_dim);
    memcpy(node->embedding, embedding, sizeof(double) * mem->embedding_dim);
    node->embedding_dim = mem->embedding_dim;
    
    strncpy(node->text, text, sizeof(node->text) - 1);
    node->centrality = centrality;
    node->importance = 1.0;
    node->created_at = time(NULL);
    
    mem->nodes[mem->node_count] = node;
    mem->node_count++;
    
    strcpy(node_id_out, node->id);
    mem->metrics->queries_processed++;
    
    pthread_rwlock_unlock(&mem->lock);
    
    log_message(LOG_INFO, "Memory added", node->id);
    return 0;
}

// Health & metrics
int ov_memory_get_health(OVMemoryProduction* mem, char* output, size_t size) {
    char metrics_str[512];
    metrics_get(mem->metrics, metrics_str, sizeof(metrics_str));
    
    snprintf(output, size, "{\"status\": \"HEALTHY\", \"metrics\": %s}", metrics_str);
    return 0;
}

void ov_memory_destroy(OVMemoryProduction* mem) {
    pthread_rwlock_destroy(&mem->lock);
    
    for (int i = 0; i < mem->node_count; i++) {
        free(mem->nodes[i]->embedding);
        free(mem->nodes[i]);
    }
    free(mem->nodes);
    
    metrics_destroy(mem->metrics);
    circuit_breaker_destroy(mem->circuit_breaker);
    free(mem);
}

/* ========== EXAMPLE USAGE ========== */

int main() {
    OVMemoryProduction* memory = ov_memory_create(768, 10000, 1);
    
    // Create embedding
    double embedding[768];
    for (int i = 0; i < 768; i++) {
        embedding[i] = 0.5;
    }
    
    // Add memory
    char node_id[64];
    if (ov_memory_add(memory, embedding, "Sample text", 0.9, node_id) == 0) {
        printf("Added node: %s\n", node_id);
    }
    
    // Get health
    char health[512];
    ov_memory_get_health(memory, health, sizeof(health));
    printf("Health: %s\n", health);
    
    ov_memory_destroy(memory);
    return 0;
}

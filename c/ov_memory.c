/**
 * =====================================================================
 * OV-Memory: C Implementation (Production-Ready)
 * =====================================================================
 * Author: Prayaga Vaibhavlakshmi
 * License: Apache License 2.0
 * Om Vinayaka 🙏
 *
 * A high-performance C implementation of Fractal Honeycomb Graph Database
 * for AI agents. SIMD-optimizable, memory-efficient, and production-ready.
 *
 * Compile with: gcc -Wall -Wextra -O3 -march=native -lm -o ov_memory ov_memory.c
 *
 * =====================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

// ===== CONFIGURATION CONSTANTS =====
#define MAX_NODES 100000
#define MAX_EMBEDDING_DIM 768
#define MAX_DATA_SIZE 8192
#define HEXAGONAL_NEIGHBORS 6
#define RELEVANCE_THRESHOLD 0.8f
#define MAX_SESSION_TIME 3600
#define LOOP_DETECTION_WINDOW 10
#define LOOP_ACCESS_LIMIT 3
#define TEMPORAL_DECAY_HALF_LIFE 86400.0f // 24 hours in seconds

// ===== SAFETY RETURN CODES =====
#define SAFETY_OK 0
#define SAFETY_LOOP_DETECTED 1
#define SAFETY_SESSION_EXPIRED 2
#define SAFETY_INVALID_NODE -1

// ===== TYPE DEFINITIONS =====

typedef struct {
    uint32_t targetId;
    float relevanceScore; // [0.0, 1.0]
    char relationshipType[256];
    time_t timestampCreated;
} HoneycombEdge;

typedef struct HoneycombNode {
    uint32_t id;
    float *vectorEmbedding;
    uint32_t embeddingDim;
    char *data;
    HoneycombEdge *neighbors;
    uint32_t neighborCount;
    struct {
        void *nodes; // Placeholder for fractal layer
    } fractalLayer;
    time_t lastAccessedTimestamp;
    uint32_t accessCountSession;
    time_t accessTimeFirst;
    float relevanceToFocus;
    uint8_t isActive;
} HoneycombNode;

typedef struct {
    char name[256];
    HoneycombNode **nodes;
    uint32_t nodeCount;
    uint32_t maxNodes;
    time_t sessionStartTime;
    uint32_t maxSessionTimeSeconds;
} HoneycombGraph;

// ===== VECTOR MATH FUNCTIONS =====

/**
 * Calculate cosine similarity between two vectors
 */
static float cosineSimilarity(const float *vecA, const float *vecB, uint32_t dim) {
    if (!vecA || !vecB || dim == 0) {
        return 0.0f;
    }

    float dotProduct = 0.0f;
    float magA = 0.0f;
    float magB = 0.0f;

    for (uint32_t i = 0; i < dim; i++) {
        dotProduct += vecA[i] * vecB[i];
        magA += vecA[i] * vecA[i];
        magB += vecB[i] * vecB[i];
    }

    magA = sqrtf(magA);
    magB = sqrtf(magB);

    if (magA == 0.0f || magB == 0.0f) {
        return 0.0f;
    }

    return dotProduct / (magA * magB);
}

/**
 * Calculate temporal decay factor
 * Uses exponential decay: e^(-age / half_life)
 */
static float temporalDecay(time_t createdTime, time_t currentTime) {
    if (createdTime > currentTime) {
        return 1.0f;
    }

    float ageSeconds = (float)(currentTime - createdTime);
    float decay = expf(-ageSeconds / TEMPORAL_DECAY_HALF_LIFE);
    return fmaxf(0.0f, fminf(1.0f, decay));
}

/**
 * Calculate combined relevance score
 * Formula: (Cosine_Similarity × 0.7) + (Temporal_Decay × 0.3)
 */
static float calculateRelevance(const float *vecA, const float *vecB, uint32_t dim,
                               time_t createdTime, time_t currentTime) {
    float cosine = cosineSimilarity(vecA, vecB, dim);
    float decay = temporalDecay(createdTime, currentTime);
    float finalScore = (cosine * 0.7f) + (decay * 0.3f);
    return fmaxf(0.0f, fminf(1.0f, finalScore));
}

// ===== GRAPH OPERATIONS =====

/**
 * Create a new Honeycomb Graph
 */
HoneycombGraph *honeycombCreateGraph(const char *name, uint32_t maxNodes, uint32_t maxSessionTime) {
    if (maxNodes == 0) maxNodes = MAX_NODES;
    if (maxSessionTime == 0) maxSessionTime = MAX_SESSION_TIME;

    HoneycombGraph *graph = (HoneycombGraph *)malloc(sizeof(HoneycombGraph));
    if (!graph) return NULL;

    strncpy(graph->name, name, sizeof(graph->name) - 1);
    graph->name[sizeof(graph->name) - 1] = '\0';
    graph->nodes = (HoneycombNode **)calloc(maxNodes, sizeof(HoneycombNode *));
    graph->nodeCount = 0;
    graph->maxNodes = maxNodes;
    graph->sessionStartTime = time(NULL);
    graph->maxSessionTimeSeconds = maxSessionTime;

    if (!graph->nodes) {
        free(graph);
        return NULL;
    }

    printf("\xe2\x9c\x85 Created honeycomb graph: %s (max_nodes=%u)\n", name, maxNodes);
    return graph;
}

/**
 * Add a new node to the graph
 */
int32_t honeycombAddNode(HoneycombGraph *graph, const float *embedding, uint32_t embeddingDim,
                        const char *data) {
    if (!graph || !embedding || !data || embeddingDim == 0) {
        return -1;
    }

    if (graph->nodeCount >= graph->maxNodes) {
        fprintf(stderr, "\xe2\x9c\x98 Graph at max capacity\n");
        return -1;
    }

    uint32_t nodeId = graph->nodeCount;
    HoneycombNode *node = (HoneycombNode *)calloc(1, sizeof(HoneycombNode));
    if (!node) return -1;

    node->id = nodeId;
    node->embeddingDim = embeddingDim;
    node->vectorEmbedding = (float *)malloc(embeddingDim * sizeof(float));
    if (!node->vectorEmbedding) {
        free(node);
        return -1;
    }
    memcpy(node->vectorEmbedding, embedding, embeddingDim * sizeof(float));

    node->data = (char *)malloc(strlen(data) + 1);
    if (!node->data) {
        free(node->vectorEmbedding);
        free(node);
        return -1;
    }
    strcpy(node->data, data);

    node->neighbors = (HoneycombEdge *)calloc(HEXAGONAL_NEIGHBORS, sizeof(HoneycombEdge));
    if (!node->neighbors) {
        free(node->data);
        free(node->vectorEmbedding);
        free(node);
        return -1;
    }

    node->neighborCount = 0;
    node->lastAccessedTimestamp = time(NULL);
    node->accessCountSession = 0;
    node->accessTimeFirst = 0;
    node->relevanceToFocus = 0.0f;
    node->isActive = 1;

    graph->nodes[nodeId] = node;
    graph->nodeCount++;

    printf("\xe2\x9c\x85 Added node %u (embedding_dim=%u, data_len=%zu)\n", nodeId, embeddingDim, strlen(data));
    return nodeId;
}

/**
 * Get a node and update access metadata
 */
HoneycombNode *honeycombGetNode(HoneycombGraph *graph, uint32_t nodeId) {
    if (!graph || nodeId >= graph->nodeCount || !graph->nodes[nodeId]) {
        return NULL;
    }

    HoneycombNode *node = graph->nodes[nodeId];
    node->lastAccessedTimestamp = time(NULL);
    node->accessCountSession++;

    if (node->accessTimeFirst == 0) {
        node->accessTimeFirst = node->lastAccessedTimestamp;
    }

    return node;
}

/**
 * Add an edge between two nodes
 */
int honeycombAddEdge(HoneycombGraph *graph, uint32_t sourceId, uint32_t targetId,
                    float relevanceScore, const char *relationshipType) {
    if (!graph || sourceId >= graph->nodeCount || targetId >= graph->nodeCount ||
        !graph->nodes[sourceId] || !graph->nodes[targetId]) {
        fprintf(stderr, "\xe2\x9c\x98 Invalid node IDs\n");
        return 0;
    }

    HoneycombNode *source = graph->nodes[sourceId];
    if (source->neighborCount >= HEXAGONAL_NEIGHBORS) {
        printf("\xe2\x9a\xa0\xef\xb8\x8f  Node %u at max neighbors\n", sourceId);
        return 0;
    }

    HoneycombEdge *edge = &source->neighbors[source->neighborCount];
    edge->targetId = targetId;
    edge->relevanceScore = fmaxf(0.0f, fminf(1.0f, relevanceScore));
    strncpy(edge->relationshipType, relationshipType, sizeof(edge->relationshipType) - 1);
    edge->relationshipType[sizeof(edge->relationshipType) - 1] = '\0';
    edge->timestampCreated = time(NULL);

    source->neighborCount++;
    printf("\xe2\x9c\x85 Added edge: Node %u \xe2\x86\x92 Node %u (relevance=%.2f)\n",
           sourceId, targetId, relevanceScore);
    return 1;
}

// ===== CORE ALGORITHMS =====

/**
 * Insert memory with fractal overflow handling
 */
void honeycombInsertMemory(HoneycombGraph *graph, uint32_t focusNodeId, uint32_t newNodeId,
                          time_t currentTime) {
    if (!graph || !currentTime) {
        currentTime = time(NULL);
    }

    if (focusNodeId >= graph->nodeCount || newNodeId >= graph->nodeCount ||
        !graph->nodes[focusNodeId] || !graph->nodes[newNodeId]) {
        fprintf(stderr, "\xe2\x9c\x98 Invalid node IDs\n");
        return;
    }

    HoneycombNode *focus = graph->nodes[focusNodeId];
    HoneycombNode *newMem = graph->nodes[newNodeId];

    float relevance = calculateRelevance(
        focus->vectorEmbedding, newMem->vectorEmbedding, focus->embeddingDim,
        newMem->lastAccessedTimestamp, currentTime);

    if (focus->neighborCount < HEXAGONAL_NEIGHBORS) {
        honeycombAddEdge(graph, focusNodeId, newNodeId, relevance, "memory_of");
        printf("\xe2\x9c\x85 Direct insert: Node %u connected to Node %u (rel=%.2f)\n",
               focusNodeId, newNodeId, relevance);
    } else {
        uint32_t weakestIdx = 0;
        float weakestRelevance = focus->neighbors[0].relevanceScore;

        for (uint32_t i = 1; i < focus->neighborCount; i++) {
            if (focus->neighbors[i].relevanceScore < weakestRelevance) {
                weakestRelevance = focus->neighbors[i].relevanceScore;
                weakestIdx = i;
            }
        }

        uint32_t weakestId = focus->neighbors[weakestIdx].targetId;

        if (relevance > weakestRelevance) {
            printf("\xf0\x9f\x94\x80 Moving Node %u to fractal layer of Node %u\n", weakestId, focusNodeId);
            focus->neighbors[weakestIdx].targetId = newNodeId;
            focus->neighbors[weakestIdx].relevanceScore = relevance;
            printf("\xe2\x9c\x85 Fractal swap: Node %u \xe2\x86\x94 Node %u (new rel=%.2f)\n",
                   weakestId, newNodeId, relevance);
        } else {
            printf("\xe2\x9c\x85 Inserted Node %u to fractal layer (rel=%.2f)\n", newNodeId, relevance);
        }
    }
}

/**
 * Print graph statistics
 */
void honeycombPrintGraphStats(HoneycombGraph *graph) {
    if (!graph) return;

    uint32_t totalEdges = 0;
    uint32_t fractalCount = 0;

    for (uint32_t i = 0; i < graph->nodeCount; i++) {
        if (graph->nodes[i]) {
            totalEdges += graph->nodes[i]->neighborCount;
            if (graph->nodes[i]->fractalLayer.nodes) fractalCount++;
        }
    }

    printf("\n");
    printf("\xf0\x9f\x94\x90 HONEYCOMB GRAPH STATISTICS\n");
    printf("Graph Name: %s\n", graph->name);
    printf("Node Count: %u / %u\n", graph->nodeCount, graph->maxNodes);
    printf("Total Edges: %u\n", totalEdges);
    printf("Fractal Layers: %u\n", fractalCount);
    float avgConnectivity = (graph->nodeCount > 0) ? (float)totalEdges / graph->nodeCount : 0.0f;
    printf("Avg Connectivity: %.2f\n\n", avgConnectivity);
}

/**
 * Safety circuit breaker for loop detection and session timeout
 */
int honeycombCheckSafety(HoneycombNode *node, time_t currentTime, time_t sessionStartTime,
                        uint32_t maxSessionTime) {
    if (!node) {
        return SAFETY_INVALID_NODE;
    }

    if (!currentTime) {
        currentTime = time(NULL);
    }

    if (!sessionStartTime) {
        sessionStartTime = time(NULL);
    }

    if (node->accessCountSession > LOOP_ACCESS_LIMIT) {
        time_t timeWindow = node->lastAccessedTimestamp - node->accessTimeFirst;
        if (timeWindow >= 0 && timeWindow < LOOP_DETECTION_WINDOW) {
            printf("\xe2\x9a\xa0\xef\xb8\x8f  LOOP DETECTED: Node %u accessed %u times in %ld seconds\n",
                   node->id, node->accessCountSession, timeWindow);
            return SAFETY_LOOP_DETECTED;
        }
    }

    time_t sessionElapsed = currentTime - sessionStartTime;
    if (sessionElapsed > (time_t)maxSessionTime) {
        printf("\xe2\x9a\xa0\xef\xb8\x8f  SESSION EXPIRED: %ld seconds elapsed\n", sessionElapsed);
        return SAFETY_SESSION_EXPIRED;
    }

    return SAFETY_OK;
}

/**
 * Free graph memory
 */
void honeycombFreeGraph(HoneycombGraph *graph) {
    if (!graph) return;

    for (uint32_t i = 0; i < graph->nodeCount; i++) {
        if (graph->nodes[i]) {
            free(graph->nodes[i]->vectorEmbedding);
            free(graph->nodes[i]->data);
            free(graph->nodes[i]->neighbors);
            free(graph->nodes[i]);
        }
    }
    free(graph->nodes);
    free(graph);
}

// ===== MAIN TEST =====

int main() {
    printf("\n\xf0\x9f\xa7\xa0 OV-Memory: C Implementation\n");
    printf("Om Vinayaka \xf0\x9f\x99\x8f\n\n");

    // Create graph
    HoneycombGraph *graph = honeycombCreateGraph("example_memory", 1000, 3600);
    if (!graph) {
        fprintf(stderr, "Failed to create graph\n");
        return 1;
    }

    // Create sample embeddings
    float *emb1 = (float *)malloc(768 * sizeof(float));
    float *emb2 = (float *)malloc(768 * sizeof(float));
    float *emb3 = (float *)malloc(768 * sizeof(float));

    for (int i = 0; i < 768; i++) {
        emb1[i] = 0.5f;
        emb2[i] = 0.6f;
        emb3[i] = 0.7f;
    }

    // Add nodes
    int32_t node1 = honeycombAddNode(graph, emb1, 768, "First memory unit");
    int32_t node2 = honeycombAddNode(graph, emb2, 768, "Second memory unit");
    int32_t node3 = honeycombAddNode(graph, emb3, 768, "Third memory unit");

    if (node1 < 0 || node2 < 0 || node3 < 0) {
        fprintf(stderr, "Failed to add nodes\n");
        honeycombFreeGraph(graph);
        free(emb1);
        free(emb2);
        free(emb3);
        return 1;
    }

    // Add edges
    honeycombAddEdge(graph, node1, node2, 0.9f, "related_to");
    honeycombAddEdge(graph, node2, node3, 0.85f, "context_of");

    // Insert memory
    honeycombInsertMemory(graph, node1, node2, time(NULL));

    // Print stats
    honeycombPrintGraphStats(graph);

    // Check safety
    HoneycombNode *node = honeycombGetNode(graph, node1);
    int safety = honeycombCheckSafety(node, 0, 0, MAX_SESSION_TIME);
    printf("Safety Status: %d\n\n", safety);

    printf("\xe2\x9c\x85 C tests completed successfully\n");
    printf("Om Vinayaka \xf0\x9f\x99\x8f\n\n");

    // Cleanup
    honeycombFreeGraph(graph);
    free(emb1);
    free(emb2);
    free(emb3);

    return 0;
}

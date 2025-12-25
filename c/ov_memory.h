/**
 * =====================================================================
 * OV-Memory v1.1: Metabolic & Centroid Upgrade (Header)
 * =====================================================================
 * Author: Prayaga Vaibhavlakshmi
 * License: Apache License 2.0
 * Om Vinayaka 🙏
 *
 * Production-ready Fractal Honeycomb Graph Database with:
 * - Resource-Aware Metabolism Engine
 * - Centroid-Based Indexing (O(1) entry search)
 * - Binary Persistence with Fractal Layer Recursion
 * - Cross-Session Hydration
 *
 * =====================================================================
 */

#ifndef OV_MEMORY_H
#define OV_MEMORY_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <pthread.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ===== CONFIGURATION CONSTANTS =====
#define MAX_NODES 100000
#define MAX_EMBEDDING_DIM 768
#define MAX_DATA_SIZE 8192
#define HEXAGONAL_NEIGHBORS 6
#define RELEVANCE_THRESHOLD 0.8f
#define MAX_SESSION_TIME 3600
#define LOOP_DETECTION_WINDOW 10
#define LOOP_ACCESS_LIMIT 3
#define TEMPORAL_DECAY_HALF_LIFE 86400.0f
#define CENTROID_COUNT 5
#define CENTROID_SCAN_PERCENTAGE 0.05f

// ===== AUDIT THRESHOLDS =====
#define AUDIT_SEMANTIC_TRIGGER 1260 // 21 minutes in seconds
#define AUDIT_FRACTAL_TRIGGER 1080  // 18 minutes
#define AUDIT_CRITICAL_SEAL_TRIGGER 300 // 5 minutes

// ===== SAFETY RETURN CODES =====
#define SAFETY_OK 0
#define SAFETY_LOOP_DETECTED 1
#define SAFETY_SESSION_EXPIRED 2
#define SAFETY_INVALID_NODE -1

// ===== METABOLIC STATE ENUM =====
typedef enum {
    METABOLIC_HEALTHY = 0,
    METABOLIC_STRESSED = 1,
    METABOLIC_CRITICAL = 2
} MetabolicState;

// ===== TYPE DEFINITIONS =====

/**
 * AgentMetabolism: Resource constraints and dynamic state
 */
typedef struct {
    int messages_remaining;     // Messages left before hard stop
    int minutes_remaining;      // Session time left (seconds)
    bool is_api_mode;          // API rate-limited vs Direct account
    float context_availability; // Percentage of context window used (0-100)
    float metabolic_weight;    // Relevance multiplier (0.5 - 1.5)
    MetabolicState state;      // HEALTHY, STRESSED, CRITICAL
    time_t audit_last_run;     // Timestamp of last metabolic audit
} AgentMetabolism;

/**
 * HoneycombEdge: Connection between two nodes
 */
typedef struct {
    uint32_t targetId;
    float relevanceScore;
    char relationshipType[256];
    time_t timestampCreated;
} HoneycombEdge;

/**
 * HoneycombNode: Individual memory unit
 */
typedef struct HoneycombNode {
    uint32_t id;
    float *vectorEmbedding;
    uint32_t embeddingDim;
    char *data;
    HoneycombEdge *neighbors;
    uint32_t neighborCount;
    struct HoneycombGraph *fractalLayer;
    time_t lastAccessedTimestamp;
    uint32_t accessCountSession;
    time_t accessTimeFirst;
    float relevanceToFocus;
    float metabolic_weight;    // Individual metabolism score
    uint8_t isActive;
    uint8_t isFractalSeed;     // Marker for compressed session node
} HoneycombNode;

/**
 * CentroidMap: Fast entry indexing via hub nodes
 */
typedef struct {
    uint32_t *hubNodeIds;
    float *hubCentrality;
    uint32_t hubCount;
    uint32_t maxHubs;
} CentroidMap;

/**
 * HoneycombGraph: Main container with v1.1 enhancements
 */
typedef struct HoneycombGraph {
    char name[256];
    HoneycombNode **nodes;
    uint32_t nodeCount;
    uint32_t maxNodes;
    time_t sessionStartTime;
    uint32_t maxSessionTimeSeconds;
    
    // ===== v1.1 Additions =====
    AgentMetabolism metabolism;
    CentroidMap centroidMap;
    pthread_mutex_t lock;       // Thread safety
    uint8_t isDirty;           // Needs persistence
} HoneycombGraph;

// ===== FUNCTION DECLARATIONS (v1.0) =====

// Graph creation/destruction
HoneycombGraph *honeycombCreateGraph(const char *name, uint32_t maxNodes, uint32_t maxSessionTime);
void honeycombFreeGraph(HoneycombGraph *graph);

// Node operations
int32_t honeycombAddNode(HoneycombGraph *graph, const float *embedding, uint32_t embeddingDim, const char *data);
HoneycombNode *honeycombGetNode(HoneycombGraph *graph, uint32_t nodeId);
int honeycombAddEdge(HoneycombGraph *graph, uint32_t sourceId, uint32_t targetId, float relevanceScore, const char *relationshipType);

// Memory management
void honeycombInsertMemory(HoneycombGraph *graph, uint32_t focusNodeId, uint32_t newNodeId, time_t currentTime);
int honeycombCheckSafety(HoneycombNode *node, time_t currentTime, time_t sessionStartTime, uint32_t maxSessionTime);

// Utilities
void honeycombPrintGraphStats(HoneycombGraph *graph);

// ===== FUNCTION DECLARATIONS (v1.1 - Metabolism Engine) =====

/**
 * Initialize metabolism for a graph with resource constraints
 */
void honeycombInitializeMetabolism(HoneycombGraph *graph, int maxMessages, int maxMinutes, bool isApiMode);

/**
 * Update metabolism state based on current resource usage
 */
void honeycombUpdateMetabolism(HoneycombGraph *graph, int messagesUsed, int secondsElapsed, float contextUsed);

/**
 * Calculate resource-aware relevance score
 * Formula: R_final = (S_sem * 0.6) + (T_decay * 0.2) + (R_resource * 0.2)
 */
float honeycombCalculateMetabolicRelevance(
    const float *vecA, const float *vecB, uint32_t dim,
    time_t createdTime, time_t currentTime,
    float resourceAvailability, float metabolicWeight
);

/**
 * Run metabolic audits on specific thresholds
 */
void honeycombMetabolicAudit(HoneycombGraph *graph);

// ===== FUNCTION DECLARATIONS (v1.1 - Centroid Indexing) =====

/**
 * Initialize centroid map with hub nodes
 */
void honeycombInitializeCentroidMap(HoneycombGraph *graph);

/**
 * Update centrality scores for all nodes
 */
void honeycombRecalculateCentrality(HoneycombGraph *graph);

/**
 * Fast entry point search using centroid map (O(k) where k = hub count)
 */
uint32_t honeycombFindMostRelevantNode(HoneycombGraph *graph, const float *queryVector, uint32_t embeddingDim);

// ===== FUNCTION DECLARATIONS (v1.1 - Binary Persistence) =====

/**
 * Save graph (including nested fractal layers) to binary file
 * Format: [Header: "Om Vinayaka"] [Metadata] [Nodes] [Edges] [Fractal Layers]
 */
int honeycombSaveBinary(HoneycombGraph *graph, const char *filename);

/**
 * Load graph from binary file with full fractal reconstruction
 */
HoneycombGraph *honeycombLoadBinary(const char *filename);

/**
 * Export graph to GraphViz format with metabolic coloring
 * Red: metabolic_weight < 0.5, Orange: 0.5-0.8, Green: > 0.8
 * Fractal seeds shown as double octagons
 */
void honeycombExportGraphviz(HoneycombGraph *graph, const char *filename);

// ===== FUNCTION DECLARATIONS (v1.1 - Cross-Session Hydration) =====

/**
 * Hydrate session from previous fractal seeds
 * Scans disk for saved seeds matching userVector with cosine similarity > 0.85
 */
int honeycombHydrateSession(HoneycombGraph *graph, const float *userVector, uint32_t embeddingDim, const char *sessionDir);

/**
 * Create and compress a "Fractal Seed" (summary node) before session end
 */
uint32_t honeycombCreateFractalSeed(HoneycombGraph *graph, const char *seedLabel);

// ===== UTILITY FUNCTIONS =====

void honeycombPrintMetabolicState(HoneycombGraph *graph);

#ifdef __cplusplus
}
#endif

#endif // OV_MEMORY_H

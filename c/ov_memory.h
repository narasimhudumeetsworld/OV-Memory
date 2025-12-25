/*
 * =====================================================================
 * OV-Memory: Fractal Honeycomb Graph Database
 * =====================================================================
 * Author: Prayaga Vaibhavlakshmi
 * License: Apache License 2.0
 * Om Vinayaka 🙏
 *
 * A high-performance, C-based memory system for AI agents using a
 * Fractal Honeycomb topology for drift-resistant, bounded-connectivity
 * semantic storage.
 * =====================================================================
 */

#ifndef OV_MEMORY_H
#define OV_MEMORY_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <stdbool.h>

/* ===== CONFIGURATION MACROS ===== */
#define MAX_NODES 100000
#define MAX_EMBEDDING_DIM 768
#define MAX_DATA_SIZE 8192
#define MAX_RELATIONSHIP_TYPE 64
#define HEXAGONAL_NEIGHBORS 6
#define RELEVANCE_THRESHOLD 0.8
#define MAX_SESSION_TIME 3600
#define LOOP_DETECTION_WINDOW 10
#define LOOP_ACCESS_LIMIT 3
#define EMBEDDING_DIM_DEFAULT 768
#define TEMPORAL_DECAY_HALF_LIFE 86400.0f // 24 hours in seconds

/* ===== SAFETY RETURN CODES ===== */
#define SAFETY_OK 0
#define SAFETY_LOOP_DETECTED 1
#define SAFETY_SESSION_EXPIRED 2
#define SAFETY_INVALID_NODE -1

/* ===== FORWARD DECLARATIONS ===== */
typedef struct HoneycombEdge HoneycombEdge;
typedef struct HoneycombNode HoneycombNode;
typedef struct HoneycombGraph HoneycombGraph;

/* ===== HONEYCOMB EDGE STRUCTURE ===== */
typedef struct HoneycombEdge {
    int target_id;                              // ID of connected node
    float relevance_score;                      // [0.0, 1.0]
    char relationship_type[MAX_RELATIONSHIP_TYPE];
    long timestamp_created;                     // UNIX timestamp
} HoneycombEdge;

/* ===== HONEYCOMB NODE STRUCTURE ===== */
typedef struct HoneycombNode {
    int id;                                     // Unique identifier
    float* vector_embedding;                    // Embedding array (768-dim default)
    int embedding_dim;                          // Actual dimension used
    char* data;                                 // Text payload
    int data_length;                            // Length of data
    
    // Hexagonal connectivity
    HoneycombEdge neighbors[HEXAGONAL_NEIGHBORS]; // Max 6 connections
    int neighbor_count;                         // Current neighbor count
    
    // Fractal layer for overflow
    struct HoneycombGraph* fractal_layer;       // Nested graph (sub-branches)
    
    // Safety and metadata
    long last_accessed_timestamp;               // UNIX timestamp (seconds)
    int access_count_session;                   // Times accessed this session
    long access_time_first;                     // First access time (for loop detection)
    float relevance_to_focus;                   // Relevance to current query
    bool is_active;                             // Node is in use
} HoneycombNode;

/* ===== HONEYCOMB GRAPH CONTAINER ===== */
typedef struct HoneycombGraph {
    HoneycombNode* nodes;                       // Dynamic array of nodes
    int node_count;                             // Current number of nodes
    int max_nodes;                              // Max capacity
    char graph_name[128];                       // Graph identifier
    
    // Session safety
    long session_start_time;                    // When session started
    int max_session_time_seconds;               // Max session duration
    
    // Thread safety
    pthread_mutex_t graph_lock;                 // Global graph lock
    pthread_mutex_t* node_locks;                // Per-node locks (optional)
} HoneycombGraph;

/* ===== VECTOR MATH FUNCTIONS ===== */
float cosine_similarity(float* vec_a, float* vec_b, int dim);
float temporal_decay(long created_time, long current_time);
float calculate_relevance(float* vec_a, float* vec_b, int dim, 
                         long created_time, long current_time);

/* ===== GRAPH CREATION AND LIFECYCLE ===== */
HoneycombGraph* honeycomb_create_graph(const char* name, int max_nodes, int max_session_time);
void honeycomb_free_graph(HoneycombGraph* graph);
void honeycomb_reset_session(HoneycombGraph* graph);

/* ===== NODE OPERATIONS ===== */
int honeycomb_add_node(HoneycombGraph* graph, float* embedding, int embedding_dim,
                      const char* data, int data_length);
HoneycombNode* honeycomb_get_node(HoneycombGraph* graph, int node_id);
void honeycomb_update_node_data(HoneycombGraph* graph, int node_id, 
                               const char* new_data, int data_length);

/* ===== EDGE OPERATIONS ===== */
bool honeycomb_add_edge(HoneycombGraph* graph, int source_id, int target_id,
                       float relevance_score, const char* relationship_type);
bool honeycomb_remove_edge(HoneycombGraph* graph, int source_id, int target_id);

/* ===== CORE ALGORITHMS ===== */
void honeycomb_insert_memory(HoneycombGraph* graph, int focus_node_id, 
                            int new_node_id, long current_time);
char* honeycomb_get_jit_context(HoneycombGraph* graph, float* query_vector, 
                               int embedding_dim, int max_tokens);
int honeycomb_check_safety(HoneycombNode* node, long current_time,
                          long session_start_time, int max_session_time);

/* ===== TRAVERSAL AND SEARCH ===== */
int* honeycomb_find_neighbors(HoneycombGraph* graph, int node_id, int* count);
int honeycomb_find_most_relevant_node(HoneycombGraph* graph, float* query_vector, 
                                      int embedding_dim);
void honeycomb_traverse_by_relevance(HoneycombGraph* graph, int start_node_id,
                                    float min_relevance);

/* ===== UTILITY AND DEBUGGING ===== */
void honeycomb_print_node(HoneycombGraph* graph, int node_id);
void honeycomb_print_graph_stats(HoneycombGraph* graph);
void honeycomb_compact_memory(HoneycombGraph* graph);

#endif // OV_MEMORY_H

/*
 * =====================================================================
 * OV-Memory: C Implementation
 * =====================================================================
 * Author: Prayaga Vaibhavlakshmi
 * License: Apache License 2.0
 * Om Vinayaka 🙏
 *
 * High-performance C implementation of Fractal Honeycomb Graph Database
 * =====================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* Configuration constants */
#define MAX_NODES 100000
#define MAX_EMBEDDING_DIM 768
#define MAX_DATA_SIZE 8192
#define HEXAGONAL_NEIGHBORS 6
#define RELEVANCE_THRESHOLD 0.8
#define MAX_SESSION_TIME 3600
#define LOOP_DETECTION_WINDOW 10
#define LOOP_ACCESS_LIMIT 3
#define TEMPORAL_DECAY_HALF_LIFE 86400.0

/* Safety return codes */
#define SAFETY_OK 0
#define SAFETY_LOOP_DETECTED 1
#define SAFETY_SESSION_EXPIRED 2
#define SAFETY_INVALID_NODE -1

/* Edge structure */
typedef struct {
    int target_id;
    float relevance_score;
    char relationship_type[64];
    time_t timestamp_created;
} HoneycombEdge;

/* Node structure */
typedef struct {
    int id;
    float *vector_embedding;
    int embedding_dim;
    char data[MAX_DATA_SIZE];
    int data_length;
    HoneycombEdge *neighbors;
    int neighbor_count;
    time_t last_accessed_timestamp;
    int access_count_session;
    time_t access_time_first;
    float relevance_to_focus;
    int is_active;
} HoneycombNode;

/* Graph structure */
typedef struct {
    char name[256];
    HoneycombNode *nodes;
    int node_count;
    int max_nodes;
    time_t session_start_time;
    int max_session_time_seconds;
} HoneycombGraph;

/* ===== VECTOR MATH FUNCTIONS ===== */

/**
 * Calculate cosine similarity between two vectors
 */
float cosine_similarity(float *vec_a, float *vec_b, int dim) {
    if (!vec_a || !vec_b || dim <= 0) {
        return 0.0f;
    }

    float dot_product = 0.0f;
    float mag_a = 0.0f;
    float mag_b = 0.0f;

    for (int i = 0; i < dim; i++) {
        dot_product += vec_a[i] * vec_b[i];
        mag_a += vec_a[i] * vec_a[i];
        mag_b += vec_b[i] * vec_b[i];
    }

    mag_a = sqrtf(mag_a);
    mag_b = sqrtf(mag_b);

    if (mag_a == 0.0f || mag_b == 0.0f) {
        return 0.0f;
    }

    return dot_product / (mag_a * mag_b);
}

/**
 * Calculate temporal decay factor
 */
float temporal_decay(time_t created_time, time_t current_time) {
    if (created_time > current_time) {
        return 1.0f;
    }

    double age_seconds = difftime(current_time, created_time);
    float decay = (float)exp(-age_seconds / TEMPORAL_DECAY_HALF_LIFE);
    return fmax(0.0f, fmin(1.0f, decay));
}

/**
 * Calculate combined relevance score
 */
float calculate_relevance(float *vec_a, float *vec_b, int dim,
                         time_t created_time, time_t current_time) {
    float cosine = cosine_similarity(vec_a, vec_b, dim);
    float decay = temporal_decay(created_time, current_time);
    float final_score = (cosine * 0.7f) + (decay * 0.3f);
    return fmax(0.0f, fmin(1.0f, final_score));
}

/* ===== GRAPH OPERATIONS ===== */

/**
 * Create a new Honeycomb Graph
 */
HoneycombGraph* honeycomb_create_graph(const char *name, int max_nodes, int max_session_time) {
    HoneycombGraph *graph = (HoneycombGraph *)malloc(sizeof(HoneycombGraph));
    if (!graph) return NULL;

    strncpy(graph->name, name, 255);
    graph->nodes = (HoneycombNode *)calloc(max_nodes, sizeof(HoneycombNode));
    if (!graph->nodes) {
        free(graph);
        return NULL;
    }

    graph->node_count = 0;
    graph->max_nodes = max_nodes;
    graph->session_start_time = time(NULL);
    graph->max_session_time_seconds = max_session_time;

    printf("✅ Created honeycomb graph: %s (max_nodes=%d)\n", name, max_nodes);
    return graph;
}

/**
 * Add a new node to the graph
 */
int honeycomb_add_node(HoneycombGraph *graph, float *embedding, int embedding_dim,
                     const char *data, int data_length) {
    if (!graph || graph->node_count >= graph->max_nodes) {
        printf("❌ Graph at max capacity\n");
        return -1;
    }

    int node_id = graph->node_count;
    HoneycombNode *node = &graph->nodes[node_id];

    node->id = node_id;
    node->vector_embedding = (float *)malloc(embedding_dim * sizeof(float));
    if (!node->vector_embedding) return -1;

    memcpy(node->vector_embedding, embedding, embedding_dim * sizeof(float));
    node->embedding_dim = embedding_dim;
    
    strncpy(node->data, data, MAX_DATA_SIZE - 1);
    node->data_length = data_length > MAX_DATA_SIZE ? MAX_DATA_SIZE - 1 : data_length;

    node->neighbors = (HoneycombEdge *)calloc(HEXAGONAL_NEIGHBORS, sizeof(HoneycombEdge));
    node->neighbor_count = 0;
    node->last_accessed_timestamp = time(NULL);
    node->access_count_session = 0;
    node->access_time_first = 0;
    node->relevance_to_focus = 0.0f;
    node->is_active = 1;

    graph->node_count++;
    printf("✅ Added node %d (embedding_dim=%d, data_len=%d)\n", node_id, embedding_dim, data_length);
    return node_id;
}

/**
 * Add an edge between two nodes
 */
int honeycomb_add_edge(HoneycombGraph *graph, int source_id, int target_id,
                     float relevance_score, const char *relationship_type) {
    if (!graph || source_id < 0 || target_id < 0 || 
        source_id >= graph->node_count || target_id >= graph->node_count) {
        return 0;
    }

    HoneycombNode *source = &graph->nodes[source_id];
    if (source->neighbor_count >= HEXAGONAL_NEIGHBORS) {
        printf("⚠️  Node %d at max neighbors\n", source_id);
        return 0;
    }

    HoneycombEdge *edge = &source->neighbors[source->neighbor_count];
    edge->target_id = target_id;
    edge->relevance_score = fmax(0.0f, fmin(1.0f, relevance_score));
    strncpy(edge->relationship_type, relationship_type, 63);
    edge->timestamp_created = time(NULL);

    source->neighbor_count++;
    printf("✅ Added edge: Node %d → Node %d (relevance=%.2f)\n", 
           source_id, target_id, relevance_score);
    return 1;
}

/**
 * Print graph statistics
 */
void honeycomb_print_graph_stats(HoneycombGraph *graph) {
    if (!graph) return;

    int total_edges = 0;
    for (int i = 0; i < graph->node_count; i++) {
        total_edges += graph->nodes[i].neighbor_count;
    }

    printf("\n╔════════════════════════════════════════╗\n");
    printf("║  HONEYCOMB GRAPH STATISTICS              ║\n");
    printf("╚════════════════════════════════════════╝\n");
    printf("Graph Name: %s\n", graph->name);
    printf("Node Count: %d / %d\n", graph->node_count, graph->max_nodes);
    printf("Total Edges: %d\n", total_edges);
    
    if (graph->node_count > 0) {
        float avg_connectivity = (float)total_edges / (float)graph->node_count;
        printf("Avg Connectivity: %.2f\n", avg_connectivity);
    }
    printf("\n");
}

/**
 * Free graph memory
 */
void honeycomb_free_graph(HoneycombGraph *graph) {
    if (!graph) return;

    for (int i = 0; i < graph->node_count; i++) {
        if (graph->nodes[i].vector_embedding) {
            free(graph->nodes[i].vector_embedding);
        }
        if (graph->nodes[i].neighbors) {
            free(graph->nodes[i].neighbors);
        }
    }

    if (graph->nodes) {
        free(graph->nodes);
    }
    free(graph);
}

/* ===== EXAMPLE USAGE ===== */

int main(void) {
    printf("\n🧠 OV-Memory: C Implementation\n");
    printf("Om Vinayaka 🙏\n\n");

    /* Create graph */
    HoneycombGraph *graph = honeycomb_create_graph("example_memory", 1000, 3600);
    if (!graph) {
        fprintf(stderr, "Failed to create graph\n");
        return 1;
    }

    /* Create sample embeddings */
    float *emb1 = (float *)malloc(768 * sizeof(float));
    float *emb2 = (float *)malloc(768 * sizeof(float));
    float *emb3 = (float *)malloc(768 * sizeof(float));

    for (int i = 0; i < 768; i++) {
        emb1[i] = 0.5f;
        emb2[i] = 0.6f;
        emb3[i] = 0.7f;
    }

    /* Add nodes */
    int node1 = honeycomb_add_node(graph, emb1, 768, "First memory unit", 17);
    int node2 = honeycomb_add_node(graph, emb2, 768, "Second memory unit", 18);
    int node3 = honeycomb_add_node(graph, emb3, 768, "Third memory unit", 17);

    /* Add edges */
    honeycomb_add_edge(graph, node1, node2, 0.9f, "related_to");
    honeycomb_add_edge(graph, node2, node3, 0.85f, "context_of");

    /* Print stats */
    honeycomb_print_graph_stats(graph);

    printf("✅ C tests completed successfully\n");
    printf("Om Vinayaka 🙏\n\n");

    /* Cleanup */
    free(emb1);
    free(emb2);
    free(emb3);
    honeycomb_free_graph(graph);

    return 0;
}

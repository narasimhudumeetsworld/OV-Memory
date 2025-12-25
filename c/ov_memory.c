/*
 * =====================================================================
 * OV-Memory: Implementation
 * =====================================================================
 * Full C implementation of Fractal Honeycomb Graph Database
 * License: Apache License 2.0
 * =====================================================================
 */

#include "ov_memory.h"

/* ===== VECTOR MATH: COSINE SIMILARITY ===== */
float cosine_similarity(float* vec_a, float* vec_b, int dim) {
    if (!vec_a || !vec_b || dim <= 0) return 0.0f;
    
    float dot_product = 0.0f;
    float mag_a = 0.0f, mag_b = 0.0f;
    
    // Calculate dot product and magnitudes
    for (int i = 0; i < dim; i++) {
        dot_product += vec_a[i] * vec_b[i];
        mag_a += vec_a[i] * vec_a[i];
        mag_b += vec_b[i] * vec_b[i];
    }
    
    mag_a = sqrtf(mag_a);
    mag_b = sqrtf(mag_b);
    
    // Avoid division by zero
    if (mag_a == 0.0f || mag_b == 0.0f) return 0.0f;
    
    return dot_product / (mag_a * mag_b);
}

/* ===== TEMPORAL DECAY FACTOR ===== */
float temporal_decay(long created_time, long current_time) {
    if (created_time > current_time) return 1.0f;
    
    long age_seconds = current_time - created_time;
    float age_f = (float)age_seconds;
    
    // Exponential decay: e^(-age / half_life)
    float decay = expf(-age_f / TEMPORAL_DECAY_HALF_LIFE);
    
    return fmaxf(0.0f, fminf(1.0f, decay));
}

/* ===== CALCULATE RELEVANCE (COMBINED SCORE) ===== */
float calculate_relevance(float* vec_a, float* vec_b, int dim,
                         long created_time, long current_time) {
    float cosine = cosine_similarity(vec_a, vec_b, dim);
    float decay = temporal_decay(created_time, current_time);
    
    // Final_Score = (Cosine × 0.7) + (Decay × 0.3)
    float final_score = (cosine * 0.7f) + (decay * 0.3f);
    
    return fmaxf(0.0f, fminf(1.0f, final_score));
}

/* ===== GRAPH CREATION ===== */
HoneycombGraph* honeycomb_create_graph(const char* name, int max_nodes, int max_session_time) {
    HoneycombGraph* graph = (HoneycombGraph*)malloc(sizeof(HoneycombGraph));
    if (!graph) return NULL;
    
    graph->nodes = (HoneycombNode*)calloc(max_nodes, sizeof(HoneycombNode));
    if (!graph->nodes) {
        free(graph);
        return NULL;
    }
    
    strncpy(graph->graph_name, name, 127);
    graph->graph_name[127] = '\0';
    graph->node_count = 0;
    graph->max_nodes = max_nodes;
    graph->session_start_time = time(NULL);
    graph->max_session_time_seconds = max_session_time;
    
    // Initialize thread locks
    pthread_mutex_init(&graph->graph_lock, NULL);
    graph->node_locks = (pthread_mutex_t*)calloc(max_nodes, sizeof(pthread_mutex_t));
    for (int i = 0; i < max_nodes; i++) {
        pthread_mutex_init(&graph->node_locks[i], NULL);
    }
    
    printf("✅ Created honeycomb graph: %s (max_nodes=%d)\n", name, max_nodes);
    return graph;
}

/* ===== FREE GRAPH ===== */
void honeycomb_free_graph(HoneycombGraph* graph) {
    if (!graph) return;
    
    pthread_mutex_lock(&graph->graph_lock);
    
    for (int i = 0; i < graph->node_count; i++) {
        if (graph->nodes[i].data) free(graph->nodes[i].data);
        if (graph->nodes[i].vector_embedding) free(graph->nodes[i].vector_embedding);
        if (graph->nodes[i].fractal_layer) honeycomb_free_graph(graph->nodes[i].fractal_layer);
        pthread_mutex_destroy(&graph->node_locks[i]);
    }
    
    free(graph->nodes);
    free(graph->node_locks);
    pthread_mutex_unlock(&graph->graph_lock);
    pthread_mutex_destroy(&graph->graph_lock);
    
    free(graph);
    printf("✅ Freed honeycomb graph\n");
}

/* ===== ADD NODE ===== */
int honeycomb_add_node(HoneycombGraph* graph, float* embedding, int embedding_dim,
                      const char* data, int data_length) {
    if (!graph || !embedding || !data) return -1;
    
    pthread_mutex_lock(&graph->graph_lock);
    
    if (graph->node_count >= graph->max_nodes) {
        pthread_mutex_unlock(&graph->graph_lock);
        printf("❌ Graph at max capacity\n");
        return -1;
    }
    
    int node_id = graph->node_count;
    HoneycombNode* node = &graph->nodes[node_id];
    
    node->id = node_id;
    node->embedding_dim = embedding_dim;
    node->vector_embedding = (float*)malloc(embedding_dim * sizeof(float));
    memcpy(node->vector_embedding, embedding, embedding_dim * sizeof(float));
    
    node->data_length = (data_length < MAX_DATA_SIZE) ? data_length : MAX_DATA_SIZE;
    node->data = (char*)malloc(node->data_length + 1);
    strncpy(node->data, data, node->data_length);
    node->data[node->data_length] = '\0';
    
    node->neighbor_count = 0;
    node->fractal_layer = NULL;
    node->last_accessed_timestamp = time(NULL);
    node->access_count_session = 0;
    node->access_time_first = 0;
    node->relevance_to_focus = 0.0f;
    node->is_active = true;
    
    graph->node_count++;
    pthread_mutex_unlock(&graph->graph_lock);
    
    printf("✅ Added node %d (embedding_dim=%d, data_len=%d)\n", node_id, embedding_dim, node->data_length);
    return node_id;
}

/* ===== GET NODE ===== */
HoneycombNode* honeycomb_get_node(HoneycombGraph* graph, int node_id) {
    if (!graph || node_id < 0 || node_id >= graph->node_count) return NULL;
    
    HoneycombNode* node = &graph->nodes[node_id];
    pthread_mutex_lock(&graph->node_locks[node_id]);
    
    node->last_accessed_timestamp = time(NULL);
    node->access_count_session++;
    
    if (node->access_time_first == 0) {
        node->access_time_first = node->last_accessed_timestamp;
    }
    
    pthread_mutex_unlock(&graph->node_locks[node_id]);
    return node;
}

/* ===== ADD EDGE ===== */
bool honeycomb_add_edge(HoneycombGraph* graph, int source_id, int target_id,
                       float relevance_score, const char* relationship_type) {
    if (!graph || source_id < 0 || target_id < 0 || source_id >= graph->node_count || target_id >= graph->node_count) {
        return false;
    }
    
    pthread_mutex_lock(&graph->node_locks[source_id]);
    
    HoneycombNode* source = &graph->nodes[source_id];
    if (source->neighbor_count >= HEXAGONAL_NEIGHBORS) {
        pthread_mutex_unlock(&graph->node_locks[source_id]);
        printf("⚠️  Node %d at max neighbors\n", source_id);
        return false;
    }
    
    HoneycombEdge* edge = &source->neighbors[source->neighbor_count];
    edge->target_id = target_id;
    edge->relevance_score = fmaxf(0.0f, fminf(1.0f, relevance_score));
    strncpy(edge->relationship_type, relationship_type, MAX_RELATIONSHIP_TYPE - 1);
    edge->relationship_type[MAX_RELATIONSHIP_TYPE - 1] = '\0';
    edge->timestamp_created = time(NULL);
    
    source->neighbor_count++;
    pthread_mutex_unlock(&graph->node_locks[source_id]);
    
    printf("✅ Added edge: Node %d → Node %d (relevance=%.2f)\n", source_id, target_id, relevance_score);
    return true;
}

/* ===== FRACTAL INSERTION (CORE INNOVATION) ===== */
void honeycomb_insert_memory(HoneycombGraph* graph, int focus_node_id,
                            int new_node_id, long current_time) {
    if (!graph || focus_node_id < 0 || new_node_id < 0) return;
    
    pthread_mutex_lock(&graph->node_locks[focus_node_id]);
    
    HoneycombNode* focus = &graph->nodes[focus_node_id];
    HoneycombNode* new_mem = &graph->nodes[new_node_id];
    
    // Calculate relevance between focus and new memory
    float relevance = calculate_relevance(
        focus->vector_embedding, new_mem->vector_embedding,
        focus->embedding_dim, new_mem->last_accessed_timestamp, current_time
    );
    
    // If space available, add directly
    if (focus->neighbor_count < HEXAGONAL_NEIGHBORS) {
        honeycomb_add_edge(graph, focus_node_id, new_node_id, relevance, "memory_of");
        printf("✅ Direct insert: Node %d connected to Node %d (rel=%.2f)\n", 
               focus_node_id, new_node_id, relevance);
    } else {
        // Find weakest neighbor
        int weakest_idx = 0;
        float weakest_relevance = focus->neighbors[0].relevance_score;
        
        for (int i = 1; i < HEXAGONAL_NEIGHBORS; i++) {
            if (focus->neighbors[i].relevance_score < weakest_relevance) {
                weakest_relevance = focus->neighbors[i].relevance_score;
                weakest_idx = i;
            }
        }
        
        // If new is stronger, move weakest to fractal layer
        if (relevance > weakest_relevance) {
            int weakest_id = focus->neighbors[weakest_idx].target_id;
            
            // Create fractal layer if needed
            if (!focus->fractal_layer) {
                char fractal_name[128];
                snprintf(fractal_name, 128, "fractal_of_node_%d", focus_node_id);
                focus->fractal_layer = honeycomb_create_graph(fractal_name, MAX_NODES / 10, MAX_SESSION_TIME);
            }
            
            // Move weakest to fractal
            printf("🔀 Moving Node %d to fractal layer of Node %d\n", weakest_id, focus_node_id);
            
            // Remove and replace
            focus->neighbors[weakest_idx].target_id = new_node_id;
            focus->neighbors[weakest_idx].relevance_score = relevance;
            
            printf("✅ Fractal swap: Node %d ↔ Node %d (new rel=%.2f)\n",
                   weakest_id, new_node_id, relevance);
        } else {
            // Insert to fractal directly
            if (!focus->fractal_layer) {
                char fractal_name[128];
                snprintf(fractal_name, 128, "fractal_of_node_%d", focus_node_id);
                focus->fractal_layer = honeycomb_create_graph(fractal_name, MAX_NODES / 10, MAX_SESSION_TIME);
            }
            printf("✅ Inserted Node %d to fractal layer (rel=%.2f)\n", new_node_id, relevance);
        }
    }
    
    pthread_mutex_unlock(&graph->node_locks[focus_node_id]);
}

/* ===== JIT CONTEXT RETRIEVAL ===== */
char* honeycomb_get_jit_context(HoneycombGraph* graph, float* query_vector,
                               int embedding_dim, int max_tokens) {
    if (!graph || !query_vector || max_tokens <= 0) return NULL;
    
    // Find most relevant starting node
    int start_id = honeycomb_find_most_relevant_node(graph, query_vector, embedding_dim);
    if (start_id < 0) return NULL;
    
    // Allocate result buffer
    char* result = (char*)malloc(max_tokens + 1);
    if (!result) return NULL;
    
    result[0] = '\0';
    int current_length = 0;
    
    // BFS traversal with relevance filtering
    bool* visited = (bool*)calloc(graph->node_count, sizeof(bool));
    int* queue = (int*)malloc(graph->node_count * sizeof(int));
    int front = 0, rear = 0;
    
    queue[rear++] = start_id;
    visited[start_id] = true;
    
    while (front < rear && current_length < max_tokens) {
        int node_id = queue[front++];
        HoneycombNode* node = honeycomb_get_node(graph, node_id);
        
        if (!node || !node->is_active) continue;
        
        // Append node data if space available
        int data_len = strlen(node->data);
        if (current_length + data_len + 2 < max_tokens) {
            strcat(result, node->data);
            strcat(result, " ");
            current_length += data_len + 1;
        }
        
        // Queue neighbors with relevance > threshold
        for (int i = 0; i < node->neighbor_count; i++) {
            HoneycombEdge* edge = &node->neighbors[i];
            if (edge->relevance_score > RELEVANCE_THRESHOLD && !visited[edge->target_id]) {
                visited[edge->target_id] = true;
                if (rear < graph->node_count) {
                    queue[rear++] = edge->target_id;
                }
            }
        }
    }
    
    free(visited);
    free(queue);
    
    printf("✅ JIT context retrieved (length=%d tokens)\n", current_length);
    return result;
}

/* ===== SAFETY CIRCUIT BREAKER ===== */
int honeycomb_check_safety(HoneycombNode* node, long current_time,
                          long session_start_time, int max_session_time) {
    if (!node) return SAFETY_INVALID_NODE;
    
    // Check for loops: >3 accesses in <10 second window
    if (node->access_count_session > LOOP_ACCESS_LIMIT) {
        long time_window = node->last_accessed_timestamp - node->access_time_first;
        if (time_window < LOOP_DETECTION_WINDOW && time_window >= 0) {
            printf("⚠️  LOOP DETECTED: Node %d accessed %d times in %ld seconds\n",
                   node->id, node->access_count_session, time_window);
            return SAFETY_LOOP_DETECTED;
        }
    }
    
    // Check session timeout
    long session_elapsed = current_time - session_start_time;
    if (session_elapsed > max_session_time) {
        printf("⚠️  SESSION EXPIRED: %ld seconds elapsed\n", session_elapsed);
        return SAFETY_SESSION_EXPIRED;
    }
    
    return SAFETY_OK;
}

/* ===== FIND MOST RELEVANT NODE ===== */
int honeycomb_find_most_relevant_node(HoneycombGraph* graph, float* query_vector,
                                      int embedding_dim) {
    if (!graph || !query_vector || graph->node_count == 0) return -1;
    
    int best_id = 0;
    float best_relevance = -1.0f;
    
    long current_time = time(NULL);
    
    for (int i = 0; i < graph->node_count; i++) {
        if (!graph->nodes[i].is_active) continue;
        
        float relevance = calculate_relevance(
            query_vector, graph->nodes[i].vector_embedding,
            embedding_dim, graph->nodes[i].last_accessed_timestamp, current_time
        );
        
        if (relevance > best_relevance) {
            best_relevance = relevance;
            best_id = i;
        }
    }
    
    printf("✅ Found most relevant node: %d (relevance=%.2f)\n", best_id, best_relevance);
    return best_id;
}

/* ===== PRINT GRAPH STATS ===== */
void honeycomb_print_graph_stats(HoneycombGraph* graph) {
    if (!graph) return;
    
    printf("\n╔════════════════════════════════════════╗\n");
    printf("║  HONEYCOMB GRAPH STATISTICS              ║\n");
    printf("╚════════════════════════════════════════╝\n");
    printf("Graph Name: %s\n", graph->graph_name);
    printf("Node Count: %d / %d\n", graph->node_count, graph->max_nodes);
    
    int total_edges = 0;
    int total_fractal_layers = 0;
    
    for (int i = 0; i < graph->node_count; i++) {
        total_edges += graph->nodes[i].neighbor_count;
        if (graph->nodes[i].fractal_layer) total_fractal_layers++;
    }
    
    printf("Total Edges: %d\n", total_edges);
    printf("Fractal Layers: %d\n", total_fractal_layers);
    printf("Avg Connectivity: %.2f\n", (float)total_edges / (graph->node_count ? graph->node_count : 1));
    printf("\n");
}

/* ===== RESET SESSION ===== */
void honeycomb_reset_session(HoneycombGraph* graph) {
    if (!graph) return;
    
    pthread_mutex_lock(&graph->graph_lock);
    graph->session_start_time = time(NULL);
    
    for (int i = 0; i < graph->node_count; i++) {
        graph->nodes[i].access_count_session = 0;
        graph->nodes[i].access_time_first = 0;
    }
    pthread_mutex_unlock(&graph->graph_lock);
    printf("✅ Session reset\n");
}

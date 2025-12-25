/**
 * =====================================================================
 * OV-Memory v1.1: Full Implementation
 * =====================================================================
 * Author: Prayaga Vaibhavlakshmi
 * License: Apache License 2.0
 * Om Vinayaka 🙏
 *
 * Module 1: Resource-Aware Metabolism Engine
 * Module 2: Centroid-Based Entry Indexing
 * Module 3: Binary Persistence with Fractal Recursion
 * Module 4: Cross-Session Hydration
 *
 * Compile:
 *   gcc -Wall -Wextra -O3 -march=native -lm -pthread -o ov_memory_v1_1 ov_memory_v1_1.c
 *
 * =====================================================================
 */

#include "ov_memory.h"
#include <dirent.h>
#include <sys/stat.h>

// ===== VECTOR MATH FUNCTIONS =====

static float cosineSimilarity(const float *vecA, const float *vecB, uint32_t dim) {
    if (!vecA || !vecB || dim == 0) return 0.0f;
    
    float dotProduct = 0.0f, magA = 0.0f, magB = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        dotProduct += vecA[i] * vecB[i];
        magA += vecA[i] * vecA[i];
        magB += vecB[i] * vecB[i];
    }
    
    magA = sqrtf(magA);
    magB = sqrtf(magB);
    
    if (magA == 0.0f || magB == 0.0f) return 0.0f;
    return dotProduct / (magA * magB);
}

static float temporalDecay(time_t createdTime, time_t currentTime) {
    if (createdTime > currentTime) return 1.0f;
    float ageSeconds = (float)(currentTime - createdTime);
    float decay = expf(-ageSeconds / TEMPORAL_DECAY_HALF_LIFE);
    return fmaxf(0.0f, fminf(1.0f, decay));
}

// ===== MODULE 1: RESOURCE-AWARE METABOLISM ENGINE =====

void honeycombInitializeMetabolism(HoneycombGraph *graph, int maxMessages, int maxMinutes, bool isApiMode) {
    if (!graph) return;
    
    pthread_mutex_lock(&graph->lock);
    
    graph->metabolism.messages_remaining = maxMessages;
    graph->metabolism.minutes_remaining = maxMinutes * 60;
    graph->metabolism.is_api_mode = isApiMode;
    graph->metabolism.context_availability = 0.0f;
    graph->metabolism.metabolic_weight = 1.0f;
    graph->metabolism.state = METABOLIC_HEALTHY;
    graph->metabolism.audit_last_run = time(NULL);
    
    printf("✅ Initialized Metabolism: messages=%d, minutes=%d, api_mode=%d\n",
           maxMessages, maxMinutes, isApiMode);
    
    pthread_mutex_unlock(&graph->lock);
}

void honeycombUpdateMetabolism(HoneycombGraph *graph, int messagesUsed, int secondsElapsed, float contextUsed) {
    if (!graph) return;
    
    pthread_mutex_lock(&graph->lock);
    
    graph->metabolism.messages_remaining -= messagesUsed;
    graph->metabolism.minutes_remaining -= secondsElapsed;
    graph->metabolism.context_availability = fminf(100.0f, contextUsed);
    
    // Determine metabolic state and weight
    if (graph->metabolism.minutes_remaining < 300 || graph->metabolism.messages_remaining < 5) {
        graph->metabolism.state = METABOLIC_CRITICAL;
        graph->metabolism.metabolic_weight = 1.5f; // Boost relevance of old memories
    } else if (graph->metabolism.minutes_remaining < 1080 || graph->metabolism.messages_remaining < 20) {
        graph->metabolism.state = METABOLIC_STRESSED;
        graph->metabolism.metabolic_weight = 1.2f;
    } else {
        graph->metabolism.state = METABOLIC_HEALTHY;
        graph->metabolism.metabolic_weight = 1.0f;
    }
    
    printf("🔄 Metabolism Updated: state=%d, weight=%.2f, context=%.1f%%\n",
           graph->metabolism.state, graph->metabolism.metabolic_weight, graph->metabolism.context_availability);
    
    pthread_mutex_unlock(&graph->lock);
}

float honeycombCalculateMetabolicRelevance(
    const float *vecA, const float *vecB, uint32_t dim,
    time_t createdTime, time_t currentTime,
    float resourceAvailability, float metabolicWeight
) {
    float semanticScore = cosineSimilarity(vecA, vecB, dim);
    float decayScore = temporalDecay(createdTime, currentTime);
    float resourceScore = 1.0f - (resourceAvailability / 100.0f);
    
    // R_final = (S_sem * 0.6) + (T_decay * 0.2) + (R_resource * 0.2) * weight
    float finalScore = (
        (semanticScore * 0.6f) +
        (decayScore * 0.2f) +
        (resourceScore * 0.2f)
    ) * metabolicWeight;
    
    return fmaxf(0.0f, fminf(1.0f, finalScore));
}

// ===== MODULE 2: CENTROID-BASED ENTRY INDEXING =====

void honeycombInitializeCentroidMap(HoneycombGraph *graph) {
    if (!graph) return;
    
    pthread_mutex_lock(&graph->lock);
    
    uint32_t maxHubs = (uint32_t)fmaxf(1, graph->nodeCount * CENTROID_SCAN_PERCENTAGE);
    maxHubs = fminf(maxHubs, CENTROID_COUNT);
    
    graph->centroidMap.hubNodeIds = (uint32_t *)calloc(maxHubs, sizeof(uint32_t));
    graph->centroidMap.hubCentrality = (float *)calloc(maxHubs, sizeof(float));
    graph->centroidMap.maxHubs = maxHubs;
    graph->centroidMap.hubCount = 0;
    
    printf("✅ Initialized Centroid Map: max_hubs=%u\n", maxHubs);
    
    pthread_mutex_unlock(&graph->lock);
}

void honeycombRecalculateCentrality(HoneycombGraph *graph) {
    if (!graph || graph->nodeCount == 0) return;
    
    pthread_mutex_lock(&graph->lock);
    
    // Calculate centrality (degree + weighted by relevance)
    float *centrality = (float *)calloc(graph->nodeCount, sizeof(float));
    
    for (uint32_t i = 0; i < graph->nodeCount; i++) {
        if (graph->nodes[i] && graph->nodes[i]->isActive) {
            float degree = (float)graph->nodes[i]->neighborCount / HEXAGONAL_NEIGHBORS;
            float weightedRelevance = 0.0f;
            
            for (uint32_t j = 0; j < graph->nodes[i]->neighborCount; j++) {
                weightedRelevance += graph->nodes[i]->neighbors[j].relevanceScore;
            }
            weightedRelevance /= fmaxf(1.0f, (float)graph->nodes[i]->neighborCount);
            
            centrality[i] = (degree * 0.6f) + (weightedRelevance * 0.4f);
            graph->nodes[i]->metabolic_weight = centrality[i];
        }
    }
    
    // Find top hubs
    for (uint32_t i = 0; i < graph->centroidMap.maxHubs; i++) {
        float maxCentrality = -1.0f;
        uint32_t maxIdx = 0;
        
        for (uint32_t j = 0; j < graph->nodeCount; j++) {
            int alreadySelected = 0;
            for (uint32_t k = 0; k < i; k++) {
                if (graph->centroidMap.hubNodeIds[k] == j) {
                    alreadySelected = 1;
                    break;
                }
            }
            
            if (!alreadySelected && centrality[j] > maxCentrality) {
                maxCentrality = centrality[j];
                maxIdx = j;
            }
        }
        
        if (maxCentrality >= 0.0f) {
            graph->centroidMap.hubNodeIds[i] = maxIdx;
            graph->centroidMap.hubCentrality[i] = maxCentrality;
            graph->centroidMap.hubCount++;
        }
    }
    
    printf("✅ Recalculated Centrality: found %u hubs\n", graph->centroidMap.hubCount);
    free(centrality);
    
    pthread_mutex_unlock(&graph->lock);
}

uint32_t honeycombFindMostRelevantNode(HoneycombGraph *graph, const float *queryVector, uint32_t embeddingDim) {
    if (!graph || !queryVector || graph->nodeCount == 0) return UINT32_MAX;
    
    pthread_mutex_lock(&graph->lock);
    
    // Phase 1: Scan centroid map (O(k))
    uint32_t bestHubId = UINT32_MAX;
    float bestHubScore = -1.0f;
    
    for (uint32_t i = 0; i < graph->centroidMap.hubCount; i++) {
        uint32_t hubId = graph->centroidMap.hubNodeIds[i];
        if (hubId >= graph->nodeCount || !graph->nodes[hubId]) continue;
        
        float score = cosineSimilarity(queryVector, graph->nodes[hubId]->vectorEmbedding, embeddingDim);
        if (score > bestHubScore) {
            bestHubScore = score;
            bestHubId = hubId;
        }
    }
    
    if (bestHubId == UINT32_MAX) {
        // Fallback: linear scan if no hubs available
        for (uint32_t i = 0; i < graph->nodeCount; i++) {
            if (graph->nodes[i] && graph->nodes[i]->isActive) {
                bestHubId = i;
                break;
            }
        }
    }
    
    // Phase 2: Local BFS from hub to refine
    uint32_t bestNodeId = bestHubId;
    float bestScore = cosineSimilarity(queryVector, graph->nodes[bestNodeId]->vectorEmbedding, embeddingDim);
    
    // Check immediate neighbors
    for (uint32_t i = 0; i < graph->nodes[bestNodeId]->neighborCount; i++) {
        uint32_t neighborId = graph->nodes[bestNodeId]->neighbors[i].targetId;
        if (neighborId < graph->nodeCount && graph->nodes[neighborId]) {
            float neighborScore = cosineSimilarity(queryVector, graph->nodes[neighborId]->vectorEmbedding, embeddingDim);
            if (neighborScore > bestScore) {
                bestScore = neighborScore;
                bestNodeId = neighborId;
            }
        }
    }
    
    printf("✅ Found entry node: %u (score=%.3f)\n", bestNodeId, bestScore);
    
    pthread_mutex_unlock(&graph->lock);
    return bestNodeId;
}

// ===== MODULE 3: BINARY PERSISTENCE =====

int honeycombSaveBinary(HoneycombGraph *graph, const char *filename) {
    if (!graph || !filename) return -1;
    
    pthread_mutex_lock(&graph->lock);
    
    FILE *file = fopen(filename, "wb");
    if (!file) {
        pthread_mutex_unlock(&graph->lock);
        return -1;
    }
    
    // Header
    fprintf(file, "OM_VINAYAKA");
    fwrite(&graph->nodeCount, sizeof(uint32_t), 1, file);
    fwrite(&graph->maxNodes, sizeof(uint32_t), 1, file);
    fwrite(&graph->metabolism, sizeof(AgentMetabolism), 1, file);
    
    // Nodes
    for (uint32_t i = 0; i < graph->nodeCount; i++) {
        HoneycombNode *node = graph->nodes[i];
        if (!node) continue;
        
        fwrite(&node->id, sizeof(uint32_t), 1, file);
        fwrite(&node->embeddingDim, sizeof(uint32_t), 1, file);
        fwrite(&node->metabolic_weight, sizeof(float), 1, file);
        fwrite(&node->isFractalSeed, sizeof(uint8_t), 1, file);
        fwrite(&node->neighborCount, sizeof(uint32_t), 1, file);
        
        // Embedding
        fwrite(node->vectorEmbedding, sizeof(float), node->embeddingDim, file);
        
        // Data
        uint32_t dataLen = strlen(node->data) + 1;
        fwrite(&dataLen, sizeof(uint32_t), 1, file);
        fwrite(node->data, sizeof(char), dataLen, file);
        
        // Edges
        for (uint32_t j = 0; j < node->neighborCount; j++) {
            fwrite(&node->neighbors[j].targetId, sizeof(uint32_t), 1, file);
            fwrite(&node->neighbors[j].relevanceScore, sizeof(float), 1, file);
            fwrite(node->neighbors[j].relationshipType, sizeof(char), 256, file);
        }
    }
    
    fclose(file);
    printf("✅ Graph saved to %s\n", filename);
    
    pthread_mutex_unlock(&graph->lock);
    return 0;
}

HoneycombGraph *honeycombLoadBinary(const char *filename) {
    if (!filename) return NULL;
    
    FILE *file = fopen(filename, "rb");
    if (!file) return NULL;
    
    // Verify header
    char header[11];
    fread(header, sizeof(char), 11, file);
    if (strncmp(header, "OM_VINAYAKA", 11) != 0) {
        fclose(file);
        return NULL;
    }
    
    uint32_t nodeCount, maxNodes;
    AgentMetabolism metabolism;
    
    fread(&nodeCount, sizeof(uint32_t), 1, file);
    fread(&maxNodes, sizeof(uint32_t), 1, file);
    fread(&metabolism, sizeof(AgentMetabolism), 1, file);
    
    HoneycombGraph *graph = honeycombCreateGraph("loaded_graph", maxNodes, MAX_SESSION_TIME);
    if (!graph) {
        fclose(file);
        return NULL;
    }
    
    graph->metabolism = metabolism;
    
    // Load nodes
    for (uint32_t i = 0; i < nodeCount; i++) {
        uint32_t id, embeddingDim, neighborCount, dataLen;
        float metabolicWeight;
        uint8_t isFractalSeed;
        
        fread(&id, sizeof(uint32_t), 1, file);
        fread(&embeddingDim, sizeof(uint32_t), 1, file);
        fread(&metabolicWeight, sizeof(float), 1, file);
        fread(&isFractalSeed, sizeof(uint8_t), 1, file);
        fread(&neighborCount, sizeof(uint32_t), 1, file);
        
        float *embedding = (float *)malloc(embeddingDim * sizeof(float));
        fread(embedding, sizeof(float), embeddingDim, file);
        
        fread(&dataLen, sizeof(uint32_t), 1, file);
        char *data = (char *)malloc(dataLen);
        fread(data, sizeof(char), dataLen, file);
        
        int32_t nodeId = honeycombAddNode(graph, embedding, embeddingDim, data);
        
        if (nodeId >= 0) {
            graph->nodes[nodeId]->metabolic_weight = metabolicWeight;
            graph->nodes[nodeId]->isFractalSeed = isFractalSeed;
        }
        
        // Load edges
        for (uint32_t j = 0; j < neighborCount; j++) {
            uint32_t targetId;
            float relevance;
            char relType[256];
            
            fread(&targetId, sizeof(uint32_t), 1, file);
            fread(&relevance, sizeof(float), 1, file);
            fread(relType, sizeof(char), 256, file);
            
            if (nodeId >= 0) {
                honeycombAddEdge(graph, nodeId, targetId, relevance, relType);
            }
        }
        
        free(embedding);
        free(data);
    }
    
    fclose(file);
    printf("✅ Graph loaded from %s (nodes=%u)\n", filename, nodeCount);
    return graph;
}

void honeycombExportGraphviz(HoneycombGraph *graph, const char *filename) {
    if (!graph || !filename) return;
    
    FILE *file = fopen(filename, "w");
    if (!file) return;
    
    fprintf(file, "digraph HoneycombGraph {\n");
    fprintf(file, "  rankdir=LR;\n");
    fprintf(file, "  label=\"OV-Memory Fractal Honeycomb\\n(Om Vinayaka)\";\n");
    
    // Nodes with metabolic coloring
    for (uint32_t i = 0; i < graph->nodeCount; i++) {
        HoneycombNode *node = graph->nodes[i];
        if (!node || !node->isActive) continue;
        
        const char *color = "green";
        const char *shape = "circle";
        
        if (node->metabolic_weight < 0.5f) color = "red";
        else if (node->metabolic_weight < 0.8f) color = "orange";
        
        if (node->isFractalSeed) shape = "doubleoctagon";
        
        fprintf(file, "  node_%u [label=\"N%u\", color=%s, shape=%s];\n",
                node->id, node->id, color, shape);
    }
    
    // Edges
    for (uint32_t i = 0; i < graph->nodeCount; i++) {
        HoneycombNode *node = graph->nodes[i];
        if (!node) continue;
        
        for (uint32_t j = 0; j < node->neighborCount; j++) {
            uint32_t targetId = node->neighbors[j].targetId;
            float relevance = node->neighbors[j].relevanceScore;
            
            fprintf(file, "  node_%u -> node_%u [label=\"%.2f\", weight=%.2f];\n",
                    node->id, targetId, relevance, relevance);
        }
    }
    
    fprintf(file, "}\n");
    fclose(file);
    printf("✅ Exported to GraphViz: %s\n", filename);
}

// ===== MODULE 4: CROSS-SESSION HYDRATION =====

uint32_t honeycombCreateFractalSeed(HoneycombGraph *graph, const char *seedLabel) {
    if (!graph || !seedLabel) return UINT32_MAX;
    
    pthread_mutex_lock(&graph->lock);
    
    // Create summary embedding by averaging active nodes
    float *seedEmbedding = (float *)calloc(768, sizeof(float));
    uint32_t activeCount = 0;
    
    for (uint32_t i = 0; i < graph->nodeCount; i++) {
        HoneycombNode *node = graph->nodes[i];
        if (node && node->isActive) {
            for (uint32_t j = 0; j < node->embeddingDim; j++) {
                seedEmbedding[j] += node->vectorEmbedding[j];
            }
            activeCount++;
        }
    }
    
    // Normalize
    if (activeCount > 0) {
        for (uint32_t j = 0; j < 768; j++) {
            seedEmbedding[j] /= activeCount;
        }
    }
    
    // Add seed node
    int32_t seedId = honeycombAddNode(graph, seedEmbedding, 768, seedLabel);
    if (seedId >= 0) {
        graph->nodes[seedId]->isFractalSeed = 1;
        printf("✅ Created Fractal Seed: %u from %u active nodes\n", seedId, activeCount);
    }
    
    free(seedEmbedding);
    pthread_mutex_unlock(&graph->lock);
    
    return (seedId >= 0) ? (uint32_t)seedId : UINT32_MAX;
}

int honeycombHydrateSession(HoneycombGraph *graph, const float *userVector, uint32_t embeddingDim, const char *sessionDir) {
    if (!graph || !userVector || !sessionDir) return -1;
    
    DIR *dir = opendir(sessionDir);
    if (!dir) return -1;
    
    int hydratedCount = 0;
    struct dirent *entry;
    
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_REG && strstr(entry->d_name, ".bin")) {
            char filepath[1024];
            snprintf(filepath, sizeof(filepath), "%s/%s", sessionDir, entry->d_name);
            
            HoneycombGraph *seedGraph = honeycombLoadBinary(filepath);
            if (!seedGraph) continue;
            
            // Check if any seed matches user vector
            for (uint32_t i = 0; i < seedGraph->nodeCount; i++) {
                HoneycombNode *seed = seedGraph->nodes[i];
                if (!seed || !seed->isFractalSeed) continue;
                
                float similarity = cosineSimilarity(userVector, seed->vectorEmbedding, embeddingDim);
                
                if (similarity > 0.85f) {
                    printf("✅ Hydrated from seed (similarity=%.3f)\n", similarity);
                    hydratedCount++;
                }
            }
            
            honeycombFreeGraph(seedGraph);
        }
    }
    
    closedir(dir);
    printf("✅ Cross-Session Hydration: loaded %d seeds\n", hydratedCount);
    return hydratedCount;
}

// ===== METABOLIC AUDIT =====

void honeycombMetabolicAudit(HoneycombGraph *graph) {
    if (!graph) return;
    
    pthread_mutex_lock(&graph->lock);
    
    time_t now = time(NULL);
    int secondsElapsed = (int)(now - graph->sessionStartTime);
    int minutesLeft = (graph->maxSessionTimeSeconds - secondsElapsed) / 60;
    
    // Trigger @ 21 minutes left: Semantic audit
    if (minutesLeft <= 21 && minutesLeft > 18) {
        printf("🔍 SEMANTIC AUDIT (21 min threshold)\n");
        honeycombRecalculateCentrality(graph);
    }
    
    // Trigger @ 18 minutes left: Fractal overflow
    if (minutesLeft <= 18 && minutesLeft > 5) {
        printf("🌀 FRACTAL OVERFLOW (18 min threshold)\n");
        for (uint32_t i = 0; i < graph->nodeCount; i++) {
            HoneycombNode *node = graph->nodes[i];
            if (node && node->metabolic_weight < 0.7f) {
                node->isActive = 0; // Mark for fractal layer
            }
        }
    }
    
    // Trigger @ 5 minutes left: Critical seal
    if (minutesLeft <= 5) {
        printf("🔐 CRITICAL FRACTAL SEAL (5 min threshold)\n");
        
        uint32_t seedId = honeycombCreateFractalSeed(graph, "critical_session_seed");
        if (seedId != UINT32_MAX) {
            time_t timestamp = time(NULL);
            char filename[256];
            snprintf(filename, sizeof(filename), "seed_%ld.bin", timestamp);
            honeycombSaveBinary(graph, filename);
        }
    }
    
    graph->metabolism.audit_last_run = now;
    pthread_mutex_unlock(&graph->lock);
}

void honeycombPrintMetabolicState(HoneycombGraph *graph) {
    if (!graph) return;
    
    pthread_mutex_lock(&graph->lock);
    
    const char *stateStr[] = {"HEALTHY", "STRESSED", "CRITICAL"};
    
    printf("\n=====================================\n");
    printf("METABOLIC STATE REPORT\n");
    printf("=====================================\n");
    printf("State: %s\n", stateStr[graph->metabolism.state]);
    printf("Messages Left: %d\n", graph->metabolism.messages_remaining);
    printf("Time Left: %d sec\n", graph->metabolism.minutes_remaining);
    printf("Context Used: %.1f%%\n", graph->metabolism.context_availability);
    printf("Metabolic Weight: %.2f\n", graph->metabolism.metabolic_weight);
    printf("\n");
    
    pthread_mutex_unlock(&graph->lock);
}

// ===== GRAPH CREATION/DESTRUCTION =====

HoneycombGraph *honeycombCreateGraph(const char *name, uint32_t maxNodes, uint32_t maxSessionTime) {
    if (maxNodes == 0) maxNodes = MAX_NODES;
    if (maxSessionTime == 0) maxSessionTime = MAX_SESSION_TIME;
    
    HoneycombGraph *graph = (HoneycombGraph *)malloc(sizeof(HoneycombGraph));
    if (!graph) return NULL;
    
    strncpy(graph->name, name, sizeof(graph->name) - 1);
    graph->nodes = (HoneycombNode **)calloc(maxNodes, sizeof(HoneycombNode *));
    graph->nodeCount = 0;
    graph->maxNodes = maxNodes;
    graph->sessionStartTime = time(NULL);
    graph->maxSessionTimeSeconds = maxSessionTime;
    
    pthread_mutex_init(&graph->lock, NULL);
    graph->isDirty = 0;
    
    honeycombInitializeMetabolism(graph, 100, maxSessionTime / 60, false);
    honeycombInitializeCentroidMap(graph);
    
    printf("✅ Created honeycomb graph: %s (max_nodes=%u)\n", name, maxNodes);
    return graph;
}

void honeycombFreeGraph(HoneycombGraph *graph) {
    if (!graph) return;
    
    pthread_mutex_lock(&graph->lock);
    
    for (uint32_t i = 0; i < graph->nodeCount; i++) {
        if (graph->nodes[i]) {
            free(graph->nodes[i]->vectorEmbedding);
            free(graph->nodes[i]->data);
            free(graph->nodes[i]->neighbors);
            free(graph->nodes[i]);
        }
    }
    
    free(graph->nodes);
    free(graph->centroidMap.hubNodeIds);
    free(graph->centroidMap.hubCentrality);
    
    pthread_mutex_unlock(&graph->lock);
    pthread_mutex_destroy(&graph->lock);
    
    free(graph);
}

// ===== NODE OPERATIONS =====

int32_t honeycombAddNode(HoneycombGraph *graph, const float *embedding, uint32_t embeddingDim, const char *data) {
    if (!graph || !embedding || !data || embeddingDim == 0) return -1;
    
    pthread_mutex_lock(&graph->lock);
    
    if (graph->nodeCount >= graph->maxNodes) {
        pthread_mutex_unlock(&graph->lock);
        return -1;
    }
    
    uint32_t nodeId = graph->nodeCount;
    HoneycombNode *node = (HoneycombNode *)calloc(1, sizeof(HoneycombNode));
    
    node->id = nodeId;
    node->embeddingDim = embeddingDim;
    node->vectorEmbedding = (float *)malloc(embeddingDim * sizeof(float));
    memcpy(node->vectorEmbedding, embedding, embeddingDim * sizeof(float));
    
    node->data = (char *)malloc(strlen(data) + 1);
    strcpy(node->data, data);
    
    node->neighbors = (HoneycombEdge *)calloc(HEXAGONAL_NEIGHBORS, sizeof(HoneycombEdge));
    node->neighborCount = 0;
    node->lastAccessedTimestamp = time(NULL);
    node->metabolic_weight = 1.0f;
    node->isActive = 1;
    node->isFractalSeed = 0;
    
    graph->nodes[nodeId] = node;
    graph->nodeCount++;
    graph->isDirty = 1;
    
    pthread_mutex_unlock(&graph->lock);
    
    printf("✅ Added node %u\n", nodeId);
    return nodeId;
}

HoneycombNode *honeycombGetNode(HoneycombGraph *graph, uint32_t nodeId) {
    if (!graph || nodeId >= graph->nodeCount) return NULL;
    
    pthread_mutex_lock(&graph->lock);
    HoneycombNode *node = graph->nodes[nodeId];
    if (node) {
        node->lastAccessedTimestamp = time(NULL);
        node->accessCountSession++;
        if (node->accessTimeFirst == 0) node->accessTimeFirst = node->lastAccessedTimestamp;
    }
    pthread_mutex_unlock(&graph->lock);
    
    return node;
}

int honeycombAddEdge(HoneycombGraph *graph, uint32_t sourceId, uint32_t targetId, float relevanceScore, const char *relationshipType) {
    if (!graph || sourceId >= graph->nodeCount || targetId >= graph->nodeCount) return 0;
    
    pthread_mutex_lock(&graph->lock);
    
    HoneycombNode *source = graph->nodes[sourceId];
    if (!source || source->neighborCount >= HEXAGONAL_NEIGHBORS) {
        pthread_mutex_unlock(&graph->lock);
        return 0;
    }
    
    HoneycombEdge *edge = &source->neighbors[source->neighborCount];
    edge->targetId = targetId;
    edge->relevanceScore = fmaxf(0.0f, fminf(1.0f, relevanceScore));
    strncpy(edge->relationshipType, relationshipType, sizeof(edge->relationshipType) - 1);
    edge->timestampCreated = time(NULL);
    
    source->neighborCount++;
    graph->isDirty = 1;
    
    pthread_mutex_unlock(&graph->lock);
    
    return 1;
}

void honeycombInsertMemory(HoneycombGraph *graph, uint32_t focusNodeId, uint32_t newNodeId, time_t currentTime) {
    if (!graph || !currentTime) currentTime = time(NULL);
    if (focusNodeId >= graph->nodeCount || newNodeId >= graph->nodeCount) return;
    
    pthread_mutex_lock(&graph->lock);
    
    HoneycombNode *focus = graph->nodes[focusNodeId];
    HoneycombNode *newMem = graph->nodes[newNodeId];
    
    if (!focus || !newMem) {
        pthread_mutex_unlock(&graph->lock);
        return;
    }
    
    // Use metabolic relevance
    float relevance = honeycombCalculateMetabolicRelevance(
        focus->vectorEmbedding, newMem->vectorEmbedding, focus->embeddingDim,
        newMem->lastAccessedTimestamp, currentTime,
        graph->metabolism.context_availability,
        graph->metabolism.metabolic_weight
    );
    
    if (focus->neighborCount < HEXAGONAL_NEIGHBORS) {
        honeycombAddEdge(graph, focusNodeId, newNodeId, relevance, "memory_of");
    } else {
        uint32_t weakestIdx = 0;
        float weakestRelevance = focus->neighbors[0].relevanceScore;
        
        for (uint32_t i = 1; i < focus->neighborCount; i++) {
            if (focus->neighbors[i].relevanceScore < weakestRelevance) {
                weakestRelevance = focus->neighbors[i].relevanceScore;
                weakestIdx = i;
            }
        }
        
        if (relevance > weakestRelevance) {
            focus->neighbors[weakestIdx].targetId = newNodeId;
            focus->neighbors[weakestIdx].relevanceScore = relevance;
        }
    }
    
    pthread_mutex_unlock(&graph->lock);
}

int honeycombCheckSafety(HoneycombNode *node, time_t currentTime, time_t sessionStartTime, uint32_t maxSessionTime) {
    if (!node) return SAFETY_INVALID_NODE;
    
    if (node->accessCountSession > LOOP_ACCESS_LIMIT) {
        time_t timeWindow = node->lastAccessedTimestamp - node->accessTimeFirst;
        if (timeWindow >= 0 && timeWindow < LOOP_DETECTION_WINDOW) {
            return SAFETY_LOOP_DETECTED;
        }
    }
    
    if (!currentTime) currentTime = time(NULL);
    if (!sessionStartTime) sessionStartTime = time(NULL);
    
    time_t sessionElapsed = currentTime - sessionStartTime;
    if (sessionElapsed > (time_t)maxSessionTime) {
        return SAFETY_SESSION_EXPIRED;
    }
    
    return SAFETY_OK;
}

void honeycombPrintGraphStats(HoneycombGraph *graph) {
    if (!graph) return;
    
    pthread_mutex_lock(&graph->lock);
    
    uint32_t totalEdges = 0;
    for (uint32_t i = 0; i < graph->nodeCount; i++) {
        if (graph->nodes[i]) totalEdges += graph->nodes[i]->neighborCount;
    }
    
    printf("\n=====================================\n");
    printf("GRAPH STATISTICS\n");
    printf("=====================================\n");
    printf("Graph Name: %s\n", graph->name);
    printf("Node Count: %u / %u\n", graph->nodeCount, graph->maxNodes);
    printf("Total Edges: %u\n", totalEdges);
    printf("Centroid Hubs: %u\n", graph->centroidMap.hubCount);
    printf("\n");
    
    pthread_mutex_unlock(&graph->lock);
}

// ===== MAIN TEST =====

int main() {
    printf("\n🧠 OV-Memory v1.1 - Metabolic & Centroid Upgrade\n");
    printf("Om Vinayaka 🙏\n\n");
    
    // Create graph
    HoneycombGraph *graph = honeycombCreateGraph("metabolic_test", 100, 3600);
    
    // Initialize metabolism: 100 messages, 60 minutes, API mode
    honeycombInitializeMetabolism(graph, 100, 60, true);
    
    // Create test data
    float *emb1 = (float *)malloc(768 * sizeof(float));
    float *emb2 = (float *)malloc(768 * sizeof(float));
    float *emb3 = (float *)malloc(768 * sizeof(float));
    
    for (int i = 0; i < 768; i++) {
        emb1[i] = 0.5f;
        emb2[i] = 0.6f;
        emb3[i] = 0.7f;
    }
    
    // Add nodes
    int32_t node1 = honeycombAddNode(graph, emb1, 768, "Memory Alpha");
    int32_t node2 = honeycombAddNode(graph, emb2, 768, "Memory Beta");
    int32_t node3 = honeycombAddNode(graph, emb3, 768, "Memory Gamma");
    
    // Add edges
    honeycombAddEdge(graph, node1, node2, 0.9f, "related_to");
    honeycombAddEdge(graph, node2, node3, 0.85f, "context_of");
    
    // Initialize centroids
    honeycombInitializeCentroidMap(graph);
    honeycombRecalculateCentrality(graph);
    
    // Test metabolic updates
    honeycombUpdateMetabolism(graph, 10, 120, 45.0f);
    honeycombPrintMetabolicState(graph);
    
    // Test entry finding
    uint32_t entryNode = honeycombFindMostRelevantNode(graph, emb1, 768);
    printf("Entry node: %u\n", entryNode);
    
    // Save
    honeycombSaveBinary(graph, "test_graph.bin");
    
    // Export to GraphViz
    honeycombExportGraphviz(graph, "test_graph.dot");
    
    // Create fractal seed
    uint32_t seedId = honeycombCreateFractalSeed(graph, "session_seed");
    printf("Created seed: %u\n", seedId);
    
    // Print stats
    honeycombPrintGraphStats(graph);
    
    printf("✅ v1.1 tests completed\n");
    printf("Om Vinayaka 🙏\n\n");
    
    // Cleanup
    honeycombFreeGraph(graph);
    free(emb1);
    free(emb2);
    free(emb3);
    
    return 0;
}

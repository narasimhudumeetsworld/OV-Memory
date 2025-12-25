/**
 * =====================================================================
 * OV-Memory v1.1: JavaScript / TypeScript Implementation
 * =====================================================================
 * Author: Prayaga Vaibhavlakshmi
 * License: Apache License 2.0
 * Om Vinayaka 🙏
 *
 * Full v1.1 with Metabolism, Centroid Indexing, Binary Persistence, Hydration
 *
 * Usage:
 *   npm install ov-memory
 *   const OV = require('ov_memory_v1_1');
 *
 * =====================================================================
 */

// ===== CONFIGURATION =====
const CONFIG = {
    MAX_NODES: 100_000,
    MAX_EMBEDDING_DIM: 768,
    HEXAGONAL_NEIGHBORS: 6,
    RELEVANCE_THRESHOLD: 0.8,
    MAX_SESSION_TIME: 3600,
    LOOP_DETECTION_WINDOW: 10,
    LOOP_ACCESS_LIMIT: 3,
    TEMPORAL_DECAY_HALF_LIFE: 86400.0,
    CENTROID_COUNT: 5,
    CENTROID_SCAN_PERCENTAGE: 0.05,
    AUDIT_SEMANTIC_TRIGGER: 1260,  // 21 min
    AUDIT_FRACTAL_TRIGGER: 1080,   // 18 min
    AUDIT_CRITICAL_SEAL_TRIGGER: 300  // 5 min
};

// ===== ENUMS =====
const MetabolicState = {
    HEALTHY: 0,
    STRESSED: 1,
    CRITICAL: 2
};

const SafetyCode = {
    OK: 0,
    LOOP_DETECTED: 1,
    SESSION_EXPIRED: 2,
    INVALID_NODE: -1
};

// ===== DATA STRUCTURES =====

class AgentMetabolism {
    constructor(maxMessages = 100, maxMinutes = 60, isApiMode = false) {
        this.messages_remaining = maxMessages;
        this.minutes_remaining = maxMinutes * 60;
        this.is_api_mode = isApiMode;
        this.context_availability = 0.0;
        this.metabolic_weight = 1.0;
        this.state = MetabolicState.HEALTHY;
        this.audit_last_run = Date.now() / 1000;
    }
}

class HoneycombEdge {
    constructor(targetId, relevanceScore, relationshipType) {
        this.target_id = targetId;
        this.relevance_score = Math.max(0.0, Math.min(1.0, relevanceScore));
        this.relationship_type = relationshipType;
        this.timestamp_created = Date.now() / 1000;
    }
}

class HoneycombNode {
    constructor(id, vectorEmbedding, data) {
        this.id = id;
        this.vector_embedding = new Float32Array(vectorEmbedding);
        this.data = data;
        this.neighbors = [];
        this.fractal_layer = null;
        this.last_accessed_timestamp = Date.now() / 1000;
        this.access_count_session = 0;
        this.access_time_first = 0.0;
        this.relevance_to_focus = 0.0;
        this.metabolic_weight = 1.0;
        this.is_active = true;
        this.is_fractal_seed = false;
    }
}

class CentroidMap {
    constructor(maxHubs = CONFIG.CENTROID_COUNT) {
        this.hub_node_ids = [];
        this.hub_centrality = [];
        this.max_hubs = maxHubs;
    }
}

class HoneycombGraph {
    constructor(name, maxNodes = CONFIG.MAX_NODES, maxSessionTime = CONFIG.MAX_SESSION_TIME) {
        this.name = name;
        this.nodes = new Map();
        this.node_count = 0;
        this.max_nodes = maxNodes;
        this.session_start_time = Date.now() / 1000;
        this.max_session_time_seconds = maxSessionTime;
        
        this.metabolism = new AgentMetabolism(100, maxSessionTime / 60, false);
        this.centroid_map = new CentroidMap();
        this.is_dirty = false;
    }
}

// ===== VECTOR MATH =====

function cosineSimilarity(vecA, vecB) {
    if (!vecA || !vecB || vecA.length === 0) return 0.0;
    
    let dotProduct = 0.0;
    let magA = 0.0;
    let magB = 0.0;
    
    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
        magA += vecA[i] * vecA[i];
        magB += vecB[i] * vecB[i];
    }
    
    magA = Math.sqrt(magA);
    magB = Math.sqrt(magB);
    
    if (magA === 0.0 || magB === 0.0) return 0.0;
    return dotProduct / (magA * magB);
}

function temporalDecay(createdTime, currentTime) {
    if (createdTime > currentTime) return 1.0;
    
    const ageSeconds = currentTime - createdTime;
    const decay = Math.exp(-ageSeconds / CONFIG.TEMPORAL_DECAY_HALF_LIFE);
    return Math.max(0.0, Math.min(1.0, decay));
}

// ===== MODULE 1: METABOLISM ENGINE =====

function initializeMetabolism(graph, maxMessages, maxMinutes, isApiMode) {
    graph.metabolism = new AgentMetabolism(maxMessages, maxMinutes, isApiMode);
    console.log(`✅ Initialized Metabolism: messages=${maxMessages}, minutes=${maxMinutes}, api_mode=${isApiMode}`);
}

function updateMetabolism(graph, messagesUsed, secondsElapsed, contextUsed) {
    graph.metabolism.messages_remaining -= messagesUsed;
    graph.metabolism.minutes_remaining -= secondsElapsed;
    graph.metabolism.context_availability = Math.min(100.0, contextUsed);
    
    if (graph.metabolism.minutes_remaining < 300 || graph.metabolism.messages_remaining < 5) {
        graph.metabolism.state = MetabolicState.CRITICAL;
        graph.metabolism.metabolic_weight = 1.5;
    } else if (graph.metabolism.minutes_remaining < 1080 || graph.metabolism.messages_remaining < 20) {
        graph.metabolism.state = MetabolicState.STRESSED;
        graph.metabolism.metabolic_weight = 1.2;
    } else {
        graph.metabolism.state = MetabolicState.HEALTHY;
        graph.metabolism.metabolic_weight = 1.0;
    }
    
    console.log(`🔄 Metabolism Updated: state=${Object.keys(MetabolicState)[graph.metabolism.state]}, ` +
                `weight=${graph.metabolism.metabolic_weight.toFixed(2)}, ` +
                `context=${graph.metabolism.context_availability.toFixed(1)}%`);
}

function calculateMetabolicRelevance(vecA, vecB, createdTime, currentTime, resourceAvail, metabolicWeight) {
    const semantic = cosineSimilarity(vecA, vecB);
    const decay = temporalDecay(createdTime, currentTime);
    const resource = 1.0 - (resourceAvail / 100.0);
    
    const final = (
        (semantic * 0.6) +
        (decay * 0.2) +
        (resource * 0.2)
    ) * metabolicWeight;
    
    return Math.max(0.0, Math.min(1.0, final));
}

// ===== MODULE 2: CENTROID INDEXING =====

function initializeCentroidMap(graph) {
    const maxHubs = Math.max(1, Math.floor(graph.node_count * CONFIG.CENTROID_SCAN_PERCENTAGE));
    graph.centroid_map.max_hubs = Math.min(maxHubs, CONFIG.CENTROID_COUNT);
    console.log(`✅ Initialized Centroid Map: max_hubs=${graph.centroid_map.max_hubs}`);
}

function recalculateCentrality(graph) {
    if (graph.node_count === 0) return;
    
    const centrality = new Map();
    
    // Calculate centrality
    for (const [nodeId, node] of graph.nodes) {
        if (node && node.is_active) {
            const degree = node.neighbors.length / CONFIG.HEXAGONAL_NEIGHBORS;
            const avgRelevance = node.neighbors.length > 0
                ? node.neighbors.reduce((s, e) => s + e.relevance_score, 0) / node.neighbors.length
                : 0.0;
            
            const score = (degree * 0.6) + (avgRelevance * 0.4);
            centrality.set(nodeId, score);
            node.metabolic_weight = score;
        }
    }
    
    // Find top hubs
    const sorted = Array.from(centrality.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, graph.centroid_map.max_hubs);
    
    graph.centroid_map.hub_node_ids = sorted.map(x => x[0]);
    graph.centroid_map.hub_centrality = sorted.map(x => x[1]);
    
    console.log(`✅ Recalculated Centrality: found ${graph.centroid_map.hub_node_ids.length} hubs`);
}

function findMostRelevantNode(graph, queryVector) {
    if (graph.node_count === 0 || !queryVector) return null;
    
    const query = new Float32Array(queryVector);
    
    // Phase 1: Scan hubs
    let bestHubId = null;
    let bestHubScore = -1.0;
    
    for (const hubId of graph.centroid_map.hub_node_ids) {
        const node = graph.nodes.get(hubId);
        if (node) {
            const score = cosineSimilarity(query, node.vector_embedding);
            if (score > bestHubScore) {
                bestHubScore = score;
                bestHubId = hubId;
            }
        }
    }
    
    if (bestHubId === null) {
        for (const [nodeId, node] of graph.nodes) {
            if (node && node.is_active) {
                bestHubId = nodeId;
                break;
            }
        }
    }
    
    if (bestHubId === null) return null;
    
    // Phase 2: Refine with neighbors
    let bestNodeId = bestHubId;
    let bestScore = cosineSimilarity(query, graph.nodes.get(bestHubId).vector_embedding);
    
    for (const edge of graph.nodes.get(bestHubId).neighbors) {
        const neighbor = graph.nodes.get(edge.target_id);
        if (neighbor) {
            const score = cosineSimilarity(query, neighbor.vector_embedding);
            if (score > bestScore) {
                bestScore = score;
                bestNodeId = edge.target_id;
            }
        }
    }
    
    console.log(`✅ Found entry node: ${bestNodeId} (score=${bestScore.toFixed(3)})`);
    return bestNodeId;
}

// ===== MODULE 3: PERSISTENCE =====

function saveBinary(graph, filename) {
    const data = {
        header: 'OM_VINAYAKA',
        node_count: graph.node_count,
        max_nodes: graph.max_nodes,
        metabolism: graph.metabolism,
        nodes: {}
    };
    
    for (const [nodeId, node] of graph.nodes) {
        data.nodes[nodeId] = {
            id: node.id,
            embedding: Array.from(node.vector_embedding),
            data: node.data,
            metabolic_weight: node.metabolic_weight,
            is_fractal_seed: node.is_fractal_seed,
            neighbors: node.neighbors.map(e => ({
                target_id: e.target_id,
                relevance_score: e.relevance_score,
                relationship_type: e.relationship_type
            }))
        };
    }
    
    const fs = require('fs');
    fs.writeFileSync(filename, JSON.stringify(data, null, 2));
    console.log(`✅ Graph saved to ${filename}`);
    return 0;
}

function loadBinary(filename) {
    try {
        const fs = require('fs');
        const data = JSON.parse(fs.readFileSync(filename, 'utf8'));
        
        if (data.header !== 'OM_VINAYAKA') return null;
        
        const graph = createGraph("loaded_graph", data.max_nodes, CONFIG.MAX_SESSION_TIME);
        graph.metabolism = data.metabolism;
        
        // Restore nodes
        for (const [nodeId, nodeData] of Object.entries(data.nodes)) {
            const embedding = new Float32Array(nodeData.embedding);
            addNode(graph, embedding, nodeData.data);
            
            if (graph.nodes.has(parseInt(nodeId))) {
                graph.nodes.get(parseInt(nodeId)).metabolic_weight = nodeData.metabolic_weight;
                graph.nodes.get(parseInt(nodeId)).is_fractal_seed = nodeData.is_fractal_seed;
            }
        }
        
        // Restore edges
        for (const [nodeId, nodeData] of Object.entries(data.nodes)) {
            const nid = parseInt(nodeId);
            if (graph.nodes.has(nid)) {
                for (const edgeData of nodeData.neighbors) {
                    addEdge(graph, nid, edgeData.target_id, edgeData.relevance_score, edgeData.relationship_type);
                }
            }
        }
        
        console.log(`✅ Graph loaded from ${filename} (nodes=${graph.node_count})`);
        return graph;
    } catch (e) {
        console.error(`❌ Load failed: ${e}`);
        return null;
    }
}

function exportGraphviz(graph, filename) {
    let content = 'digraph HoneycombGraph {\n';
    content += '  rankdir=LR;\n';
    content += '  label="OV-Memory Fractal Honeycomb\\n(Om Vinayaka)";\n';
    
    // Nodes
    for (const [nodeId, node] of graph.nodes) {
        if (!node.is_active) continue;
        
        let color = 'green';
        if (node.metabolic_weight < 0.5) color = 'red';
        else if (node.metabolic_weight < 0.8) color = 'orange';
        
        const shape = node.is_fractal_seed ? 'doubleoctagon' : 'circle';
        content += `  node_${nodeId} [label="N${nodeId}", color=${color}, shape=${shape}];\n`;
    }
    
    // Edges
    for (const [nodeId, node] of graph.nodes) {
        for (const edge of node.neighbors) {
            content += `  node_${nodeId} -> node_${edge.target_id} [label="${edge.relevance_score.toFixed(2)}", weight=${edge.relevance_score.toFixed(2)}];\n`;
        }
    }
    
    content += '}\n';
    
    const fs = require('fs');
    fs.writeFileSync(filename, content);
    console.log(`✅ Exported to GraphViz: ${filename}`);
}

// ===== MODULE 4: HYDRATION =====

function createFractalSeed(graph, seedLabel) {
    const activeNodes = Array.from(graph.nodes.values()).filter(n => n.is_active);
    if (activeNodes.length === 0) return null;
    
    // Average embeddings
    const seedEmbedding = new Float32Array(CONFIG.MAX_EMBEDDING_DIM);
    for (const node of activeNodes) {
        for (let i = 0; i < node.vector_embedding.length; i++) {
            seedEmbedding[i] += node.vector_embedding[i];
        }
    }
    for (let i = 0; i < seedEmbedding.length; i++) {
        seedEmbedding[i] /= activeNodes.length;
    }
    
    const seedId = addNode(graph, seedEmbedding, seedLabel);
    if (seedId !== null) {
        graph.nodes.get(seedId).is_fractal_seed = true;
        console.log(`✅ Created Fractal Seed: ${seedId} from ${activeNodes.length} nodes`);
    }
    
    return seedId;
}

function hydrateSession(graph, userVector, sessionDir) {
    let hydratedCount = 0;
    const userArray = new Float32Array(userVector);
    
    try {
        const fs = require('fs');
        const files = fs.readdirSync(sessionDir);
        
        for (const filename of files) {
            if (filename.endsWith('.json')) {
                const filepath = `${sessionDir}/${filename}`;
                const seedGraph = loadBinary(filepath);
                if (seedGraph) {
                    for (const seedNode of seedGraph.nodes.values()) {
                        if (seedNode.is_fractal_seed) {
                            const sim = cosineSimilarity(userArray, seedNode.vector_embedding);
                            if (sim > 0.85) {
                                console.log(`✅ Hydrated from seed (similarity=${sim.toFixed(3)})`);
                                hydratedCount++;
                            }
                        }
                    }
                }
            }
        }
    } catch (e) {
        console.error(`Hydration scan failed: ${e}`);
    }
    
    console.log(`✅ Cross-Session Hydration: loaded ${hydratedCount} seeds`);
    return hydratedCount;
}

// ===== METABOLIC AUDIT =====

function metabolicAudit(graph) {
    const now = Date.now() / 1000;
    const secondsElapsed = Math.floor(now - graph.session_start_time);
    const minutesLeft = Math.floor((graph.max_session_time_seconds - secondsElapsed) / 60);
    
    if (minutesLeft <= 21 && minutesLeft > 18) {
        console.log('🔍 SEMANTIC AUDIT (21 min threshold)');
        recalculateCentrality(graph);
    }
    
    if (minutesLeft <= 18 && minutesLeft > 5) {
        console.log('🌀 FRACTAL OVERFLOW (18 min threshold)');
        for (const node of graph.nodes.values()) {
            if (node.metabolic_weight < 0.7) {
                node.is_active = false;
            }
        }
    }
    
    if (minutesLeft <= 5) {
        console.log('🔐 CRITICAL FRACTAL SEAL (5 min threshold)');
        const seedId = createFractalSeed(graph, 'critical_session_seed');
        if (seedId !== null) {
            const timestamp = Math.floor(Date.now() / 1000);
            saveBinary(graph, `seed_${timestamp}.json`);
        }
    }
}

function printMetabolicState(graph) {
    const stateNames = ['HEALTHY', 'STRESSED', 'CRITICAL'];
    console.log('\n' + '='.repeat(40));
    console.log('METABOLIC STATE REPORT');
    console.log('='.repeat(40));
    console.log(`State: ${stateNames[graph.metabolism.state]}`);
    console.log(`Messages Left: ${graph.metabolism.messages_remaining}`);
    console.log(`Time Left: ${graph.metabolism.minutes_remaining} sec`);
    console.log(`Context Used: ${graph.metabolism.context_availability.toFixed(1)}%`);
    console.log(`Metabolic Weight: ${graph.metabolism.metabolic_weight.toFixed(2)}\n`);
}

// ===== GRAPH OPERATIONS =====

function createGraph(name, maxNodes = CONFIG.MAX_NODES, maxSessionTime = CONFIG.MAX_SESSION_TIME) {
    const graph = new HoneycombGraph(name, maxNodes, maxSessionTime);
    initializeMetabolism(graph, 100, maxSessionTime / 60, false);
    initializeCentroidMap(graph);
    console.log(`✅ Created honeycomb graph: ${name} (max_nodes=${maxNodes})`);
    return graph;
}

function addNode(graph, embedding, data) {
    if (graph.node_count >= graph.max_nodes) return null;
    
    const nodeId = graph.node_count;
    const node = new HoneycombNode(nodeId, embedding, data.substring(0, 8192));
    
    graph.nodes.set(nodeId, node);
    graph.node_count++;
    graph.is_dirty = true;
    
    console.log(`✅ Added node ${nodeId} (embedding_dim=${embedding.length})`);
    return nodeId;
}

function addEdge(graph, sourceId, targetId, relevanceScore, relationshipType) {
    if (!graph.nodes.has(sourceId) || !graph.nodes.has(targetId)) return false;
    
    const source = graph.nodes.get(sourceId);
    if (source.neighbors.length >= CONFIG.HEXAGONAL_NEIGHBORS) return false;
    
    const edge = new HoneycombEdge(targetId, relevanceScore, relationshipType);
    source.neighbors.push(edge);
    graph.is_dirty = true;
    
    console.log(`✅ Added edge: ${sourceId} → ${targetId} (relevance=${relevanceScore.toFixed(2)})`);
    return true;
}

function printGraphStats(graph) {
    const totalEdges = Array.from(graph.nodes.values())
        .reduce((sum, node) => sum + node.neighbors.length, 0);
    
    console.log('\n' + '='.repeat(40));
    console.log('GRAPH STATISTICS');
    console.log('='.repeat(40));
    console.log(`Graph Name: ${graph.name}`);
    console.log(`Node Count: ${graph.node_count} / ${graph.max_nodes}`);
    console.log(`Total Edges: ${totalEdges}`);
    console.log(`Centroid Hubs: ${graph.centroid_map.hub_node_ids.length}\n`);
}

// ===== MAIN TEST =====

async function main() {
    console.log('\n🧠 OV-Memory v1.1 - JavaScript Implementation');
    console.log('Om Vinayaka 🙏\n');
    
    const graph = createGraph('metabolic_test', 100, 3600);
    
    const emb1 = new Float32Array(768).fill(0.5);
    const emb2 = new Float32Array(768).fill(0.6);
    const emb3 = new Float32Array(768).fill(0.7);
    
    const node1 = addNode(graph, emb1, 'Memory Alpha');
    const node2 = addNode(graph, emb2, 'Memory Beta');
    const node3 = addNode(graph, emb3, 'Memory Gamma');
    
    addEdge(graph, node1, node2, 0.9, 'related_to');
    addEdge(graph, node2, node3, 0.85, 'context_of');
    
    recalculateCentrality(graph);
    updateMetabolism(graph, 10, 120, 45.0);
    printMetabolicState(graph);
    
    const entryNode = findMostRelevantNode(graph, emb1);
    console.log(`Entry node: ${entryNode}\n`);
    
    saveBinary(graph, 'test_graph.json');
    exportGraphviz(graph, 'test_graph.dot');
    
    const seedId = createFractalSeed(graph, 'session_seed');
    printGraphStats(graph);
    
    console.log('✅ v1.1 tests completed');
    console.log('Om Vinayaka 🙏\n');
}

// Export for use as module
module.exports = {
    createGraph,
    addNode,
    addEdge,
    initializeMetabolism,
    updateMetabolism,
    calculateMetabolicRelevance,
    initializeCentroidMap,
    recalculateCentrality,
    findMostRelevantNode,
    saveBinary,
    loadBinary,
    exportGraphviz,
    createFractalSeed,
    hydrateSession,
    metabolicAudit,
    printMetabolicState,
    printGraphStats,
    // Classes
    HoneycombGraph,
    HoneycombNode,
    HoneycombEdge,
    AgentMetabolism,
    CentroidMap
};

if (require.main === module) {
    main().catch(console.error);
}

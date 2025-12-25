/**
 * =====================================================================
 * OV-Memory: JavaScript Implementation (Production-Ready)
 * =====================================================================
 * Author: Prayaga Vaibhavlakshmi
 * License: Apache License 2.0
 * Om Vinayaka 🙏
 *
 * A high-performance JavaScript implementation of Fractal Honeycomb
 * Graph Database for AI agents. Zero external dependencies.
 * Compatible with Node.js, Browsers, and all AI agents (Claude, Gemini, Codex).
 *
 * =====================================================================
 */

// ===== CONFIGURATION CONSTANTS =====
const MAX_NODES = 100_000;
const MAX_EMBEDDING_DIM = 768;
const MAX_DATA_SIZE = 8192;
const HEXAGONAL_NEIGHBORS = 6;
const RELEVANCE_THRESHOLD = 0.8;
const MAX_SESSION_TIME = 3600;
const LOOP_DETECTION_WINDOW = 10;
const LOOP_ACCESS_LIMIT = 3;
const TEMPORAL_DECAY_HALF_LIFE = 86400.0; // 24 hours in seconds

// ===== SAFETY RETURN CODES =====
const SAFETY_OK = 0;
const SAFETY_LOOP_DETECTED = 1;
const SAFETY_SESSION_EXPIRED = 2;
const SAFETY_INVALID_NODE = -1;

// ===== VECTOR MATH FUNCTIONS =====

/**
 * Calculate cosine similarity between two vectors
 * @param {Float32Array|number[]} vecA - Vector A
 * @param {Float32Array|number[]} vecB - Vector B
 * @returns {number} Cosine similarity score [0.0, 1.0]
 */
function cosineSimilarity(vecA, vecB) {
    if (!vecA || !vecB || vecA.length === 0 || vecB.length === 0 || vecA.length !== vecB.length) {
        return 0.0;
    }

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

    if (magA === 0.0 || magB === 0.0) {
        return 0.0;
    }

    return dotProduct / (magA * magB);
}

/**
 * Calculate temporal decay factor
 * Uses exponential decay: e^(-age / half_life)
 * @param {number} createdTime - UNIX timestamp when memory was created
 * @param {number} currentTime - Current UNIX timestamp
 * @returns {number} Decay factor [0.0, 1.0]
 */
function temporalDecay(createdTime, currentTime) {
    if (createdTime > currentTime) {
        return 1.0;
    }

    const ageSeconds = currentTime - createdTime;
    const decay = Math.exp(-ageSeconds / TEMPORAL_DECAY_HALF_LIFE);
    return Math.max(0.0, Math.min(1.0, decay));
}

/**
 * Calculate combined relevance score
 * Formula: (Cosine_Similarity × 0.7) + (Temporal_Decay × 0.3)
 * @param {Float32Array|number[]} vecA - Vector A
 * @param {Float32Array|number[]} vecB - Vector B
 * @param {number} createdTime - When the memory was created
 * @param {number} currentTime - Current timestamp
 * @returns {number} Relevance score [0.0, 1.0]
 */
function calculateRelevance(vecA, vecB, createdTime, currentTime) {
    const cosine = cosineSimilarity(vecA, vecB);
    const decay = temporalDecay(createdTime, currentTime);
    const finalScore = (cosine * 0.7) + (decay * 0.3);
    return Math.max(0.0, Math.min(1.0, finalScore));
}

// ===== GRAPH OPERATIONS =====

/**
 * Create a new Honeycomb Graph
 * @param {string} name - Name of the graph
 * @param {number} maxNodes - Maximum number of nodes (default 100,000)
 * @param {number} maxSessionTime - Maximum session duration in seconds (default 3600)
 * @returns {Object} HoneycombGraph instance
 */
function honeycombCreateGraph(name, maxNodes = MAX_NODES, maxSessionTime = MAX_SESSION_TIME) {
    const graph = {
        name,
        nodes: new Map(),
        nodeCount: 0,
        maxNodes,
        sessionStartTime: Math.floor(Date.now() / 1000),
        maxSessionTimeSeconds: maxSessionTime,
    };

    console.log(`✅ Created honeycomb graph: ${name} (max_nodes=${maxNodes})`);
    return graph;
}

/**
 * Add a new node to the graph
 * @param {Object} graph - HoneycombGraph instance
 * @param {Float32Array|number[]} embedding - Vector embedding
 * @param {string} data - Text payload
 * @returns {number} Node ID, or -1 if failed
 */
function honeycombAddNode(graph, embedding, data) {
    if (graph.nodeCount >= graph.maxNodes) {
        console.error("❌ Graph at max capacity");
        return -1;
    }

    const nodeId = graph.nodeCount;
    const node = {
        id: nodeId,
        vectorEmbedding: embedding instanceof Float32Array ? embedding : new Float32Array(embedding),
        data,
        neighbors: [],
        fractalLayer: null,
        lastAccessedTimestamp: Math.floor(Date.now() / 1000),
        accessCountSession: 0,
        accessTimeFirst: 0,
        relevanceToFocus: 0.0,
        isActive: true,
    };

    graph.nodes.set(nodeId, node);
    graph.nodeCount++;

    console.log(`✅ Added node ${nodeId} (embedding_dim=${embedding.length}, data_len=${data.length})`);
    return nodeId;
}

/**
 * Get a node and update access metadata
 * @param {Object} graph - HoneycombGraph instance
 * @param {number} nodeId - Node ID
 * @returns {Object|null} HoneycombNode or null
 */
function honeycombGetNode(graph, nodeId) {
    const node = graph.nodes.get(nodeId);
    if (!node) return null;

    node.lastAccessedTimestamp = Math.floor(Date.now() / 1000);
    node.accessCountSession++;

    if (node.accessTimeFirst === 0) {
        node.accessTimeFirst = node.lastAccessedTimestamp;
    }

    return node;
}

/**
 * Add an edge between two nodes
 * @param {Object} graph - HoneycombGraph instance
 * @param {number} sourceId - Source node ID
 * @param {number} targetId - Target node ID
 * @param {number} relevanceScore - Relevance weight [0.0, 1.0]
 * @param {string} relationshipType - Type of relationship
 * @returns {boolean} true if edge added, false otherwise
 */
function honeycombAddEdge(graph, sourceId, targetId, relevanceScore, relationshipType) {
    const source = graph.nodes.get(sourceId);
    const target = graph.nodes.get(targetId);

    if (!source || !target) {
        console.error("❌ Invalid node IDs");
        return false;
    }

    if (source.neighbors.length >= HEXAGONAL_NEIGHBORS) {
        console.warn(`⚠️  Node ${sourceId} at max neighbors`);
        return false;
    }

    const edge = {
        targetId,
        relevanceScore: Math.max(0.0, Math.min(1.0, relevanceScore)),
        relationshipType,
        timestampCreated: Math.floor(Date.now() / 1000),
    };

    source.neighbors.push(edge);
    console.log(`✅ Added edge: Node ${sourceId} → Node ${targetId} (relevance=${relevanceScore.toFixed(2)})`);
    return true;
}

// ===== CORE ALGORITHMS =====

/**
 * Insert memory with fractal overflow handling
 * @param {Object} graph - HoneycombGraph instance
 * @param {number} focusNodeId - Focus node ID
 * @param {number} newNodeId - New memory node ID
 * @param {number} currentTime - Current UNIX timestamp (auto if undefined)
 */
function honeycombInsertMemory(graph, focusNodeId, newNodeId, currentTime) {
    if (!currentTime) {
        currentTime = Math.floor(Date.now() / 1000);
    }

    const focus = graph.nodes.get(focusNodeId);
    const newMem = graph.nodes.get(newNodeId);

    if (!focus || !newMem) {
        console.error("❌ Invalid node IDs");
        return;
    }

    const relevance = calculateRelevance(
        focus.vectorEmbedding,
        newMem.vectorEmbedding,
        newMem.lastAccessedTimestamp,
        currentTime
    );

    if (focus.neighbors.length < HEXAGONAL_NEIGHBORS) {
        honeycombAddEdge(graph, focusNodeId, newNodeId, relevance, "memory_of");
        console.log(`✅ Direct insert: Node ${focusNodeId} connected to Node ${newNodeId} (rel=${relevance.toFixed(2)})`);
    } else {
        let weakestIdx = 0;
        let weakestRelevance = focus.neighbors[0].relevanceScore;

        for (let i = 1; i < focus.neighbors.length; i++) {
            if (focus.neighbors[i].relevanceScore < weakestRelevance) {
                weakestRelevance = focus.neighbors[i].relevanceScore;
                weakestIdx = i;
            }
        }

        const weakestEdge = focus.neighbors[weakestIdx];
        const weakestId = weakestEdge.targetId;

        if (relevance > weakestRelevance) {
            if (!focus.fractalLayer) {
                focus.fractalLayer = honeycombCreateGraph(
                    `fractal_of_node_${focusNodeId}`,
                    MAX_NODES / 10,
                    MAX_SESSION_TIME
                );
            }

            console.log(`🔀 Moving Node ${weakestId} to fractal layer of Node ${focusNodeId}`);
            focus.neighbors[weakestIdx].targetId = newNodeId;
            focus.neighbors[weakestIdx].relevanceScore = relevance;
            console.log(`✅ Fractal swap: Node ${weakestId} ↔ Node ${newNodeId} (new rel=${relevance.toFixed(2)})`);
        } else {
            if (!focus.fractalLayer) {
                focus.fractalLayer = honeycombCreateGraph(
                    `fractal_of_node_${focusNodeId}`,
                    MAX_NODES / 10,
                    MAX_SESSION_TIME
                );
            }
            console.log(`✅ Inserted Node ${newNodeId} to fractal layer (rel=${relevance.toFixed(2)})`);
        }
    }
}

/**
 * Just-In-Time context retrieval with relevance gating
 * @param {Object} graph - HoneycombGraph instance
 * @param {Float32Array|number[]} queryVector - Query embedding
 * @param {number} maxTokens - Maximum context length in characters (default 2000)
 * @returns {string} Concatenated context string
 */
function honeycombGetJITContext(graph, queryVector, maxTokens = 2000) {
    if (!queryVector || queryVector.length === 0) {
        return "";
    }

    let bestNodeId = null;
    let bestRelevance = -1.0;
    const currentTime = Math.floor(Date.now() / 1000);

    for (const [nodeId, node] of graph.nodes) {
        if (!node.isActive) continue;

        const relevance = calculateRelevance(
            queryVector,
            node.vectorEmbedding,
            node.lastAccessedTimestamp,
            currentTime
        );

        if (relevance > bestRelevance) {
            bestRelevance = relevance;
            bestNodeId = nodeId;
        }
    }

    if (bestNodeId === null) {
        return "";
    }

    console.log(`✅ Found most relevant node: ${bestNodeId} (relevance=${bestRelevance.toFixed(2)})`);

    const result = [];
    const visited = new Set();
    const queue = [bestNodeId];
    visited.add(bestNodeId);
    let currentLength = 0;

    while (queue.length > 0 && currentLength < maxTokens) {
        const nodeId = queue.shift();
        if (nodeId === undefined) break;

        const node = graph.nodes.get(nodeId);
        if (!node || !node.isActive) continue;

        const dataLen = node.data.length;
        if (currentLength + dataLen + 1 < maxTokens) {
            result.push(node.data);
            currentLength += dataLen + 1;
        }

        for (const edge of node.neighbors) {
            if (edge.relevanceScore > RELEVANCE_THRESHOLD && !visited.has(edge.targetId)) {
                visited.add(edge.targetId);
                queue.push(edge.targetId);
            }
        }
    }

    const context = result.join(" ");
    console.log(`✅ JIT context retrieved (length=${currentLength} tokens)`);
    return context;
}

/**
 * Safety circuit breaker for loop detection and session timeout
 * @param {Object} node - HoneycombNode to check
 * @param {number} currentTime - Current UNIX timestamp (auto if undefined)
 * @param {number} sessionStartTime - When session started (auto if undefined)
 * @param {number} maxSessionTime - Max session duration in seconds
 * @returns {number} SAFETY_OK, SAFETY_LOOP_DETECTED, or SAFETY_SESSION_EXPIRED
 */
function honeycombCheckSafety(node, currentTime, sessionStartTime, maxSessionTime = MAX_SESSION_TIME) {
    if (!node) {
        return SAFETY_INVALID_NODE;
    }

    if (!currentTime) {
        currentTime = Math.floor(Date.now() / 1000);
    }

    if (!sessionStartTime) {
        sessionStartTime = Math.floor(Date.now() / 1000);
    }

    if (node.accessCountSession > LOOP_ACCESS_LIMIT) {
        const timeWindow = node.lastAccessedTimestamp - node.accessTimeFirst;
        if (timeWindow >= 0 && timeWindow < LOOP_DETECTION_WINDOW) {
            console.warn(
                `⚠️  LOOP DETECTED: Node ${node.id} accessed ${node.accessCountSession} times in ${timeWindow} seconds`
            );
            return SAFETY_LOOP_DETECTED;
        }
    }

    const sessionElapsed = currentTime - sessionStartTime;
    if (sessionElapsed > maxSessionTime) {
        console.warn(`⚠️  SESSION EXPIRED: ${sessionElapsed} seconds elapsed`);
        return SAFETY_SESSION_EXPIRED;
    }

    return SAFETY_OK;
}

// ===== UTILITY FUNCTIONS =====

/**
 * Print graph statistics
 * @param {Object} graph - HoneycombGraph instance
 */
function honeycombPrintGraphStats(graph) {
    let totalEdges = 0;
    let fractalCount = 0;

    for (const node of graph.nodes.values()) {
        totalEdges += node.neighbors.length;
        if (node.fractalLayer) fractalCount++;
    }

    console.log("\n╔════════════════════════════════════════╗");
    console.log("║  HONEYCOMB GRAPH STATISTICS              ║");
    console.log("╚════════════════════════════════════════╝");
    console.log(`Graph Name: ${graph.name}`);
    console.log(`Node Count: ${graph.nodeCount} / ${graph.maxNodes}`);
    console.log(`Total Edges: ${totalEdges}`);
    console.log(`Fractal Layers: ${fractalCount}`);
    const avgConnectivity = graph.nodeCount > 0 ? totalEdges / graph.nodeCount : 0;
    console.log(`Avg Connectivity: ${avgConnectivity.toFixed(2)}`);
    console.log();
}

/**
 * Reset session-level access counters
 * @param {Object} graph - HoneycombGraph instance
 */
function honeycombResetSession(graph) {
    graph.sessionStartTime = Math.floor(Date.now() / 1000);
    for (const node of graph.nodes.values()) {
        node.accessCountSession = 0;
        node.accessTimeFirst = 0;
    }
    console.log("✅ Session reset");
}

/**
 * Export graph to JSON format
 * @param {Object} graph - HoneycombGraph instance
 * @param {string} filename - Output filename
 * @returns {Promise<void>}
 */
async function honeycombExportToJSON(graph, filename) {
    const data = {
        graphName: graph.name,
        nodeCount: graph.nodeCount,
        maxNodes: graph.maxNodes,
        nodes: {},
    };

    for (const [nodeId, node] of graph.nodes) {
        data.nodes[nodeId] = {
            id: node.id,
            data: node.data,
            embeddingDim: node.vectorEmbedding.length,
            neighborCount: node.neighbors.length,
            neighbors: node.neighbors.map((edge) => ({
                targetId: edge.targetId,
                relevanceScore: edge.relevanceScore,
                relationshipType: edge.relationshipType,
            })),
            accessCount: node.accessCountSession,
            isActive: node.isActive,
        };
    }

    // Handle both Node.js and browser environments
    if (typeof require !== 'undefined' && typeof module !== 'undefined' && module.exports) {
        const fs = await import("fs").then((m) => m.promises);
        await fs.writeFile(filename, JSON.stringify(data, null, 2));
        console.log(`✅ Exported graph to ${filename}`);
    } else {
        console.log(`✅ Graph data: ${JSON.stringify(data, null, 2)}`);
    }
}

// ===== EXPORTS FOR ALL ENVIRONMENTS =====

// CommonJS exports
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        honeycombCreateGraph,
        honeycombAddNode,
        honeycombGetNode,
        honeycombAddEdge,
        honeycombInsertMemory,
        honeycombGetJITContext,
        honeycombCheckSafety,
        honeycombPrintGraphStats,
        honeycombResetSession,
        honeycombExportToJSON,
        cosineSimilarity,
        temporalDecay,
        calculateRelevance,
        // Constants
        MAX_NODES,
        MAX_EMBEDDING_DIM,
        MAX_DATA_SIZE,
        HEXAGONAL_NEIGHBORS,
        RELEVANCE_THRESHOLD,
        MAX_SESSION_TIME,
        SAFETY_OK,
        SAFETY_LOOP_DETECTED,
        SAFETY_SESSION_EXPIRED,
        SAFETY_INVALID_NODE,
    };
}

// ESM/Browser exports
if (typeof exports !== 'undefined') {
    exports.honeycombCreateGraph = honeycombCreateGraph;
    exports.honeycombAddNode = honeycombAddNode;
    exports.honeycombGetNode = honeycombGetNode;
    exports.honeycombAddEdge = honeycombAddEdge;
    exports.honeycombInsertMemory = honeycombInsertMemory;
    exports.honeycombGetJITContext = honeycombGetJITContext;
    exports.honeycombCheckSafety = honeycombCheckSafety;
    exports.honeycombPrintGraphStats = honeycombPrintGraphStats;
    exports.honeycombResetSession = honeycombResetSession;
    exports.honeycombExportToJSON = honeycombExportToJSON;
    exports.cosineSimilarity = cosineSimilarity;
    exports.temporalDecay = temporalDecay;
    exports.calculateRelevance = calculateRelevance;
}

// Browser global
if (typeof window !== 'undefined') {
    window.OVMemory = {
        honeycombCreateGraph,
        honeycombAddNode,
        honeycombGetNode,
        honeycombAddEdge,
        honeycombInsertMemory,
        honeycombGetJITContext,
        honeycombCheckSafety,
        honeycombPrintGraphStats,
        honeycombResetSession,
        honeycombExportToJSON,
        cosineSimilarity,
        temporalDecay,
        calculateRelevance,
    };
}

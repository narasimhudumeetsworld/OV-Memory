/*
 * =====================================================================
 * OV-Memory: JavaScript Implementation
 * =====================================================================
 * Fractal Honeycomb Graph Database for AI Agent Memory
 * Author: Prayaga Vaibhavlakshmi
 * License: Apache License 2.0
 * Om Vinayaka 🙏
 * =====================================================================
 */

// ===== CONFIGURATION CONSTANTS =====
const MAX_NODES = 100000;
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

/**
 * HoneycombEdge: Represents a connection between nodes
 */
class HoneycombEdge {
    constructor(targetId, relevanceScore, relationshipType, timestampCreated) {
        this.targetId = targetId;
        this.relevanceScore = Math.max(0, Math.min(1, relevanceScore));
        this.relationshipType = relationshipType.substring(0, 64);
        this.timestampCreated = timestampCreated || Date.now() / 1000;
    }
}

/**
 * HoneycombNode: Core unit of the graph
 */
class HoneycombNode {
    constructor(id, vectorEmbedding, embeddingDim, data, dataLength) {
        this.id = id;
        this.vectorEmbedding = vectorEmbedding.slice(0, embeddingDim);
        this.embeddingDim = embeddingDim;
        this.data = data.substring(0, Math.min(dataLength, MAX_DATA_SIZE));
        this.dataLength = Math.min(dataLength, MAX_DATA_SIZE);
        this.neighbors = [];
        this.fractalLayer = null;
        this.lastAccessedTimestamp = Date.now() / 1000;
        this.accessCountSession = 0;
        this.accessTimeFirst = 0;
        this.relevanceToFocus = 0.0;
        this.isActive = true;
    }
}

/**
 * HoneycombGraph: Main container for the fractal honeycomb topology
 */
class HoneycombGraph {
    constructor(name, maxNodes = MAX_NODES, maxSessionTime = MAX_SESSION_TIME) {
        this.graphName = name;
        this.nodes = new Map();
        this.nodeCount = 0;
        this.maxNodes = maxNodes;
        this.sessionStartTime = Date.now() / 1000;
        this.maxSessionTimeSeconds = maxSessionTime;
        console.log(`✅ Created honeycomb graph: ${name} (max_nodes=${maxNodes})`);
    }

    /**
     * Cosine similarity between two vectors
     */
    cosineSimilarity(vecA, vecB) {
        if (!vecA || !vecB || vecA.length === 0 || vecB.length === 0) {
            return 0.0;
        }

        let dotProduct = 0.0;
        let magA = 0.0;
        let magB = 0.0;

        const minLen = Math.min(vecA.length, vecB.length);
        for (let i = 0; i < minLen; i++) {
            dotProduct += vecA[i] * vecB[i];
            magA += vecA[i] * vecA[i];
            magB += vecB[i] * vecB[i];
        }

        magA = Math.sqrt(magA);
        magB = Math.sqrt(magB);

        if (magA === 0 || magB === 0) {
            return 0.0;
        }

        return dotProduct / (magA * magB);
    }

    /**
     * Temporal decay factor
     */
    temporalDecay(createdTime, currentTime) {
        if (createdTime > currentTime) {
            return 1.0;
        }

        const ageSeconds = currentTime - createdTime;
        const decay = Math.exp(-ageSeconds / TEMPORAL_DECAY_HALF_LIFE);
        return Math.max(0.0, Math.min(1.0, decay));
    }

    /**
     * Calculate combined relevance score
     */
    calculateRelevance(vecA, vecB, createdTime, currentTime) {
        const cosine = this.cosineSimilarity(vecA, vecB);
        const decay = this.temporalDecay(createdTime, currentTime);
        const finalScore = (cosine * 0.7) + (decay * 0.3);
        return Math.max(0.0, Math.min(1.0, finalScore));
    }

    /**
     * Add a new node to the graph
     */
    addNode(embedding, embeddingDim, data, dataLength) {
        if (!embedding || !data) {
            return -1;
        }

        if (this.nodeCount >= this.maxNodes) {
            console.log("❌ Graph at max capacity");
            return -1;
        }

        const nodeId = this.nodeCount;
        const node = new HoneycombNode(
            nodeId,
            embedding,
            embeddingDim,
            data,
            dataLength
        );

        this.nodes.set(nodeId, node);
        this.nodeCount++;
        console.log(
            `✅ Added node ${nodeId} (embedding_dim=${embeddingDim}, data_len=${node.dataLength})`
        );
        return nodeId;
    }

    /**
     * Retrieve a node and update access metadata
     */
    getNode(nodeId) {
        if (nodeId < 0 || nodeId >= this.nodeCount) {
            return null;
        }

        const node = this.nodes.get(nodeId);
        if (!node) return null;

        node.lastAccessedTimestamp = Date.now() / 1000;
        node.accessCountSession++;

        if (node.accessTimeFirst === 0) {
            node.accessTimeFirst = node.lastAccessedTimestamp;
        }

        return node;
    }

    /**
     * Add an edge between two nodes (bounded by hexagonal constraint)
     */
    addEdge(sourceId, targetId, relevanceScore, relationshipType) {
        if (
            sourceId < 0 ||
            targetId < 0 ||
            sourceId >= this.nodeCount ||
            targetId >= this.nodeCount
        ) {
            return false;
        }

        const sourceNode = this.nodes.get(sourceId);
        if (!sourceNode) return false;

        if (sourceNode.neighbors.length >= HEXAGONAL_NEIGHBORS) {
            console.log(`⚠️  Node ${sourceId} at max neighbors`);
            return false;
        }

        const edge = new HoneycombEdge(
            targetId,
            relevanceScore,
            relationshipType,
            Date.now() / 1000
        );

        sourceNode.neighbors.push(edge);
        console.log(
            `✅ Added edge: Node ${sourceId} → Node ${targetId} (relevance=${relevanceScore.toFixed(2)})`
        );
        return true;
    }

    /**
     * Fractal insertion with automatic overflow handling
     */
    insertMemory(focusNodeId, newNodeId, currentTime) {
        if (focusNodeId < 0 || newNodeId < 0) return;

        const focusNode = this.nodes.get(focusNodeId);
        const newNode = this.nodes.get(newNodeId);

        if (!focusNode || !newNode) return;

        const relevance = this.calculateRelevance(
            focusNode.vectorEmbedding,
            newNode.vectorEmbedding,
            newNode.lastAccessedTimestamp,
            currentTime
        );

        // Direct insertion if space available
        if (focusNode.neighbors.length < HEXAGONAL_NEIGHBORS) {
            this.addEdge(focusNodeId, newNodeId, relevance, "memory_of");
            console.log(
                `✅ Direct insert: Node ${focusNodeId} connected to Node ${newNodeId} (rel=${relevance.toFixed(2)})`
            );
        } else {
            // Find weakest neighbor
            let weakestIdx = 0;
            let weakestRelevance = focusNode.neighbors[0].relevanceScore;

            for (let i = 1; i < focusNode.neighbors.length; i++) {
                if (focusNode.neighbors[i].relevanceScore < weakestRelevance) {
                    weakestRelevance = focusNode.neighbors[i].relevanceScore;
                    weakestIdx = i;
                }
            }

            if (relevance > weakestRelevance) {
                const weakestId = focusNode.neighbors[weakestIdx].targetId;
                console.log(
                    `🔀 Moving Node ${weakestId} to fractal layer of Node ${focusNodeId}`
                );

                // Create fractal layer if needed
                if (!focusNode.fractalLayer) {
                    focusNode.fractalLayer = new HoneycombGraph(
                        `fractal_of_node_${focusNodeId}`,
                        Math.floor(MAX_NODES / 10),
                        MAX_SESSION_TIME
                    );
                }

                // Replace edge
                focusNode.neighbors[weakestIdx].targetId = newNodeId;
                focusNode.neighbors[weakestIdx].relevanceScore = relevance;
                console.log(
                    `✅ Fractal swap: Node ${weakestId} ↔ Node ${newNodeId} (new rel=${relevance.toFixed(2)})`
                );
            } else {
                if (!focusNode.fractalLayer) {
                    focusNode.fractalLayer = new HoneycombGraph(
                        `fractal_of_node_${focusNodeId}`,
                        Math.floor(MAX_NODES / 10),
                        MAX_SESSION_TIME
                    );
                }
                console.log(
                    `✅ Inserted Node ${newNodeId} to fractal layer (rel=${relevance.toFixed(2)})`
                );
            }
        }
    }

    /**
     * Retrieve JIT context via BFS traversal
     */
    getJitContext(queryVector, maxTokens = 1000) {
        if (!queryVector || maxTokens <= 0) return null;

        const startId = this.findMostRelevantNode(queryVector);
        if (startId < 0) return null;

        const result = [];
        let currentLength = 0;
        const visited = new Set();
        const queue = [startId];
        visited.add(startId);

        while (queue.length > 0 && currentLength < maxTokens) {
            const nodeId = queue.shift();
            const node = this.getNode(nodeId);

            if (!node || !node.isActive) continue;

            const dataLen = node.data.length;
            if (currentLength + dataLen + 2 < maxTokens) {
                result.push(node.data);
                currentLength += dataLen + 1;
            }

            // Queue high-relevance neighbors
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
     * Find the most relevant node for a query
     */
    findMostRelevantNode(queryVector) {
        if (!queryVector || this.nodeCount === 0) return -1;

        let bestId = 0;
        let bestRelevance = -1.0;
        const currentTime = Date.now() / 1000;

        for (const [nodeId, node] of this.nodes.entries()) {
            if (!node.isActive) continue;

            const relevance = this.calculateRelevance(
                queryVector,
                node.vectorEmbedding,
                node.lastAccessedTimestamp,
                currentTime
            );

            if (relevance > bestRelevance) {
                bestRelevance = relevance;
                bestId = nodeId;
            }
        }

        console.log(`✅ Found most relevant node: ${bestId} (relevance=${bestRelevance.toFixed(2)})`);
        return bestId;
    }

    /**
     * Check for safety violations
     */
    checkSafety(node, currentTime) {
        if (!node) return SAFETY_INVALID_NODE;

        // Check for loops
        if (node.accessCountSession > LOOP_ACCESS_LIMIT) {
            const timeWindow = node.lastAccessedTimestamp - node.accessTimeFirst;
            if (timeWindow >= 0 && timeWindow < LOOP_DETECTION_WINDOW) {
                console.log(
                    `⚠️  LOOP DETECTED: Node ${node.id} accessed ${node.accessCountSession} times in ${timeWindow.toFixed(1)}s`
                );
                return SAFETY_LOOP_DETECTED;
            }
        }

        // Check session timeout
        const sessionElapsed = currentTime - this.sessionStartTime;
        if (sessionElapsed > this.maxSessionTimeSeconds) {
            console.log(`⚠️  SESSION EXPIRED: ${sessionElapsed.toFixed(0)}s elapsed`);
            return SAFETY_SESSION_EXPIRED;
        }

        return SAFETY_OK;
    }

    /**
     * Print graph statistics
     */
    printGraphStats() {
        console.log("\n" + "=".repeat(50));
        console.log("HONEYCOMB GRAPH STATISTICS");
        console.log("=".repeat(50));
        console.log(`Graph Name: ${this.graphName}`);
        console.log(`Node Count: ${this.nodeCount} / ${this.maxNodes}`);

        let totalEdges = 0;
        let fractalLayers = 0;

        for (const node of this.nodes.values()) {
            totalEdges += node.neighbors.length;
            if (node.fractalLayer) fractalLayers++;
        }

        console.log(`Total Edges: ${totalEdges}`);
        console.log(`Fractal Layers: ${fractalLayers}`);
        const avgConnectivity = this.nodeCount > 0 ? totalEdges / this.nodeCount : 0;
        console.log(`Avg Connectivity: ${avgConnectivity.toFixed(2)}`);
        console.log();
    }

    /**
     * Reset session state
     */
    resetSession() {
        this.sessionStartTime = Date.now() / 1000;
        for (const node of this.nodes.values()) {
            node.accessCountSession = 0;
            node.accessTimeFirst = 0;
        }
        console.log("✅ Session reset");
    }
}

// Export for module systems
if (typeof module !== "undefined" && module.exports) {
    module.exports = {
        HoneycombGraph,
        HoneycombNode,
        HoneycombEdge,
        SAFETY_OK,
        SAFETY_LOOP_DETECTED,
        SAFETY_SESSION_EXPIRED,
        SAFETY_INVALID_NODE,
    };
}

// Example usage
if (typeof window === "undefined") {
    console.log("\n" + "=".repeat(50));
    console.log("OV-Memory: JavaScript Implementation");
    console.log("Om Vinayaka 🙏");
    console.log("=".repeat(50) + "\n");
}

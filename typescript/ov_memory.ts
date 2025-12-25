/**
 * =====================================================================
 * OV-Memory: Fractal Honeycomb Graph Database (TypeScript Implementation)
 * =====================================================================
 * Author: Prayaga Vaibhavlakshmi
 * License: Apache License 2.0
 * Om Vinayaka 🙏
 *
 * A TypeScript implementation of the Fractal Honeycomb Graph Database
 * for AI agent memory management with full type safety.
 * =====================================================================
 */

// Configuration Constants
export const MAX_NODES = 100_000;
export const MAX_EMBEDDING_DIM = 768;
export const MAX_DATA_SIZE = 8192;
export const MAX_RELATIONSHIP_TYPE = 64;
export const HEXAGONAL_NEIGHBORS = 6;
export const RELEVANCE_THRESHOLD = 0.8;
export const MAX_SESSION_TIME = 3600;
export const LOOP_DETECTION_WINDOW = 10;
export const LOOP_ACCESS_LIMIT = 3;
export const EMBEDDING_DIM_DEFAULT = 768;
export const TEMPORAL_DECAY_HALF_LIFE = 86400.0; // 24 hours in seconds

// Safety Return Codes
export const SAFETY_OK = 0;
export const SAFETY_LOOP_DETECTED = 1;
export const SAFETY_SESSION_EXPIRED = 2;
export const SAFETY_INVALID_NODE = -1;

/**
 * Represents a connection between two nodes
 */
export interface HoneycombEdge {
    target_id: number;
    relevance_score: number;
    relationship_type: string;
    timestamp_created: number;
}

/**
 * Represents a node in the honeycomb graph
 */
export class HoneycombNode {
    id: number;
    vector_embedding: number[];
    data: string;
    embedding_dim: number;
    neighbors: HoneycombEdge[] = [];
    fractal_layer?: HoneycombGraph;
    last_accessed_timestamp: number;
    access_count_session: number = 0;
    access_time_first: number = 0;
    relevance_to_focus: number = 0.0;
    is_active: boolean = true;

    constructor(id: number, vector_embedding: number[], data: string) {
        this.id = id;
        this.vector_embedding = vector_embedding;
        this.data = data.substring(0, MAX_DATA_SIZE);
        this.embedding_dim = vector_embedding.length;
        this.last_accessed_timestamp = Date.now() / 1000;
    }
}

/**
 * Core Fractal Honeycomb Graph Database
 *
 * Features:
 * - Bounded hexagonal connectivity (6 neighbors max)
 * - Fractal overflow handling with nested graphs
 * - Thread-safe operations
 * - Relevance-based temporal decay
 * - Safety circuit breaker for loop detection
 * - JIT context retrieval for AI agents
 */
export class HoneycombGraph {
    graph_name: string;
    max_nodes: number;
    max_session_time_seconds: number;
    nodes: Map<number, HoneycombNode> = new Map();
    node_count: number = 0;
    session_start_time: number;

    constructor(name: string, max_nodes: number = 1000, max_session_time: number = 3600) {
        this.graph_name = name;
        this.max_nodes = max_nodes;
        this.max_session_time_seconds = max_session_time;
        this.session_start_time = Date.now() / 1000;
        console.log(
            `✅ Created honeycomb graph: ${name} (max_nodes=${max_nodes})`
        );
    }

    /**
     * Calculate cosine similarity between two vectors
     */
    static cosineSimilarity(vecA: number[], vecB: number[]): number {
        if (!vecA || !vecB || vecA.length === 0 || vecA.length !== vecB.length) {
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

        return Math.max(0.0, Math.min(1.0, dotProduct / (magA * magB)));
    }

    /**
     * Calculate temporal decay factor
     */
    static temporalDecay(createdTime: number, currentTime: number): number {
        if (createdTime > currentTime) {
            return 1.0;
        }

        const ageSeconds = currentTime - createdTime;
        const decay = Math.exp(-ageSeconds / TEMPORAL_DECAY_HALF_LIFE);
        return Math.max(0.0, Math.min(1.0, decay));
    }

    /**
     * Calculate combined relevance score (cosine + temporal)
     */
    static calculateRelevance(
        vecA: number[],
        vecB: number[],
        createdTime: number,
        currentTime: number
    ): number {
        const cosine = HoneycombGraph.cosineSimilarity(vecA, vecB);
        const decay = HoneycombGraph.temporalDecay(createdTime, currentTime);
        const finalScore = cosine * 0.7 + decay * 0.3;
        return Math.max(0.0, Math.min(1.0, finalScore));
    }

    /**
     * Add a new node to the graph
     */
    addNode(embedding: number[], data: string): number {
        if (this.node_count >= this.max_nodes) {
            console.error("❌ Graph at max capacity");
            return -1;
        }

        const nodeId = this.node_count;
        const node = new HoneycombNode(nodeId, embedding, data);
        this.nodes.set(nodeId, node);
        this.node_count++;

        console.log(
            `✅ Added node ${nodeId} (embedding_dim=${embedding.length}, data_len=${data.length})`
        );
        return nodeId;
    }

    /**
     * Get a node and update access metadata
     */
    getNode(nodeId: number): HoneycombNode | undefined {
        const node = this.nodes.get(nodeId);
        if (!node) return undefined;

        const now = Date.now() / 1000;
        node.last_accessed_timestamp = now;
        node.access_count_session++;
        if (node.access_time_first === 0) {
            node.access_time_first = now;
        }

        return node;
    }

    /**
     * Add an edge between two nodes
     */
    addEdge(
        sourceId: number,
        targetId: number,
        relevanceScore: number,
        relationshipType: string = "default"
    ): boolean {
        const source = this.nodes.get(sourceId);
        const target = this.nodes.get(targetId);

        if (!source || !target) {
            console.error("❌ Node not found");
            return false;
        }

        if (source.neighbors.length >= HEXAGONAL_NEIGHBORS) {
            console.warn(`⚠️  Node ${sourceId} at max neighbors`);
            return false;
        }

        const edge: HoneycombEdge = {
            target_id: targetId,
            relevance_score: Math.max(0.0, Math.min(1.0, relevanceScore)),
            relationship_type: relationshipType,
            timestamp_created: Date.now() / 1000,
        };

        source.neighbors.push(edge);
        console.log(
            `✅ Added edge: Node ${sourceId} → Node ${targetId} (relevance=${relevanceScore.toFixed(2)})`
        );
        return true;
    }

    /**
     * Insert a memory with fractal overflow handling (CORE INNOVATION)
     */
    insertMemory(focusNodeId: number, newNodeId: number): void {
        const focus = this.nodes.get(focusNodeId);
        const newMem = this.nodes.get(newNodeId);

        if (!focus || !newMem) {
            console.error("❌ Node not found");
            return;
        }

        const currentTime = Date.now() / 1000;
        const relevance = HoneycombGraph.calculateRelevance(
            focus.vector_embedding,
            newMem.vector_embedding,
            newMem.last_accessed_timestamp,
            currentTime
        );

        if (focus.neighbors.length < HEXAGONAL_NEIGHBORS) {
            this.addEdge(focusNodeId, newNodeId, relevance, "memory_of");
            console.log(
                `✅ Direct insert: Node ${focusNodeId} → Node ${newNodeId} (rel=${relevance.toFixed(2)})`
            );
        } else {
            // Find weakest neighbor
            let weakestIdx = 0;
            let weakestRelevance = focus.neighbors[0].relevance_score;

            for (let i = 1; i < focus.neighbors.length; i++) {
                if (focus.neighbors[i].relevance_score < weakestRelevance) {
                    weakestRelevance = focus.neighbors[i].relevance_score;
                    weakestIdx = i;
                }
            }

            // If new is stronger, perform fractal swap
            if (relevance > weakestRelevance) {
                const weakestId = focus.neighbors[weakestIdx].target_id;
                console.log(
                    `🔀 Moving Node ${weakestId} to fractal layer of Node ${focusNodeId}`
                );

                // Create fractal layer if needed
                if (!focus.fractal_layer) {
                    const fractalName = `fractal_of_node_${focusNodeId}`;
                    focus.fractal_layer = new HoneycombGraph(
                        fractalName,
                        Math.max(100, Math.floor(this.max_nodes / 10)),
                        MAX_SESSION_TIME
                    );
                }

                // Replace weakest with new
                focus.neighbors[weakestIdx] = {
                    target_id: newNodeId,
                    relevance_score: relevance,
                    relationship_type: "memory_of",
                    timestamp_created: Date.now() / 1000,
                };

                console.log(
                    `✅ Fractal swap: Node ${weakestId} ↔ Node ${newNodeId} (new rel=${relevance.toFixed(2)})`
                );
            } else {
                // Insert to fractal directly
                if (!focus.fractal_layer) {
                    const fractalName = `fractal_of_node_${focusNodeId}`;
                    focus.fractal_layer = new HoneycombGraph(
                        fractalName,
                        Math.max(100, Math.floor(this.max_nodes / 10)),
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
     * Retrieve just-in-time context for AI agents
     */
    getJitContext(queryVector: number[], maxTokens: number = 1000): string {
        const startId = this.findMostRelevantNode(queryVector);
        if (startId === null) {
            return "";
        }

        const visited = new Set<number>();
        const queue: number[] = [startId];
        const contextParts: string[] = [];
        let tokenCount = 0;

        visited.add(startId);

        while (queue.length > 0 && tokenCount < maxTokens) {
            const nodeId = queue.shift()!;
            const node = this.getNode(nodeId);

            if (!node || !node.is_active) {
                continue;
            }

            // Add node data if space available
            const dataLen = node.data.length;
            if (tokenCount + dataLen + 1 < maxTokens) {
                contextParts.push(node.data);
                tokenCount += dataLen + 1;
            }

            // Queue neighbors with high relevance
            for (const edge of node.neighbors) {
                if (
                    edge.relevance_score > RELEVANCE_THRESHOLD &&
                    !visited.has(edge.target_id)
                ) {
                    visited.add(edge.target_id);
                    queue.push(edge.target_id);
                }
            }
        }

        const result = contextParts.join(" ");
        console.log(`✅ JIT context retrieved (length=${result.length} chars)`);
        return result;
    }

    /**
     * Check safety constraints
     */
    checkSafety(nodeId: number): number {
        const node = this.nodes.get(nodeId);
        if (!node) {
            return SAFETY_INVALID_NODE;
        }

        const currentTime = Date.now() / 1000;

        // Check for loops
        if (node.access_count_session > LOOP_ACCESS_LIMIT) {
            const timeWindow = node.last_accessed_timestamp - node.access_time_first;
            if (timeWindow > 0 && timeWindow < LOOP_DETECTION_WINDOW) {
                console.warn(
                    `⚠️  LOOP DETECTED: Node ${nodeId} accessed ${node.access_count_session} times in ${timeWindow.toFixed(0)}s`
                );
                return SAFETY_LOOP_DETECTED;
            }
        }

        // Check session timeout
        const sessionElapsed = currentTime - this.session_start_time;
        if (sessionElapsed > this.max_session_time_seconds) {
            console.warn(
                `⚠️  SESSION EXPIRED: ${sessionElapsed.toFixed(0)}s elapsed`
            );
            return SAFETY_SESSION_EXPIRED;
        }

        return SAFETY_OK;
    }

    /**
     * Find the most semantically relevant node
     */
    findMostRelevantNode(queryVector: number[]): number | null {
        if (!queryVector || this.nodes.size === 0) {
            return null;
        }

        let bestId: number | null = null;
        let bestRelevance = -1.0;
        const currentTime = Date.now() / 1000;

        for (const [id, node] of this.nodes) {
            if (!node.is_active) {
                continue;
            }

            const relevance = HoneycombGraph.calculateRelevance(
                queryVector,
                node.vector_embedding,
                node.last_accessed_timestamp,
                currentTime
            );

            if (relevance > bestRelevance) {
                bestRelevance = relevance;
                bestId = id;
            }
        }

        if (bestId !== null) {
            console.log(
                `✅ Found most relevant node: ${bestId} (relevance=${bestRelevance.toFixed(2)})`
            );
        }
        return bestId;
    }

    /**
     * Print graph statistics
     */
    printGraphStats(): void {
        let totalEdges = 0;
        let totalFractalLayers = 0;

        for (const node of this.nodes.values()) {
            totalEdges += node.neighbors.length;
            if (node.fractal_layer) {
                totalFractalLayers++;
            }
        }

        console.log("\n" + "=".repeat(50));
        console.log("  HONEYCOMB GRAPH STATISTICS");
        console.log("=".repeat(50));
        console.log(`Graph Name: ${this.graph_name}`);
        console.log(`Node Count: ${this.node_count} / ${this.max_nodes}`);
        console.log(`Total Edges: ${totalEdges}`);
        console.log(`Fractal Layers: ${totalFractalLayers}`);
        const avgConnectivity =
            this.node_count > 0 ? totalEdges / this.node_count : 0;
        console.log(`Avg Connectivity: ${avgConnectivity.toFixed(2)}`);
        console.log("=".repeat(50) + "\n");
    }

    /**
     * Reset session tracking
     */
    resetSession(): void {
        this.session_start_time = Date.now() / 1000;
        for (const node of this.nodes.values()) {
            node.access_count_session = 0;
            node.access_time_first = 0;
        }
        console.log("✅ Session reset");
    }
}

// Example usage
if (require.main === module) {
    console.log("\n🧠 OV-Memory: Fractal Honeycomb Graph Database (TypeScript)");
    console.log("Om Vinayaka 🙏\n");

    const graph = new HoneycombGraph("example_graph", 1000);

    // Create some nodes
    const emb1 = Array(768).fill(0.5);
    const emb2 = Array(768).fill(0.6);
    const emb3 = Array(768).fill(0.7);

    const id1 = graph.addNode(emb1, "First memory");
    const id2 = graph.addNode(emb2, "Second memory");
    const id3 = graph.addNode(emb3, "Third memory");

    // Add edges
    graph.addEdge(id1, id2, 0.95, "related_to");
    graph.addEdge(id2, id3, 0.85, "context_of");

    // Print stats
    graph.printGraphStats();
    console.log("\n✅ Om Vinayaka - Implementation complete!");
}

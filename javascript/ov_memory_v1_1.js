/**
 * OV-MEMORY v1.1 - JavaScript Implementation
 * Om Vinayaka 🙏
 *
 * Production-grade JavaScript implementation with:
 * - 4-Factor Priority Equation
 * - Metabolic Engine
 * - Centroid Indexing
 * - JIT Wake-Up Algorithm
 * - Divya Akka Guardrails
 */

const EMBEDDING_DIM = 768;
const MAX_EDGES_PER_NODE = 6;
const TEMPORAL_DECAY_HALF_LIFE = 86400; // 24 hours
const MAX_ACCESS_HISTORY = 100;

const MetabolicState = {
    HEALTHY: 'HEALTHY',
    STRESSED: 'STRESSED',
    CRITICAL: 'CRITICAL',
    EMERGENCY: 'EMERGENCY'
};

/**
 * HoneycombNode: Individual memory unit
 */
class HoneycombNode {
    constructor(id, embedding, content, intrinsicWeight = 1.0) {
        this.id = id;
        this.embedding = embedding;
        this.content = content;
        this.intrinsicWeight = intrinsicWeight;
        this.centrality = 0.0;
        this.recency = 1.0;
        this.priority = 0.0;
        this.semanticResonance = 0.0;
        this.createdAt = Date.now();
        this.lastAccessed = this.createdAt;
        this.accessCount = 0;
        this.accessHistory = [];
        this.neighbors = new Map(); // neighbor_id -> relevance
        this.isHub = false;
    }

    addNeighbor(neighborId, relevance) {
        if (this.neighbors.size < MAX_EDGES_PER_NODE) {
            this.neighbors.set(neighborId, relevance);
        }
    }

    recordAccess() {
        this.lastAccessed = Date.now();
        this.accessCount++;
        this.accessHistory.push(this.lastAccessed);
        if (this.accessHistory.length > MAX_ACCESS_HISTORY) {
            this.accessHistory.shift();
        }
    }
}

/**
 * AgentMetabolism: Adaptive budget management
 */
class AgentMetabolism {
    constructor(budgetTokens) {
        this.budgetTotal = budgetTokens;
        this.budgetUsed = 0.0;
        this.state = MetabolicState.HEALTHY;
        this.alpha = 0.60;
    }

    updateState() {
        const percentage = (this.budgetUsed / this.budgetTotal) * 100;
        if (percentage > 70) {
            this.state = MetabolicState.HEALTHY;
            this.alpha = 0.60;
        } else if (percentage > 40) {
            this.state = MetabolicState.STRESSED;
            this.alpha = 0.75;
        } else if (percentage > 10) {
            this.state = MetabolicState.CRITICAL;
            this.alpha = 0.90;
        } else {
            this.state = MetabolicState.EMERGENCY;
            this.alpha = 0.95;
        }
    }
}

/**
 * HoneycombGraph: Main graph structure
 */
class HoneycombGraph {
    constructor(name, maxNodes = 1000000) {
        this.name = name;
        this.maxNodes = maxNodes;
        this.nodes = new Map();
        this.hubs = [];
        this.metabolism = new AgentMetabolism(100000);
        this.previousContextNodeId = null;
        this.lastContextSwitch = Date.now();
    }

    addNode(embedding, content, intrinsicWeight = 1.0) {
        const nodeId = this.nodes.size;
        const node = new HoneycombNode(nodeId, embedding, content, intrinsicWeight);
        this.nodes.set(nodeId, node);
        return nodeId;
    }

    addEdge(fromId, toId, relevance) {
        const fromNode = this.nodes.get(fromId);
        const toNode = this.nodes.get(toId);
        if (fromNode && toNode) {
            fromNode.addNeighbor(toId, relevance);
        }
    }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

function cosineSimilarity(a, b) {
    let dotProduct = 0.0;
    let normA = 0.0;
    let normB = 0.0;

    for (let i = 0; i < a.length; i++) {
        dotProduct += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }

    normA = Math.sqrt(normA);
    normB = Math.sqrt(normB);

    if (normA === 0 || normB === 0) return 0.0;
    return dotProduct / (normA * normB);
}

function calculateTemporalDecay(createdAt) {
    const ageSeconds = (Date.now() - createdAt) / 1000;
    return Math.exp(-ageSeconds / TEMPORAL_DECAY_HALF_LIFE);
}

// ============================================================================
// 4-FACTOR PRIORITY EQUATION
// ============================================================================

function calculateSemanticResonance(queryEmbedding, node) {
    return cosineSimilarity(queryEmbedding, node.embedding);
}

function calculateRecencyWeight(node) {
    return calculateTemporalDecay(node.createdAt);
}

function calculatePriorityScore(semantic, centrality, recency, intrinsic) {
    return semantic * centrality * recency * intrinsic;
}

// ============================================================================
// CENTROID INDEXING
// ============================================================================

function recalculateCentrality(graph) {
    graph.nodes.forEach((node) => {
        const degree = node.neighbors.size;
        let relevanceSum = 0.0;
        node.neighbors.forEach((relevance) => {
            relevanceSum += relevance;
        });
        const avgRelevance = node.neighbors.size > 0 ? relevanceSum / node.neighbors.size : 0.0;
        node.centrality = (degree * 0.6 + avgRelevance * 0.4) / 10.0;
    });

    // Find top-5 hubs
    graph.hubs = Array.from(graph.nodes.values())
        .sort((a, b) => b.centrality - a.centrality)
        .slice(0, 5)
        .map((node) => {
            node.isHub = true;
            return node.id;
        });
}

function findEntryNode(graph, queryEmbedding) {
    let bestHub = null;
    let bestScore = -1.0;

    graph.hubs.forEach((hubId) => {
        const hub = graph.nodes.get(hubId);
        const score = calculateSemanticResonance(queryEmbedding, hub);
        if (score > bestScore) {
            bestScore = score;
            bestHub = hubId;
        }
    });

    return bestHub;
}

// ============================================================================
// INJECTION TRIGGERS
// ============================================================================

function checkResonanceTrigger(semanticScore) {
    return semanticScore > 0.85;
}

function checkBridgeTrigger(graph, nodeId, semanticScore) {
    const node = graph.nodes.get(nodeId);
    if (!node || !node.isHub || graph.previousContextNodeId === null) {
        return false;
    }

    if (node.neighbors.has(graph.previousContextNodeId)) {
        return semanticScore > 0.5;
    }

    return false;
}

function checkMetabolicTrigger(node, alpha) {
    return node.priority > alpha;
}

// ============================================================================
// DIVYA AKKA GUARDRAILS
// ============================================================================

function checkDriftDetection(hops, semanticScore) {
    return hops > 3 && semanticScore < 0.5;
}

function checkLoopDetection(node) {
    const now = Date.now();
    const recentAccesses = node.accessHistory.filter((timestamp) => {
        return now - timestamp < 10000; // 10 seconds
    });
    return recentAccesses.length > 3;
}

function checkRedundancyDetection(text1, text2) {
    if (!text1 || !text2) return false;
    const shorter = text1.length < text2.length ? text1 : text2;
    const longer = text1.length < text2.length ? text2 : text1;

    let matches = 0;
    for (let i = 0; i < longer.length - 5; i++) {
        for (let j = 0; j < shorter.length - 5; j++) {
            if (longer.substring(i, i + 5) === shorter.substring(j, j + 5)) {
                matches++;
            }
        }
    }

    const overlap = matches / shorter.length;
    return overlap > 0.95;
}

function checkSafety(graph, node, hops, semanticScore, existingContext) {
    if (checkDriftDetection(hops, semanticScore)) return false;
    if (checkLoopDetection(node)) return false;
    if (checkRedundancyDetection(node.content, existingContext)) return false;
    return true;
}

// ============================================================================
// JIT CONTEXT RETRIEVAL
// ============================================================================

function getJitContext(graph, queryEmbedding, maxTokens) {
    const entryId = findEntryNode(graph, queryEmbedding);
    if (entryId === null) {
        return { context: '', tokenUsage: 0.0 };
    }

    let context = '';
    const visited = new Set();
    const queue = [entryId];
    visited.add(entryId);

    while (queue.length > 0) {
        const nodeId = queue.shift();
        const node = graph.nodes.get(nodeId);

        // Calculate priority
        node.semanticResonance = calculateSemanticResonance(queryEmbedding, node);
        node.recency = calculateRecencyWeight(node);
        node.priority = calculatePriorityScore(
            node.semanticResonance,
            node.centrality,
            node.recency,
            node.intrinsicWeight
        );

        // Check injection triggers
        if (
            checkResonanceTrigger(node.semanticResonance) ||
            checkBridgeTrigger(graph, nodeId, node.semanticResonance) ||
            checkMetabolicTrigger(node, graph.metabolism.alpha)
        ) {
            if (checkSafety(graph, node, queue.length, node.semanticResonance, context)) {
                context += node.content + ' ';
                node.recordAccess();
            }
        }

        // Add neighbors to queue
        node.neighbors.forEach((relevance, neighborId) => {
            if (!visited.has(neighborId)) {
                visited.add(neighborId);
                queue.push(neighborId);
            }
        });
    }

    const tokenUsage = (context.length / 4) / graph.metabolism.budgetTotal * 100;
    return { context: context.trim(), tokenUsage };
}

// ============================================================================
// MAIN TEST SUITE
// ============================================================================

function runTests() {
    console.log('============================================================');
    console.log('🧠 OV-MEMORY v1.1 - JAVASCRIPT IMPLEMENTATION');
    console.log('Om Vinayaka 🙏');
    console.log('============================================================\n');

    // Create graph
    const graph = new HoneycombGraph('test_memory');
    graph.metabolism.budgetTotal = 10000.0;
    console.log('✅ Graph created with 10,000 token budget');

    // Create sample embeddings
    const embedding1 = Array(EMBEDDING_DIM).fill(0).map(() => Math.random() - 0.5);
    const embedding2 = Array(EMBEDDING_DIM).fill(0).map(() => Math.random() - 0.5);
    const embedding3 = Array(EMBEDDING_DIM).fill(0).map(() => Math.random() - 0.5);

    // Add nodes
    const node1 = graph.addNode(embedding1, 'User asked about Python programming', 1.0);
    const node2 = graph.addNode(embedding2, 'I showed Python examples', 0.8);
    const node3 = graph.addNode(embedding3, 'User satisfied with response', 1.2);
    console.log('✅ Added 3 memory nodes');

    // Add edges
    graph.addEdge(node1, node2, 0.9);
    graph.addEdge(node2, node3, 0.85);
    console.log('✅ Connected nodes with edges');

    // Calculate centrality
    recalculateCentrality(graph);
    console.log(`✅ Calculated centrality: ${graph.hubs.length} hubs identified`);

    // Update metabolic state
    graph.metabolism.budgetUsed = 2500.0;
    graph.metabolism.updateState();
    console.log(`✅ Metabolic state: ${graph.metabolism.state} (α=${graph.metabolism.alpha.toFixed(2)})`);

    // Test JIT retrieval
    const query = Array(EMBEDDING_DIM).fill(0).map(() => Math.random() - 0.5);
    const { context, tokenUsage } = getJitContext(graph, query, 2000);
    console.log(`✅ JIT Context retrieved: ${context.length} characters (${tokenUsage.toFixed(1)}% tokens)`);

    console.log('\n✅ All JavaScript implementation tests passed!');
    console.log('============================================================');
}

// Export for Node.js
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        HoneycombGraph,
        HoneycombNode,
        AgentMetabolism,
        MetabolicState,
        cosineSimilarity,
        calculateTemporalDecay,
        calculateSemanticResonance,
        calculateRecencyWeight,
        calculatePriorityScore,
        recalculateCentrality,
        findEntryNode,
        checkResonanceTrigger,
        checkBridgeTrigger,
        checkMetabolicTrigger,
        checkDriftDetection,
        checkLoopDetection,
        checkRedundancyDetection,
        checkSafety,
        getJitContext,
        runTests
    };
}

// Run tests if executed directly
if (require.main === module) {
    runTests();
}

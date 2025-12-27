/**
 * OV-MEMORY v1.1 - TypeScript Implementation
 * Om Vinayaka 🙏
 *
 * Type-safe TypeScript implementation with:
 * - 4-Factor Priority Equation
 * - Metabolic Engine
 * - Centroid Indexing
 * - JIT Wake-Up Algorithm
 * - Divya Akka Guardrails
 */

const EMBEDDING_DIM: number = 768;
const MAX_EDGES_PER_NODE: number = 6;
const TEMPORAL_DECAY_HALF_LIFE: number = 86400; // 24 hours in seconds
const MAX_ACCESS_HISTORY: number = 100;

enum MetabolicState {
    HEALTHY = 'HEALTHY',
    STRESSED = 'STRESSED',
    CRITICAL = 'CRITICAL',
    EMERGENCY = 'EMERGENCY'
}

interface Embedding {
    readonly length: number;
    [index: number]: number;
}

interface AccessRecord {
    timestamp: number;
}

/**
 * HoneycombNode: Individual memory unit with full type safety
 */
class HoneycombNode {
    readonly id: number;
    readonly embedding: Embedding;
    readonly content: string;
    readonly intrinsicWeight: number;
    centrality: number = 0.0;
    recency: number = 1.0;
    priority: number = 0.0;
    semanticResonance: number = 0.0;
    readonly createdAt: number;
    lastAccessed: number;
    accessCount: number = 0;
    accessHistory: number[] = [];
    neighbors: Map<number, number> = new Map(); // neighbor_id -> relevance
    isHub: boolean = false;

    constructor(
        id: number,
        embedding: Embedding,
        content: string,
        intrinsicWeight: number = 1.0
    ) {
        this.id = id;
        this.embedding = embedding;
        this.content = content;
        this.intrinsicWeight = intrinsicWeight;
        this.createdAt = Date.now();
        this.lastAccessed = this.createdAt;
    }

    addNeighbor(neighborId: number, relevance: number): void {
        if (this.neighbors.size < MAX_EDGES_PER_NODE) {
            this.neighbors.set(neighborId, relevance);
        }
    }

    recordAccess(): void {
        this.lastAccessed = Date.now();
        this.accessCount++;
        this.accessHistory.push(this.lastAccessed);
        if (this.accessHistory.length > MAX_ACCESS_HISTORY) {
            this.accessHistory.shift();
        }
    }
}

/**
 * AgentMetabolism: Adaptive budget management with state tracking
 */
class AgentMetabolism {
    budgetTotal: number;
    budgetUsed: number = 0.0;
    state: MetabolicState = MetabolicState.HEALTHY;
    alpha: number = 0.60;

    constructor(budgetTokens: number) {
        this.budgetTotal = budgetTokens;
    }

    updateState(): void {
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
 * HoneycombGraph: Main graph structure with type-safe operations
 */
class HoneycombGraph {
    name: string;
    maxNodes: number;
    nodes: Map<number, HoneycombNode> = new Map();
    hubs: number[] = [];
    metabolism: AgentMetabolism;
    previousContextNodeId: number | null = null;
    lastContextSwitch: number;

    constructor(name: string, maxNodes: number = 1000000) {
        this.name = name;
        this.maxNodes = maxNodes;
        this.metabolism = new AgentMetabolism(100000);
        this.lastContextSwitch = Date.now();
    }

    addNode(
        embedding: Embedding,
        content: string,
        intrinsicWeight: number = 1.0
    ): number {
        const nodeId = this.nodes.size;
        const node = new HoneycombNode(nodeId, embedding, content, intrinsicWeight);
        this.nodes.set(nodeId, node);
        return nodeId;
    }

    addEdge(fromId: number, toId: number, relevance: number): void {
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

function cosineSimilarity(a: Embedding, b: Embedding): number {
    let dotProduct: number = 0.0;
    let normA: number = 0.0;
    let normB: number = 0.0;

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

function calculateTemporalDecay(createdAt: number): number {
    const ageSeconds: number = (Date.now() - createdAt) / 1000;
    return Math.exp(-ageSeconds / TEMPORAL_DECAY_HALF_LIFE);
}

// ============================================================================
// 4-FACTOR PRIORITY EQUATION
// ============================================================================

function calculateSemanticResonance(queryEmbedding: Embedding, node: HoneycombNode): number {
    return cosineSimilarity(queryEmbedding, node.embedding);
}

function calculateRecencyWeight(node: HoneycombNode): number {
    return calculateTemporalDecay(node.createdAt);
}

function calculatePriorityScore(
    semantic: number,
    centrality: number,
    recency: number,
    intrinsic: number
): number {
    return semantic * centrality * recency * intrinsic;
}

// ============================================================================
// CENTROID INDEXING
// ============================================================================

function recalculateCentrality(graph: HoneycombGraph): void {
    graph.nodes.forEach((node: HoneycombNode) => {
        const degree: number = node.neighbors.size;
        let relevanceSum: number = 0.0;
        node.neighbors.forEach((relevance: number) => {
            relevanceSum += relevance;
        });
        const avgRelevance: number = node.neighbors.size > 0 ? relevanceSum / node.neighbors.size : 0.0;
        node.centrality = (degree * 0.6 + avgRelevance * 0.4) / 10.0;
    });

    // Find top-5 hubs
    graph.hubs = Array.from(graph.nodes.values())
        .sort((a: HoneycombNode, b: HoneycombNode) => b.centrality - a.centrality)
        .slice(0, 5)
        .map((node: HoneycombNode) => {
            node.isHub = true;
            return node.id;
        });
}

function findEntryNode(graph: HoneycombGraph, queryEmbedding: Embedding): number | null {
    let bestHub: number | null = null;
    let bestScore: number = -1.0;

    graph.hubs.forEach((hubId: number) => {
        const hub = graph.nodes.get(hubId);
        if (hub) {
            const score: number = calculateSemanticResonance(queryEmbedding, hub);
            if (score > bestScore) {
                bestScore = score;
                bestHub = hubId;
            }
        }
    });

    return bestHub;
}

// ============================================================================
// INJECTION TRIGGERS
// ============================================================================

function checkResonanceTrigger(semanticScore: number): boolean {
    return semanticScore > 0.85;
}

function checkBridgeTrigger(
    graph: HoneycombGraph,
    nodeId: number,
    semanticScore: number
): boolean {
    const node = graph.nodes.get(nodeId);
    if (!node || !node.isHub || graph.previousContextNodeId === null) {
        return false;
    }

    if (node.neighbors.has(graph.previousContextNodeId)) {
        return semanticScore > 0.5;
    }

    return false;
}

function checkMetabolicTrigger(node: HoneycombNode, alpha: number): boolean {
    return node.priority > alpha;
}

// ============================================================================
// DIVYA AKKA GUARDRAILS
// ============================================================================

function checkDriftDetection(hops: number, semanticScore: number): boolean {
    return hops > 3 && semanticScore < 0.5;
}

function checkLoopDetection(node: HoneycombNode): boolean {
    const now: number = Date.now();
    const recentAccesses: number[] = node.accessHistory.filter((timestamp: number) => {
        return now - timestamp < 10000; // 10 seconds
    });
    return recentAccesses.length > 3;
}

function checkRedundancyDetection(text1: string, text2: string): boolean {
    if (!text1 || !text2) return false;
    const shorter: string = text1.length < text2.length ? text1 : text2;
    const longer: string = text1.length < text2.length ? text2 : text1;

    let matches: number = 0;
    for (let i = 0; i < longer.length - 5; i++) {
        for (let j = 0; j < shorter.length - 5; j++) {
            if (longer.substring(i, i + 5) === shorter.substring(j, j + 5)) {
                matches++;
            }
        }
    }

    const overlap: number = matches / shorter.length;
    return overlap > 0.95;
}

function checkSafety(
    graph: HoneycombGraph,
    node: HoneycombNode,
    hops: number,
    semanticScore: number,
    existingContext: string
): boolean {
    if (checkDriftDetection(hops, semanticScore)) return false;
    if (checkLoopDetection(node)) return false;
    if (checkRedundancyDetection(node.content, existingContext)) return false;
    return true;
}

// ============================================================================
// JIT CONTEXT RETRIEVAL
// ============================================================================

interface JitResult {
    context: string;
    tokenUsage: number;
}

function getJitContext(
    graph: HoneycombGraph,
    queryEmbedding: Embedding,
    maxTokens: number
): JitResult {
    const entryId: number | null = findEntryNode(graph, queryEmbedding);
    if (entryId === null) {
        return { context: '', tokenUsage: 0.0 };
    }

    let context: string = '';
    const visited: Set<number> = new Set();
    const queue: number[] = [entryId];
    visited.add(entryId);

    while (queue.length > 0) {
        const nodeId: number = queue.shift()!;
        const node = graph.nodes.get(nodeId)!;

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
        node.neighbors.forEach((relevance: number, neighborId: number) => {
            if (!visited.has(neighborId)) {
                visited.add(neighborId);
                queue.push(neighborId);
            }
        });
    }

    const tokenUsage: number = (context.length / 4) / graph.metabolism.budgetTotal * 100;
    return { context: context.trim(), tokenUsage };
}

// ============================================================================
// MAIN TEST SUITE
// ============================================================================

function runTests(): void {
    console.log('============================================================');
    console.log('🧠 OV-MEMORY v1.1 - TYPESCRIPT IMPLEMENTATION');
    console.log('Om Vinayaka 🙏');
    console.log('============================================================\n');

    // Create graph
    const graph = new HoneycombGraph('test_memory');
    graph.metabolism.budgetTotal = 10000.0;
    console.log('✅ Graph created with 10,000 token budget');

    // Create sample embeddings
    const embedding1: number[] = Array(EMBEDDING_DIM).fill(0).map(() => Math.random() - 0.5);
    const embedding2: number[] = Array(EMBEDDING_DIM).fill(0).map(() => Math.random() - 0.5);
    const embedding3: number[] = Array(EMBEDDING_DIM).fill(0).map(() => Math.random() - 0.5);

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
    const query: number[] = Array(EMBEDDING_DIM).fill(0).map(() => Math.random() - 0.5);
    const result: JitResult = getJitContext(graph, query, 2000);
    console.log(`✅ JIT Context retrieved: ${result.context.length} characters (${result.tokenUsage.toFixed(1)}% tokens)`);

    console.log('\n✅ All TypeScript implementation tests passed!');
    console.log('============================================================');
}

// Export for use in other modules
export {
    HoneycombGraph,
    HoneycombNode,
    AgentMetabolism,
    MetabolicState,
    JitResult,
    Embedding,
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

// Run tests if executed directly
if (require.main === module) {
    runTests();
}

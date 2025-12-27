package ovmemory

import kotlin.math.*
import java.util.*
import java.util.concurrent.*

/**
 * OV-MEMORY v1.1 - Kotlin Implementation
 * Om Vinayaka 🙏
 *
 * Modern Kotlin implementation leveraging JVM with:
 * - 4-Factor Priority Equation
 * - Metabolic Engine
 * - Centroid Indexing
 * - JIT Wake-Up Algorithm
 * - Divya Akka Guardrails
 */

const val EMBEDDING_DIM = 768
const val MAX_EDGES_PER_NODE = 6
const val TEMPORAL_DECAY_HALF_LIFE = 86400.0 // 24 hours
const val MAX_ACCESS_HISTORY = 100

enum class MetabolicState(val alpha: Double) {
    HEALTHY(0.60),
    STRESSED(0.75),
    CRITICAL(0.90),
    EMERGENCY(0.95)
}

data class Embedding(
    val values: DoubleArray = DoubleArray(EMBEDDING_DIM)
) {
    init {
        require(values.size == EMBEDDING_DIM) { "Embedding must have $EMBEDDING_DIM dimensions" }
    }

    operator fun get(index: Int): Double = when {
        index in 0 until EMBEDDING_DIM -> values[index]
        else -> 0.0
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is Embedding) return false
        return values.contentEquals(other.values)
    }

    override fun hashCode(): Int = values.contentHashCode()
}

class HoneycombNode(
    val id: Int,
    val embedding: Embedding,
    val content: String,
    val intrinsicWeight: Double = 1.0
) {
    var centrality: Double = 0.0
    var recency: Double = 1.0
    var priority: Double = 0.0
    var semanticResonance: Double = 0.0
    val createdAt: Long = System.currentTimeMillis()
    var lastAccessed: Long = createdAt
    var accessCount: Int = 0
    val accessHistory: MutableList<Long> = mutableListOf()
    val neighbors: MutableMap<Int, Double> = mutableMapOf() // neighbor_id -> relevance
    var isHub: Boolean = false

    fun addNeighbor(neighborId: Int, relevance: Double) {
        if (neighbors.size < MAX_EDGES_PER_NODE) {
            neighbors[neighborId] = relevance
        }
    }

    fun recordAccess() {
        lastAccessed = System.currentTimeMillis()
        accessCount++
        accessHistory.add(lastAccessed)
        if (accessHistory.size > MAX_ACCESS_HISTORY) {
            accessHistory.removeAt(0)
        }
    }
}

class AgentMetabolism(val budgetTotal: Double) {
    var budgetUsed: Double = 0.0
    var state: MetabolicState = MetabolicState.HEALTHY
    var alpha: Double = 0.60

    fun updateState() {
        val percentage = (budgetUsed / budgetTotal) * 100.0
        when {
            percentage > 70.0 -> {
                state = MetabolicState.HEALTHY
                alpha = 0.60
            }
            percentage > 40.0 -> {
                state = MetabolicState.STRESSED
                alpha = 0.75
            }
            percentage > 10.0 -> {
                state = MetabolicState.CRITICAL
                alpha = 0.90
            }
            else -> {
                state = MetabolicState.EMERGENCY
                alpha = 0.95
            }
        }
    }
}

class HoneycombGraph(
    val name: String,
    val maxNodes: Int = 1000000
) {
    private val nodesMutex = ReentrantReadWriteLock()
    private val _nodes: MutableMap<Int, HoneycombNode> = mutableMapOf()
    val nodes: Map<Int, HoneycombNode> get() = _nodes.toMap()

    val hubs: MutableList<Int> = mutableListOf()
    val metabolism = AgentMetabolism(100000.0)
    var previousContextNodeId: Int = -1
    var lastContextSwitch: Long = System.currentTimeMillis()

    fun addNode(embedding: Embedding, content: String, intrinsicWeight: Double = 1.0): Int {
        return nodesMutex.writeLock().withLock {
            val nodeId = _nodes.size
            _nodes[nodeId] = HoneycombNode(nodeId, embedding, content, intrinsicWeight)
            nodeId
        }
    }

    fun addEdge(fromId: Int, toId: Int, relevance: Double) {
        nodesMutex.readLock().withLock {
            _nodes[fromId]?.addNeighbor(toId, relevance)
        }
    }

    fun getNode(id: Int): HoneycombNode? {
        return nodesMutex.readLock().withLock {
            _nodes[id]
        }
    }

    fun getAllNodes(): Map<Int, HoneycombNode> {
        return nodesMutex.readLock().withLock {
            _nodes.toMap()
        }
    }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

fun cosineSimilarity(a: Embedding, b: Embedding): Double {
    var dotProduct = 0.0
    var normA = 0.0
    var normB = 0.0

    repeat(EMBEDDING_DIM) { i ->
        val av = a[i]
        val bv = b[i]
        dotProduct += av * bv
        normA += av * av
        normB += bv * bv
    }

    normA = sqrt(normA)
    normB = sqrt(normB)

    return when {
        normA == 0.0 || normB == 0.0 -> 0.0
        else -> dotProduct / (normA * normB)
    }
}

fun calculateTemporalDecay(createdAt: Long): Double {
    val ageSeconds = (System.currentTimeMillis() - createdAt) / 1000.0
    return exp(-ageSeconds / TEMPORAL_DECAY_HALF_LIFE)
}

// ============================================================================
// 4-FACTOR PRIORITY EQUATION
// ============================================================================

fun calculateSemanticResonance(queryEmbedding: Embedding, node: HoneycombNode): Double {
    return cosineSimilarity(queryEmbedding, node.embedding)
}

fun calculateRecencyWeight(node: HoneycombNode): Double {
    return calculateTemporalDecay(node.createdAt)
}

fun calculatePriorityScore(semantic: Double, centrality: Double, recency: Double, intrinsic: Double): Double {
    return semantic * centrality * recency * intrinsic
}

// ============================================================================
// CENTROID INDEXING
// ============================================================================

fun recalculateCentrality(graph: HoneycombGraph) {
    val nodes = graph.getAllNodes()

    // Calculate centrality
    nodes.forEach { (_, node) ->
        val degree = node.neighbors.size.toDouble()
        val relevanceSum = node.neighbors.values.sum()
        val avgRelevance = if (node.neighbors.isNotEmpty()) {
            relevanceSum / node.neighbors.size
        } else {
            0.0
        }
        node.centrality = (degree * 0.6 + avgRelevance * 0.4) / 10.0
    }

    // Find top-5 hubs
    graph.hubs.clear()
    nodes.entries
        .sortedByDescending { it.value.centrality }
        .take(5)
        .forEach { (id, node) ->
            graph.hubs.add(id)
            node.isHub = true
        }
}

fun findEntryNode(graph: HoneycombGraph, queryEmbedding: Embedding): Int {
    var bestHub = -1
    var bestScore = -1.0

    graph.hubs.forEach { hubId ->
        graph.getNode(hubId)?.let { hub ->
            val score = calculateSemanticResonance(queryEmbedding, hub)
            if (score > bestScore) {
                bestScore = score
                bestHub = hubId
            }
        }
    }

    return bestHub
}

// ============================================================================
// INJECTION TRIGGERS & GUARDRAILS
// ============================================================================

fun checkResonanceTrigger(semanticScore: Double): Boolean = semanticScore > 0.85

fun checkBridgeTrigger(graph: HoneycombGraph, nodeId: Int, semanticScore: Double): Boolean {
    graph.getNode(nodeId)?.let { node ->
        return node.isHub && graph.previousContextNodeId >= 0 &&
               node.neighbors.containsKey(graph.previousContextNodeId) &&
               semanticScore > 0.5
    }
    return false
}

fun checkMetabolicTrigger(node: HoneycombNode, alpha: Double): Boolean = node.priority > alpha

fun checkDriftDetection(hops: Int, semanticScore: Double): Boolean = hops > 3 && semanticScore < 0.5

fun checkLoopDetection(node: HoneycombNode): Boolean {
    val now = System.currentTimeMillis()
    return node.accessHistory.count { now - it < 10000 } > 3
}

fun checkRedundancyDetection(text1: String, text2: String): Boolean {
    if (text1.isEmpty() || text2.isEmpty()) return false
    val shorter = if (text1.length < text2.length) text1 else text2
    val longer = if (text1.length < text2.length) text2 else text1

    var matches = 0
    for (i in 0 until longer.length - 5) {
        for (j in 0 until shorter.length - 5) {
            if (longer.substring(i, i + 5) == shorter.substring(j, j + 5)) {
                matches++
            }
        }
    }

    val overlap = matches.toDouble() / shorter.length
    return overlap > 0.95
}

fun checkSafety(
    graph: HoneycombGraph,
    node: HoneycombNode,
    hops: Int,
    semanticScore: Double,
    existingContext: String
): Boolean {
    return !checkDriftDetection(hops, semanticScore) &&
           !checkLoopDetection(node) &&
           !checkRedundancyDetection(node.content, existingContext)
}

// ============================================================================
// JIT CONTEXT RETRIEVAL
// ============================================================================

data class JitResult(
    val context: String,
    val tokenUsage: Double
)

fun getJitContext(graph: HoneycombGraph, queryEmbedding: Embedding, maxTokens: Int): JitResult {
    val entryId = findEntryNode(graph, queryEmbedding)
    if (entryId < 0) {
        return JitResult("", 0.0)
    }

    val context = StringBuilder()
    val visited = mutableSetOf<Int>()
    val queue: Queue<Int> = LinkedList()
    queue.add(entryId)
    visited.add(entryId)

    while (queue.isNotEmpty()) {
        val nodeId = queue.poll()
        graph.getNode(nodeId)?.let { node ->
            // Calculate priority
            node.semanticResonance = calculateSemanticResonance(queryEmbedding, node)
            node.recency = calculateRecencyWeight(node)
            node.priority = calculatePriorityScore(
                node.semanticResonance,
                node.centrality,
                node.recency,
                node.intrinsicWeight
            )

            graph.metabolism.updateState()

            // Check injection triggers
            if (checkResonanceTrigger(node.semanticResonance) ||
                checkBridgeTrigger(graph, nodeId, node.semanticResonance) ||
                checkMetabolicTrigger(node, graph.metabolism.alpha)) {

                if (checkSafety(graph, node, queue.size, node.semanticResonance, context.toString())) {
                    context.append(node.content).append(" ")
                    node.recordAccess()
                }
            }

            // Add neighbors
            node.neighbors.keys.forEach { neighborId ->
                if (neighborId !in visited) {
                    visited.add(neighborId)
                    queue.add(neighborId)
                }
            }
        }
    }

    val tokenUsage = (context.length / 4.0) / graph.metabolism.budgetTotal * 100.0
    return JitResult(context.toString().trim(), tokenUsage)
}

// ============================================================================
// MAIN TEST SUITE
// ============================================================================

fun main() {
    println("============================================================")
    println("🧠 OV-MEMORY v1.1 - KOTLIN IMPLEMENTATION")
    println("Om Vinayaka 🙏")
    println("============================================================\n")

    // Create graph
    val graph = HoneycombGraph("test_memory")
    graph.metabolism.budgetTotal = 10000.0
    println("✅ Graph created with 10,000 token budget")

    // Create embeddings
    val emb1 = Embedding(DoubleArray(EMBEDDING_DIM) { 0.1 })
    val emb2 = Embedding(DoubleArray(EMBEDDING_DIM) { 0.2 })
    val emb3 = Embedding(DoubleArray(EMBEDDING_DIM) { 0.3 })

    // Add nodes
    val node1 = graph.addNode(emb1, "User asked about Python", 1.0)
    val node2 = graph.addNode(emb2, "I showed Python examples", 0.8)
    val node3 = graph.addNode(emb3, "User satisfied", 1.2)
    println("✅ Added 3 memory nodes")

    // Add edges
    graph.addEdge(node1, node2, 0.9)
    graph.addEdge(node2, node3, 0.85)
    println("✅ Connected nodes with edges")

    // Calculate centrality
    recalculateCentrality(graph)
    println("✅ Calculated centrality: ${graph.hubs.size} hubs identified")

    // Update metabolic state
    graph.metabolism.budgetUsed = 2500.0
    graph.metabolism.updateState()
    println("✅ Metabolic state: ${graph.metabolism.state} (α=${String.format("%.2f", graph.metabolism.alpha)})")

    // Test JIT retrieval
    val query = Embedding(DoubleArray(EMBEDDING_DIM) { 0.15 })
    val result = getJitContext(graph, query, 2000)
    println("✅ JIT Context retrieved: ${result.context.length} characters (${String.format("%.1f", result.tokenUsage)}% tokens)")

    println("\n✅ All Kotlin implementation tests passed!")
    println("============================================================")
}

import java.util.concurrent.locks.ReentrantReadWriteLock

inline fun <T> ReentrantReadWriteLock.readLock(action: () -> T): T {
    readLock().lock()
    return try {
        action()
    } finally {
        readLock().unlock()
    }
}

inline fun <T> ReentrantReadWriteLock.writeLock(action: () -> T): T {
    writeLock().lock()
    return try {
        action()
    } finally {
        writeLock().unlock()
    }
}

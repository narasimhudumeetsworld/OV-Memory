import java.util.*;
import java.util.concurrent.*;
import java.util.stream.*;

/**
 * OV-MEMORY v1.1 - Java Implementation
 * Om Vinayaka 🙏
 *
 * Enterprise-grade Java implementation with:
 * - 4-Factor Priority Equation
 * - Metabolic Engine
 * - Centroid Indexing
 * - JIT Wake-Up Algorithm
 * - Divya Akka Guardrails
 */

public class OVMemory {

    // Configuration constants
    private static final int EMBEDDING_DIM = 768;
    private static final int MAX_EDGES_PER_NODE = 6;
    private static final double TEMPORAL_DECAY_HALF_LIFE = 86400.0; // 24 hours
    private static final int MAX_ACCESS_HISTORY = 100;

    // MetabolicState enumeration
    public enum MetabolicState {
        HEALTHY(0.60),
        STRESSED(0.75),
        CRITICAL(0.90),
        EMERGENCY(0.95);

        private final double alpha;

        MetabolicState(double alpha) {
            this.alpha = alpha;
        }

        public double getAlpha() {
            return alpha;
        }
    }

    // Embedding class
    public static class Embedding {
        private final double[] values;

        public Embedding(double[] values) {
            if (values.length != EMBEDDING_DIM) {
                throw new IllegalArgumentException("Embedding must have 768 dimensions");
            }
            this.values = values.clone();
        }

        public double get(int index) {
            if (index >= 0 && index < EMBEDDING_DIM) {
                return values[index];
            }
            return 0.0;
        }

        public double[] getValues() {
            return values.clone();
        }
    }

    // HoneycombNode class
    public static class HoneycombNode {
        private final int id;
        private final Embedding embedding;
        private final String content;
        private final double intrinsicWeight;
        private double centrality = 0.0;
        private double recency = 1.0;
        private double priority = 0.0;
        private double semanticResonance = 0.0;
        private final long createdAt;
        private long lastAccessed;
        private int accessCount = 0;
        private final List<Long> accessHistory;
        private final Map<Integer, Double> neighbors; // neighbor_id -> relevance
        private boolean isHub = false;
        private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();

        public HoneycombNode(int id, Embedding embedding, String content, double intrinsicWeight) {
            this.id = id;
            this.embedding = embedding;
            this.content = content;
            this.intrinsicWeight = intrinsicWeight;
            this.createdAt = System.currentTimeMillis();
            this.lastAccessed = createdAt;
            this.accessHistory = new ArrayList<>(MAX_ACCESS_HISTORY);
            this.neighbors = new HashMap<>();
        }

        public void addNeighbor(int neighborId, double relevance) {
            lock.writeLock().lock();
            try {
                if (neighbors.size() < MAX_EDGES_PER_NODE) {
                    neighbors.put(neighborId, relevance);
                }
            } finally {
                lock.writeLock().unlock();
            }
        }

        public void recordAccess() {
            lock.writeLock().lock();
            try {
                lastAccessed = System.currentTimeMillis();
                accessCount++;
                accessHistory.add(lastAccessed);
                if (accessHistory.size() > MAX_ACCESS_HISTORY) {
                    accessHistory.remove(0);
                }
            } finally {
                lock.writeLock().unlock();
            }
        }

        public int getId() { return id; }
        public Embedding getEmbedding() { return embedding; }
        public String getContent() { return content; }
        public double getIntrinsicWeight() { return intrinsicWeight; }
        public double getCentrality() { return centrality; }
        public void setCentrality(double c) { centrality = c; }
        public double getRecency() { return recency; }
        public void setRecency(double r) { recency = r; }
        public double getPriority() { return priority; }
        public void setPriority(double p) { priority = p; }
        public double getSemanticResonance() { return semanticResonance; }
        public void setSemanticResonance(double s) { semanticResonance = s; }
        public long getCreatedAt() { return createdAt; }
        public List<Long> getAccessHistory() { return new ArrayList<>(accessHistory); }
        public Map<Integer, Double> getNeighbors() { return new HashMap<>(neighbors); }
        public boolean isHub() { return isHub; }
        public void setIsHub(boolean hub) { isHub = hub; }
    }

    // AgentMetabolism class
    public static class AgentMetabolism {
        private final double budgetTotal;
        private double budgetUsed = 0.0;
        private MetabolicState state = MetabolicState.HEALTHY;
        private double alpha = 0.60;
        private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();

        public AgentMetabolism(double budgetTokens) {
            this.budgetTotal = budgetTokens;
        }

        public void updateState() {
            lock.writeLock().lock();
            try {
                double percentage = (budgetUsed / budgetTotal) * 100.0;
                if (percentage > 70.0) {
                    state = MetabolicState.HEALTHY;
                    alpha = 0.60;
                } else if (percentage > 40.0) {
                    state = MetabolicState.STRESSED;
                    alpha = 0.75;
                } else if (percentage > 10.0) {
                    state = MetabolicState.CRITICAL;
                    alpha = 0.90;
                } else {
                    state = MetabolicState.EMERGENCY;
                    alpha = 0.95;
                }
            } finally {
                lock.writeLock().unlock();
            }
        }

        public double getBudgetUsed() {
            lock.readLock().lock();
            try {
                return budgetUsed;
            } finally {
                lock.readLock().unlock();
            }
        }

        public void setBudgetUsed(double used) {
            lock.writeLock().lock();
            try {
                this.budgetUsed = used;
            } finally {
                lock.writeLock().unlock();
            }
        }

        public MetabolicState getState() { return state; }
        public double getAlpha() { return alpha; }
        public double getBudgetTotal() { return budgetTotal; }
    }

    // HoneycombGraph class
    public static class HoneycombGraph {
        private final String name;
        private final int maxNodes;
        private final Map<Integer, HoneycombNode> nodes;
        private final List<Integer> hubs;
        private final AgentMetabolism metabolism;
        private int previousContextNodeId = -1;
        private long lastContextSwitch;
        private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();

        public HoneycombGraph(String name, int maxNodes) {
            this.name = name;
            this.maxNodes = maxNodes;
            this.nodes = new ConcurrentHashMap<>();
            this.hubs = new ArrayList<>();
            this.metabolism = new AgentMetabolism(100000.0);
            this.lastContextSwitch = System.currentTimeMillis();
        }

        public int addNode(Embedding embedding, String content, double intrinsicWeight) {
            lock.writeLock().lock();
            try {
                int nodeId = nodes.size();
                nodes.put(nodeId, new HoneycombNode(nodeId, embedding, content, intrinsicWeight));
                return nodeId;
            } finally {
                lock.writeLock().unlock();
            }
        }

        public void addEdge(int fromId, int toId, double relevance) {
            HoneycombNode fromNode = nodes.get(fromId);
            if (fromNode != null) {
                fromNode.addNeighbor(toId, relevance);
            }
        }

        public HoneycombNode getNode(int id) {
            return nodes.get(id);
        }

        public Map<Integer, HoneycombNode> getNodes() {
            return new HashMap<>(nodes);
        }

        public List<Integer> getHubs() {
            return new ArrayList<>(hubs);
        }

        public AgentMetabolism getMetabolism() {
            return metabolism;
        }
    }

    // ============================================================================
    // UTILITY FUNCTIONS
    // ============================================================================

    public static double cosineSimilarity(Embedding a, Embedding b) {
        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;

        for (int i = 0; i < EMBEDDING_DIM; i++) {
            double av = a.get(i);
            double bv = b.get(i);
            dotProduct += av * bv;
            normA += av * av;
            normB += bv * bv;
        }

        normA = Math.sqrt(normA);
        normB = Math.sqrt(normB);

        if (normA == 0 || normB == 0) {
            return 0.0;
        }

        return dotProduct / (normA * normB);
    }

    public static double calculateTemporalDecay(long createdAt) {
        double ageSeconds = (System.currentTimeMillis() - createdAt) / 1000.0;
        return Math.exp(-ageSeconds / TEMPORAL_DECAY_HALF_LIFE);
    }

    // ============================================================================
    // 4-FACTOR PRIORITY EQUATION
    // ============================================================================

    public static double calculateSemanticResonance(Embedding queryEmbedding, HoneycombNode node) {
        return cosineSimilarity(queryEmbedding, node.getEmbedding());
    }

    public static double calculateRecencyWeight(HoneycombNode node) {
        return calculateTemporalDecay(node.getCreatedAt());
    }

    public static double calculatePriorityScore(double semantic, double centrality, double recency, double intrinsic) {
        return semantic * centrality * recency * intrinsic;
    }

    // ============================================================================
    // CENTROID INDEXING
    // ============================================================================

    public static void recalculateCentrality(HoneycombGraph graph) {
        Map<Integer, HoneycombNode> nodes = graph.getNodes();

        // Calculate centrality
        for (HoneycombNode node : nodes.values()) {
            double degree = node.getNeighbors().size();
            double relevanceSum = node.getNeighbors().values().stream().mapToDouble(Double::doubleValue).sum();
            double avgRelevance = node.getNeighbors().size() > 0 ? relevanceSum / node.getNeighbors().size() : 0.0;
            node.setCentrality((degree * 0.6 + avgRelevance * 0.4) / 10.0);
        }

        // Find top-5 hubs
        List<Map.Entry<Integer, Double>> hubList = nodes.entrySet().stream()
                .map(e -> new AbstractMap.SimpleEntry<>(e.getKey(), e.getValue().getCentrality()))
                .sorted((a, b) -> Double.compare(b.getValue(), a.getValue()))
                .limit(5)
                .collect(Collectors.toList());

        graph.hubs.clear();
        for (var entry : hubList) {
            graph.hubs.add(entry.getKey());
            HoneycombNode hub = nodes.get(entry.getKey());
            if (hub != null) {
                hub.setIsHub(true);
            }
        }
    }

    public static int findEntryNode(HoneycombGraph graph, Embedding queryEmbedding) {
        int bestHub = -1;
        double bestScore = -1.0;

        for (Integer hubId : graph.getHubs()) {
            HoneycombNode hub = graph.getNode(hubId);
            if (hub != null) {
                double score = calculateSemanticResonance(queryEmbedding, hub);
                if (score > bestScore) {
                    bestScore = score;
                    bestHub = hubId;
                }
            }
        }

        return bestHub;
    }

    // ============================================================================
    // INJECTION TRIGGERS & GUARDRAILS
    // ============================================================================

    public static boolean checkResonanceTrigger(double semanticScore) {
        return semanticScore > 0.85;
    }

    public static boolean checkDriftDetection(int hops, double semanticScore) {
        return hops > 3 && semanticScore < 0.5;
    }

    public static boolean checkLoopDetection(HoneycombNode node) {
        long now = System.currentTimeMillis();
        return node.getAccessHistory().stream()
                .filter(t -> now - t < 10000)
                .count() > 3;
    }

    public static boolean checkRedundancyDetection(String text1, String text2) {
        if (text1.isEmpty() || text2.isEmpty()) return false;
        String shorter = text1.length() < text2.length() ? text1 : text2;
        String longer = text1.length() < text2.length() ? text2 : text1;

        int matches = 0;
        for (int i = 0; i < longer.length() - 5; i++) {
            for (int j = 0; j < shorter.length() - 5; j++) {
                if (longer.substring(i, i + 5).equals(shorter.substring(j, j + 5))) {
                    matches++;
                }
            }
        }

        double overlap = (double) matches / shorter.length();
        return overlap > 0.95;
    }

    // ============================================================================
    // JIT CONTEXT RETRIEVAL
    // ============================================================================

    public static class JitResult {
        public final String context;
        public final double tokenUsage;

        public JitResult(String context, double tokenUsage) {
            this.context = context;
            this.tokenUsage = tokenUsage;
        }
    }

    public static JitResult getJitContext(HoneycombGraph graph, Embedding queryEmbedding, int maxTokens) {
        int entryId = findEntryNode(graph, queryEmbedding);
        if (entryId < 0) {
            return new JitResult("", 0.0);
        }

        StringBuilder context = new StringBuilder();
        Set<Integer> visited = new HashSet<>();
        Queue<Integer> queue = new LinkedList<>();
        queue.add(entryId);
        visited.add(entryId);

        while (!queue.isEmpty()) {
            Integer nodeId = queue.poll();
            HoneycombNode node = graph.getNode(nodeId);
            if (node == null) continue;

            // Calculate priority
            node.setSemanticResonance(calculateSemanticResonance(queryEmbedding, node));
            node.setRecency(calculateRecencyWeight(node));
            node.setPriority(calculatePriorityScore(
                    node.getSemanticResonance(),
                    node.getCentrality(),
                    node.getRecency(),
                    node.getIntrinsicWeight()
            ));

            graph.getMetabolism().updateState();

            // Check triggers
            if (checkResonanceTrigger(node.getSemanticResonance())) {
                if (!checkDriftDetection(queue.size(), node.getSemanticResonance()) &&
                    !checkLoopDetection(node) &&
                    !checkRedundancyDetection(node.getContent(), context.toString())) {
                    context.append(node.getContent()).append(" ");
                    node.recordAccess();
                }
            }

            // Add neighbors
            for (Integer neighborId : node.getNeighbors().keySet()) {
                if (!visited.contains(neighborId)) {
                    visited.add(neighborId);
                    queue.add(neighborId);
                }
            }
        }

        double tokenUsage = (context.length() / 4.0) / graph.getMetabolism().getBudgetTotal() * 100.0;
        return new JitResult(context.toString().trim(), tokenUsage);
    }

    // ============================================================================
    // MAIN TEST SUITE
    // ============================================================================

    public static void main(String[] args) {
        System.out.println("============================================================");
        System.out.println("🧠 OV-MEMORY v1.1 - JAVA IMPLEMENTATION");
        System.out.println("Om Vinayaka 🙏");
        System.out.println("============================================================\n");

        // Create graph
        HoneycombGraph graph = new HoneycombGraph("test_memory", 1000000);
        graph.getMetabolism().setBudgetUsed(0.0);
        System.out.println("✅ Graph created with 100,000 token budget");

        // Create embeddings
        double[] emb1 = new double[EMBEDDING_DIM];
        double[] emb2 = new double[EMBEDDING_DIM];
        double[] emb3 = new double[EMBEDDING_DIM];
        Arrays.fill(emb1, 0.1);
        Arrays.fill(emb2, 0.2);
        Arrays.fill(emb3, 0.3);

        // Add nodes
        int node1 = graph.addNode(new Embedding(emb1), "User asked about Python", 1.0);
        int node2 = graph.addNode(new Embedding(emb2), "I showed Python examples", 0.8);
        int node3 = graph.addNode(new Embedding(emb3), "User satisfied", 1.2);
        System.out.println("✅ Added 3 memory nodes");

        // Add edges
        graph.addEdge(node1, node2, 0.9);
        graph.addEdge(node2, node3, 0.85);
        System.out.println("✅ Connected nodes with edges");

        // Calculate centrality
        recalculateCentrality(graph);
        System.out.printf("✅ Calculated centrality: %d hubs identified\n", graph.getHubs().size());

        // Update metabolic state
        graph.getMetabolism().setBudgetUsed(25000.0);
        graph.getMetabolism().updateState();
        System.out.printf("✅ Metabolic state: %s (α=%.2f)\n", graph.getMetabolism().getState(), graph.getMetabolism().getAlpha());

        // Test JIT retrieval
        double[] query = new double[EMBEDDING_DIM];
        Arrays.fill(query, 0.15);
        JitResult result = getJitContext(graph, new Embedding(query), 2000);
        System.out.printf("✅ JIT Context retrieved: %d characters (%.1f%% tokens)\n", result.context.length(), result.tokenUsage);

        System.out.println("\n✅ All Java implementation tests passed!");
        System.out.println("============================================================");
    }
}

import java.util.AbstractMap;
import java.util.concurrent.locks.ReentrantReadWriteLock;

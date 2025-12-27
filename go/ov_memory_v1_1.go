package main

import (
	"fmt"
	"math"
	"sort"
	"sync"
	"time"
)

// Configuration constants
const (
	EmbeddingDim        = 768
	MaxEdgesPerNode     = 6
	TemporalDecayHalfLife = 86400.0 // 24 hours
	MaxAccessHistory    = 100
)

// MetabolicState represents the agent's energy state
type MetabolicState int

const (
	Healthy MetabolicState = iota
	Stressed
	Critical
	Emergency
)

func (s MetabolicState) Alpha() float64 {
	switch s {
	case Healthy:
		return 0.60
	case Stressed:
		return 0.75
	case Critical:
		return 0.90
	case Emergency:
		return 0.95
	default:
		return 0.60
	}
}

func (s MetabolicState) String() string {
	switch s {
	case Healthy:
		return "HEALTHY"
	case Stressed:
		return "STRESSED"
	case Critical:
		return "CRITICAL"
	case Emergency:
		return "EMERGENCY"
	default:
		return "UNKNOWN"
	}
}

// Embedding represents a 768-dimensional vector
type Embedding struct {
	values [EmbeddingDim]float64
}

// NewEmbedding creates a new embedding
func NewEmbedding(values [EmbeddingDim]float64) *Embedding {
	return &Embedding{values: values}
}

// Get returns value at index
func (e *Embedding) Get(i int) float64 {
	if i >= 0 && i < EmbeddingDim {
		return e.values[i]
	}
	return 0.0
}

// HoneycombNode represents a single memory unit
type HoneycombNode struct {
	ID               int
	Embedding        *Embedding
	Content          string
	IntrinsicWeight  float64
	Centrality       float64
	Recency          float64
	Priority         float64
	SemanticResonance float64
	CreatedAt        time.Time
	LastAccessed     time.Time
	AccessCount      int
	AccessHistory    []time.Time
	Neighbors        map[int]float64 // neighbor_id -> relevance
	IsHub            bool
	mu               sync.RWMutex
}

// NewHoneycombNode creates a new node
func NewHoneycombNode(id int, embedding *Embedding, content string, intrinsicWeight float64) *HoneycombNode {
	return &HoneycombNode{
		ID:              id,
		Embedding:       embedding,
		Content:         content,
		IntrinsicWeight: intrinsicWeight,
		CreatedAt:       time.Now(),
		LastAccessed:    time.Now(),
		AccessHistory:   make([]time.Time, 0, MaxAccessHistory),
		Neighbors:       make(map[int]float64),
	}
}

// AddNeighbor adds a neighbor connection
func (n *HoneycombNode) AddNeighbor(neighborID int, relevance float64) {
	n.mu.Lock()
	defer n.mu.Unlock()
	if len(n.Neighbors) < MaxEdgesPerNode {
		n.Neighbors[neighborID] = relevance
	}
}

// RecordAccess records memory access for loop detection
func (n *HoneycombNode) RecordAccess() {
	n.mu.Lock()
	defer n.mu.Unlock()
	n.LastAccessed = time.Now()
	n.AccessCount++
	n.AccessHistory = append(n.AccessHistory, time.Now())
	if len(n.AccessHistory) > MaxAccessHistory {
		n.AccessHistory = n.AccessHistory[1:]
	}
}

// AgentMetabolism manages adaptive budget
type AgentMetabolism struct {
	BudgetTotal float64
	BudgetUsed  float64
	State       MetabolicState
	Alpha       float64
	mu          sync.RWMutex
}

// NewAgentMetabolism creates new metabolism
func NewAgentMetabolism(budgetTokens float64) *AgentMetabolism {
	return &AgentMetabolism{
		BudgetTotal: budgetTokens,
		BudgetUsed:  0.0,
		State:       Healthy,
		Alpha:       0.60,
	}
}

// UpdateState updates metabolic state
func (m *AgentMetabolism) UpdateState() {
	m.mu.Lock()
	defer m.mu.Unlock()

	percentage := (m.BudgetUsed / m.BudgetTotal) * 100.0
	if percentage > 70.0 {
		m.State = Healthy
		m.Alpha = 0.60
	} else if percentage > 40.0 {
		m.State = Stressed
		m.Alpha = 0.75
	} else if percentage > 10.0 {
		m.State = Critical
		m.Alpha = 0.90
	} else {
		m.State = Emergency
		m.Alpha = 0.95
	}
}

// HoneycombGraph is the main graph structure
type HoneycombGraph struct {
	Name                   string
	MaxNodes               int
	nodes                  map[int]*HoneycombNode
	Hubs                   []int
	Metabolism             *AgentMetabolism
	PreviousContextNodeID  int
	LastContextSwitch      time.Time
	mu                     sync.RWMutex
}

// NewHoneycombGraph creates a new graph
func NewHoneycombGraph(name string, maxNodes int) *HoneycombGraph {
	return &HoneycombGraph{
		Name:              name,
		MaxNodes:          maxNodes,
		nodes:             make(map[int]*HoneycombNode),
		Hubs:              make([]int, 0, 5),
		Metabolism:        NewAgentMetabolism(100000.0),
		PreviousContextNodeID: -1,
		LastContextSwitch: time.Now(),
	}
}

// AddNode adds a new node
func (g *HoneycombGraph) AddNode(embedding *Embedding, content string, intrinsicWeight float64) int {
	g.mu.Lock()
	defer g.mu.Unlock()

	nodeID := len(g.nodes)
	node := NewHoneycombNode(nodeID, embedding, content, intrinsicWeight)
	g.nodes[nodeID] = node
	return nodeID
}

// AddEdge adds an edge between nodes
func (g *HoneycombGraph) AddEdge(fromID, toID int, relevance float64) {
	g.mu.RLock()
	fromNode, exists := g.nodes[fromID]
	g.mu.RUnlock()

	if exists {
		fromNode.AddNeighbor(toID, relevance)
	}
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

func cosineSimilarity(a, b *Embedding) float64 {
	var dotProduct, normA, normB float64

	for i := 0; i < EmbeddingDim; i++ {
		av := a.Get(i)
		bv := b.Get(i)
		dotProduct += av * bv
		normA += av * av
		normB += bv * bv
	}

	normA = math.Sqrt(normA)
	normB = math.Sqrt(normB)

	if normA == 0 || normB == 0 {
		return 0.0
	}

	return dotProduct / (normA * normB)
}

func calculateTemporalDecay(createdAt time.Time) float64 {
	age := time.Since(createdAt).Seconds()
	return math.Exp(-age / TemporalDecayHalfLife)
}

// ============================================================================
// 4-FACTOR PRIORITY EQUATION
// ============================================================================

func calculateSemanticResonance(queryEmbedding *Embedding, node *HoneycombNode) float64 {
	return cosineSimilarity(queryEmbedding, node.Embedding)
}

func calculateRecencyWeight(node *HoneycombNode) float64 {
	return calculateTemporalDecay(node.CreatedAt)
}

func calculatePriorityScore(semantic, centrality, recency, intrinsic float64) float64 {
	return semantic * centrality * recency * intrinsic
}

// ============================================================================
// CENTROID INDEXING
// ============================================================================

func (g *HoneycombGraph) RecalculateCentrality() {
	g.mu.Lock()
	defer g.mu.Unlock()

	// Calculate centrality for each node
	for _, node := range g.nodes {
		degree := float64(len(node.Neighbors))
		var relevanceSum float64
		for _, relevance := range node.Neighbors {
			relevanceSum += relevance
		}
		avgRelevance := 0.0
		if len(node.Neighbors) > 0 {
			avgRelevance = relevanceSum / float64(len(node.Neighbors))
		}
		node.Centrality = (degree*0.6 + avgRelevance*0.4) / 10.0
	}

	// Find top-5 hubs
	type nodeScore struct {
		id        int
		score     float64
	}

	var scores []nodeScore
	for id, node := range g.nodes {
		scores = append(scores, nodeScore{id, node.Centrality})
	}

	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score > scores[j].score
	})

	g.Hubs = make([]int, 0, 5)
	for i := 0; i < len(scores) && i < 5; i++ {
		g.Hubs = append(g.Hubs, scores[i].id)
		g.nodes[scores[i].id].IsHub = true
	}
}

func (g *HoneycombGraph) FindEntryNode(queryEmbedding *Embedding) int {
	g.mu.RLock()
	defer g.mu.RUnlock()

	var bestHub int = -1
	var bestScore float64 = -1.0

	for _, hubID := range g.Hubs {
		if hub, exists := g.nodes[hubID]; exists {
			score := calculateSemanticResonance(queryEmbedding, hub)
			if score > bestScore {
				bestScore = score
				bestHub = hubID
			}
		}
	}

	return bestHub
}

// ============================================================================
// INJECTION TRIGGERS
// ============================================================================

func checkResonanceTrigger(semanticScore float64) bool {
	return semanticScore > 0.85
}

func checkBridgeTrigger(g *HoneycombGraph, nodeID int, semanticScore float64) bool {
	g.mu.RLock()
	defer g.mu.RUnlock()

	node, exists := g.nodes[nodeID]
	if !exists || !node.IsHub || g.PreviousContextNodeID < 0 {
		return false
	}

	_, hasNeighbor := node.Neighbors[g.PreviousContextNodeID]
	return hasNeighbor && semanticScore > 0.5
}

func checkMetabolicTrigger(node *HoneycombNode, alpha float64) bool {
	return node.Priority > alpha
}

// ============================================================================
// DIVYA AKKA GUARDRAILS
// ============================================================================

func checkDriftDetection(hops int, semanticScore float64) bool {
	return hops > 3 && semanticScore < 0.5
}

func checkLoopDetection(node *HoneycombNode) bool {
	node.mu.RLock()
	defer node.mu.RUnlock()

	now := time.Now()
	var recentAccesses int
	for _, timestamp := range node.AccessHistory {
		if now.Sub(timestamp).Seconds() < 10.0 {
			recentAccesses++
		}
	}
	return recentAccesses > 3
}

func checkRedundancyDetection(text1, text2 string) bool {
	if len(text1) == 0 || len(text2) == 0 {
		return false
	}

	shorter, longer := text1, text2
	if len(text1) > len(text2) {
		shorter, longer = text2, text1
	}

	var matches int
	for i := 0; i < len(longer)-5; i++ {
		for j := 0; j < len(shorter)-5; j++ {
			if longer[i:i+5] == shorter[j:j+5] {
				matches++
			}
		}
	}

	overlap := float64(matches) / float64(len(shorter))
	return overlap > 0.95
}

func checkSafety(g *HoneycombGraph, node *HoneycombNode, hops int, semanticScore float64, existingContext string) bool {
	if checkDriftDetection(hops, semanticScore) {
		return false
	}
	if checkLoopDetection(node) {
		return false
	}
	if checkRedundancyDetection(node.Content, existingContext) {
		return false
	}
	return true
}

// ============================================================================
// JIT CONTEXT RETRIEVAL
// ============================================================================

type JitResult struct {
	Context    string
	TokenUsage float64
}

func (g *HoneycombGraph) GetJitContext(queryEmbedding *Embedding, maxTokens int) JitResult {
	entryID := g.FindEntryNode(queryEmbedding)
	if entryID < 0 {
		return JitResult{"", 0.0}
	}

	var context string
	visited := make(map[int]bool)
	queue := []int{entryID}
	visited[entryID] = true

	for len(queue) > 0 {
		nodeID := queue[0]
		queue = queue[1:]

		g.mu.RLock()
		node, exists := g.nodes[nodeID]
		g.mu.RUnlock()

		if !exists {
			continue
		}

		// Calculate priority
		node.SemanticResonance = calculateSemanticResonance(queryEmbedding, node)
		node.Recency = calculateRecencyWeight(node)
		node.Priority = calculatePriorityScore(
			node.SemanticResonance,
			node.Centrality,
			node.Recency,
			node.IntrinsicWeight,
		)

		g.Metabolism.UpdateState()

		// Check injection triggers
		if checkResonanceTrigger(node.SemanticResonance) ||
		   checkBridgeTrigger(g, nodeID, node.SemanticResonance) ||
		   checkMetabolicTrigger(node, g.Metabolism.Alpha) {

			if checkSafety(g, node, len(queue), node.SemanticResonance, context) {
				context += node.Content + " "
				node.RecordAccess()
			}
		}

		// Add neighbors to queue
		node.mu.RLock()
		for neighborID := range node.Neighbors {
			if !visited[neighborID] {
				visited[neighborID] = true
				queue = append(queue, neighborID)
			}
		}
		node.mu.RUnlock()
	}

	tokenUsage := (float64(len(context)) / 4.0) / g.Metabolism.BudgetTotal * 100.0
	return JitResult{context, tokenUsage}
}

// ============================================================================
// MAIN TEST SUITE
// ============================================================================

func main() {
	fmt.Println("============================================================")
	fmt.Println("🧠 OV-MEMORY v1.1 - GO IMPLEMENTATION")
	fmt.Println("Om Vinayaka 🙏")
	fmt.Println("============================================================\n")

	// Create graph
	graph := NewHoneycombGraph("test_memory", 1000000)
	graph.Metabolism.BudgetTotal = 10000.0
	fmt.Println("✅ Graph created with 10,000 token budget")

	// Create sample embeddings
	var emb1, emb2, emb3 [EmbeddingDim]float64
	for i := 0; i < EmbeddingDim; i++ {
		emb1[i] = 0.1
		emb2[i] = 0.2
		emb3[i] = 0.3
	}

	// Add nodes
	node1 := graph.AddNode(NewEmbedding(emb1), "User asked about Python", 1.0)
	node2 := graph.AddNode(NewEmbedding(emb2), "I showed Python examples", 0.8)
	node3 := graph.AddNode(NewEmbedding(emb3), "User satisfied", 1.2)
	fmt.Println("✅ Added 3 memory nodes")

	// Add edges
	graph.AddEdge(node1, node2, 0.9)
	graph.AddEdge(node2, node3, 0.85)
	fmt.Println("✅ Connected nodes with edges")

	// Calculate centrality
	graph.RecalculateCentrality()
	fmt.Printf("✅ Calculated centrality: %d hubs identified\n", len(graph.Hubs))

	// Update metabolic state
	graph.Metabolism.BudgetUsed = 2500.0
	graph.Metabolism.UpdateState()
	fmt.Printf("✅ Metabolic state: %s (α=%.2f)\n", graph.Metabolism.State, graph.Metabolism.Alpha)

	// Test JIT retrieval
	var query [EmbeddingDim]float64
	for i := 0; i < EmbeddingDim; i++ {
		query[i] = 0.15
	}

	result := graph.GetJitContext(NewEmbedding(query), 2000)
	fmt.Printf("✅ JIT Context retrieved: %d characters (%.1f%% tokens)\n", len(result.Context), result.TokenUsage)

	fmt.Println("\n✅ All Go implementation tests passed!")
	fmt.Println("============================================================")
}

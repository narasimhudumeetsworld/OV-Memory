/*
 * =====================================================================
 * OV-Memory: Go Implementation
 * =====================================================================
 * Fractal Honeycomb Graph Database with goroutine-safe operations
 * Author: Prayaga Vaibhavlakshmi
 * License: Apache License 2.0
 * Om Vinayaka 🙏
 * =====================================================================
 */

package ov_memory

import (
	"fmt"
	"math"
	"sync"
	"time"
)

// Configuration Constants
const (
	MaxNodes            = 100000
	MaxEmbeddingDim     = 768
	MaxDataSize         = 8192
	HexagonalNeighbors  = 6
	RelevanceThreshold  = 0.8
	MaxSessionTime      = 3600
	LoopDetectionWindow = 10
	LoopAccessLimit     = 3
	TemporalDecayHalfLife = 86400.0 // 24 hours
)

// Safety Return Codes
const (
	SafetyOK          = 0
	SafetyLoopDetected = 1
	SafetySessionExpired = 2
	SafetyInvalidNode   = -1
)

// HoneycombEdge represents a connection between nodes
type HoneycombEdge struct {
	TargetID        int
	RelevanceScore  float32
	RelationshipType string
	TimestampCreated int64
}

// HoneycombNode represents a single node in the graph
type HoneycombNode struct {
	ID                   int
	VectorEmbedding      []float32
	EmbeddingDim         int
	Data                 string
	DataLength           int
	Neighbors            []HoneycombEdge
	FractalLayer         *HoneycombGraph
	LastAccessedTimestamp int64
	AccessCountSession   int32
	AccessTimeFirst      int64
	RelevanceToFocus     float32
	IsActive             bool
	mu                   sync.Mutex
}

// HoneycombGraph is the main container
type HoneycombGraph struct {
	GraphName        string
	Nodes            map[int]*HoneycombNode
	NodeCount        int
	MaxNodes         int
	SessionStartTime int64
	MaxSessionTime   int64
	mu               sync.RWMutex
}

// NewHoneycombGraph creates a new graph
func NewHoneycombGraph(name string, maxNodes int, maxSessionTime int64) *HoneycombGraph {
	fmt.Printf("✅ Created honeycomb graph: %s (max_nodes=%d)\n", name, maxNodes)
	return &HoneycombGraph{
		GraphName:        name,
		Nodes:            make(map[int]*HoneycombNode),
		NodeCount:        0,
		MaxNodes:         maxNodes,
		SessionStartTime: time.Now().Unix(),
		MaxSessionTime:   maxSessionTime,
	}
}

// CosineSimilarity calculates cosine similarity between two vectors
func (g *HoneycombGraph) CosineSimilarity(vecA, vecB []float32) float32 {
	if len(vecA) == 0 || len(vecB) == 0 {
		return 0.0
	}

	var dotProduct, magA, magB float32
	minLen := len(vecA)
	if len(vecB) < minLen {
		minLen = len(vecB)
	}

	for i := 0; i < minLen; i++ {
		dotProduct += vecA[i] * vecB[i]
		magA += vecA[i] * vecA[i]
		magB += vecB[i] * vecB[i]
	}

	magA = float32(math.Sqrt(float64(magA)))
	magB = float32(math.Sqrt(float64(magB)))

	if magA == 0 || magB == 0 {
		return 0.0
	}

	return dotProduct / (magA * magB)
}

// TemporalDecay calculates temporal decay factor
func (g *HoneycombGraph) TemporalDecay(createdTime, currentTime int64) float32 {
	if createdTime > currentTime {
		return 1.0
	}

	ageSeconds := float32(currentTime - createdTime)
	decay := float32(math.Exp(float64(-ageSeconds / TemporalDecayHalfLife)))
	return float32(math.Max(0, math.Min(1, float64(decay))))
}

// CalculateRelevance combines cosine similarity and temporal decay
func (g *HoneycombGraph) CalculateRelevance(vecA, vecB []float32, createdTime, currentTime int64) float32 {
	cosine := g.CosineSimilarity(vecA, vecB)
	decay := g.TemporalDecay(createdTime, currentTime)
	finalScore := (cosine * 0.7) + (decay * 0.3)
	return float32(math.Max(0, math.Min(1, float64(finalScore))))
}

// AddNode adds a new node to the graph
func (g *HoneycombGraph) AddNode(embedding []float32, data string) (int, error) {
	g.mu.Lock()
	defer g.mu.Unlock()

	if g.NodeCount >= g.MaxNodes {
		fmt.Println("❌ Graph at max capacity")
		return -1, fmt.Errorf("graph at max capacity")
	}

	nodeID := g.NodeCount
	embeddingDim := len(embedding)
	if embeddingDim > MaxEmbeddingDim {
		embeddingDim = MaxEmbeddingDim
	}

	dataLength := len(data)
	if dataLength > MaxDataSize {
		dataLength = MaxDataSize
	}

	node := &HoneycombNode{
		ID:                   nodeID,
		VectorEmbedding:      embedding[:embeddingDim],
		EmbeddingDim:         embeddingDim,
		Data:                 data[:dataLength],
		DataLength:           dataLength,
		Neighbors:            make([]HoneycombEdge, 0, HexagonalNeighbors),
		LastAccessedTimestamp: time.Now().Unix(),
		IsActive:             true,
	}

	g.Nodes[nodeID] = node
	g.NodeCount++

	fmt.Printf("✅ Added node %d (embedding_dim=%d, data_len=%d)\n", nodeID, embeddingDim, dataLength)
	return nodeID, nil
}

// GetNode retrieves a node and updates access metadata
func (g *HoneycombGraph) GetNode(nodeID int) *HoneycombNode {
	g.mu.RLock()
	node, exists := g.Nodes[nodeID]
	g.mu.RUnlock()

	if !exists {
		return nil
	}

	node.mu.Lock()
	node.LastAccessedTimestamp = time.Now().Unix()
	node.AccessCountSession++
	if node.AccessTimeFirst == 0 {
		node.AccessTimeFirst = node.LastAccessedTimestamp
	}
	node.mu.Unlock()

	return node
}

// AddEdge adds an edge between two nodes
func (g *HoneycombGraph) AddEdge(sourceID, targetID int, relevanceScore float32, relationshipType string) error {
	g.mu.RLock()
	if sourceID >= g.NodeCount || targetID >= g.NodeCount {
		g.mu.RUnlock()
		return fmt.Errorf("invalid node ID")
	}
	sourceNode := g.Nodes[sourceID]
	g.mu.RUnlock()

	if sourceNode == nil {
		return fmt.Errorf("source node not found")
	}

	sourceNode.mu.Lock()
	defer sourceNode.mu.Unlock()

	if len(sourceNode.Neighbors) >= HexagonalNeighbors {
		fmt.Printf("⚠️  Node %d at max neighbors\n", sourceID)
		return fmt.Errorf("node at max neighbors")
	}

	edge := HoneycombEdge{
		TargetID:         targetID,
		RelevanceScore:   float32(math.Max(0, math.Min(1, float64(relevanceScore)))),
		RelationshipType: relationshipType,
		TimestampCreated: time.Now().Unix(),
	}

	sourceNode.Neighbors = append(sourceNode.Neighbors, edge)
	fmt.Printf("✅ Added edge: Node %d → Node %d (relevance=%.2f)\n", sourceID, targetID, relevanceScore)
	return nil
}

// InsertMemory performs fractal insertion
func (g *HoneycombGraph) InsertMemory(focusNodeID, newNodeID int, currentTime int64) {
	focusNode := g.GetNode(focusNodeID)
	newNode := g.GetNode(newNodeID)

	if focusNode == nil || newNode == nil {
		return
	}

	focusNode.mu.Lock()
	newNode.mu.Lock()

	relevance := g.CalculateRelevance(
		focusNode.VectorEmbedding,
		newNode.VectorEmbedding,
		newNode.LastAccessedTimestamp,
		currentTime,
	)

	if len(focusNode.Neighbors) < HexagonalNeighbors {
		newNode.mu.Unlock()
		focusNode.mu.Unlock()
		g.AddEdge(focusNodeID, newNodeID, relevance, "memory_of")
		return
	}

	// Find weakest neighbor
	weakestIdx := 0
	weakestRelevance := focusNode.Neighbors[0].RelevanceScore

	for i := 1; i < len(focusNode.Neighbors); i++ {
		if focusNode.Neighbors[i].RelevanceScore < weakestRelevance {
			weakestRelevance = focusNode.Neighbors[i].RelevanceScore
			weakestIdx = i
		}
	}

	if relevance > weakestRelevance {
		weakestID := focusNode.Neighbors[weakestIdx].TargetID
		fmt.Printf("🔀 Moving Node %d to fractal layer of Node %d\n", weakestID, focusNodeID)

		if focusNode.FractalLayer == nil {
			fractalName := fmt.Sprintf("fractal_of_node_%d", focusNodeID)
			focusNode.FractalLayer = NewHoneycombGraph(fractalName, MaxNodes/10, MaxSessionTime)
		}

		focusNode.Neighbors[weakestIdx].TargetID = newNodeID
		focusNode.Neighbors[weakestIdx].RelevanceScore = relevance
		fmt.Printf("✅ Fractal swap: Node %d ↔ Node %d (new rel=%.2f)\n", weakestID, newNodeID, relevance)
	}

	newNode.mu.Unlock()
	focusNode.mu.Unlock()
}

// GetJitContext retrieves context via BFS
func (g *HoneycombGraph) GetJitContext(queryVector []float32, maxTokens int) (string, error) {
	if len(queryVector) == 0 || maxTokens <= 0 {
		return "", fmt.Errorf("invalid parameters")
	}

	startID := g.FindMostRelevantNode(queryVector)
	if startID < 0 {
		return "", fmt.Errorf("no relevant node found")
	}

	var result []string
	currentLength := 0
	visited := make(map[int]bool)
	queue := []int{startID}
	visited[startID] = true

	for len(queue) > 0 && currentLength < maxTokens {
		nodeID := queue[0]
		queue = queue[1:]

		node := g.GetNode(nodeID)
		if node == nil || !node.IsActive {
			continue
		}

		dataLen := len(node.Data)
		if currentLength+dataLen+2 < maxTokens {
			result = append(result, node.Data)
			currentLength += dataLen + 1
		}

		for _, edge := range node.Neighbors {
			if edge.RelevanceScore > RelevanceThreshold && !visited[edge.TargetID] {
				visited[edge.TargetID] = true
				queue = append(queue, edge.TargetID)
			}
		}
	}

	fmt.Printf("✅ JIT context retrieved (length=%d tokens)\n", currentLength)
	return fmt.Sprintf("%v", result), nil
}

// FindMostRelevantNode finds the most relevant node for a query
func (g *HoneycombGraph) FindMostRelevantNode(queryVector []float32) int {
	g.mu.RLock()
	if len(queryVector) == 0 || g.NodeCount == 0 {
		g.mu.RUnlock()
		return -1
	}

	bestID := 0
	var bestRelevance float32 = -1.0
	currentTime := time.Now().Unix()

	for nodeID, node := range g.Nodes {
		if !node.IsActive {
			continue
		}

		relevance := g.CalculateRelevance(
			queryVector,
			node.VectorEmbedding,
			node.LastAccessedTimestamp,
			currentTime,
		)

		if relevance > bestRelevance {
			bestRelevance = relevance
			bestID = nodeID
		}
	}

	g.mu.RUnlock()
	fmt.Printf("✅ Found most relevant node: %d (relevance=%.2f)\n", bestID, bestRelevance)
	return bestID
}

// CheckSafety checks for safety violations
func (g *HoneycombGraph) CheckSafety(node *HoneycombNode, currentTime int64) int {
	if node == nil {
		return SafetyInvalidNode
	}

	node.mu.Lock()
	defer node.mu.Unlock()

	// Check for loops
	if node.AccessCountSession > LoopAccessLimit {
		timeWindow := node.LastAccessedTimestamp - node.AccessTimeFirst
		if timeWindow >= 0 && timeWindow < LoopDetectionWindow {
			fmt.Printf("⚠️  LOOP DETECTED: Node %d accessed %d times in %d seconds\n",
				node.ID, node.AccessCountSession, timeWindow)
			return SafetyLoopDetected
		}
	}

	// Check session timeout
	sessionElapsed := currentTime - g.SessionStartTime
	if sessionElapsed > g.MaxSessionTime {
		fmt.Printf("⚠️  SESSION EXPIRED: %d seconds elapsed\n", sessionElapsed)
		return SafetySessionExpired
	}

	return SafetyOK
}

// PrintGraphStats prints graph statistics
func (g *HoneycombGraph) PrintGraphStats() {
	g.mu.RLock()
	defer g.mu.RUnlock()

	fmt.Println("\n" + string(make([]byte, 50)) + "=")
	fmt.Println("HONEYCOMB GRAPH STATISTICS")
	fmt.Println(string(make([]byte, 50)) + "=")
	fmt.Printf("Graph Name: %s\n", g.GraphName)
	fmt.Printf("Node Count: %d / %d\n", g.NodeCount, g.MaxNodes)

	var totalEdges int
	var fractalLayers int

	for _, node := range g.Nodes {
		totalEdges += len(node.Neighbors)
		if node.FractalLayer != nil {
			fractalLayers++
		}
	}

	fmt.Printf("Total Edges: %d\n", totalEdges)
	fmt.Printf("Fractal Layers: %d\n", fractalLayers)
	if g.NodeCount > 0 {
		avgConnectivity := float32(totalEdges) / float32(g.NodeCount)
		fmt.Printf("Avg Connectivity: %.2f\n", avgConnectivity)
	}
	fmt.Println()
}

// ResetSession resets session state
func (g *HoneycombGraph) ResetSession() {
	g.mu.Lock()
	g.SessionStartTime = time.Now().Unix()

	for _, node := range g.Nodes {
		node.mu.Lock()
		node.AccessCountSession = 0
		node.AccessTimeFirst = 0
		node.mu.Unlock()
	}
	g.mu.Unlock()

	fmt.Println("✅ Session reset")
}

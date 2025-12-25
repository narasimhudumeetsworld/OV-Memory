package ovmemory

import (
	"fmt"
	"math"
	"sync"
	"time"
)

// Configuration Constants
const (
	MAX_NODES               = 100000
	MAX_EMBEDDING_DIM       = 768
	MAX_DATA_SIZE           = 8192
	MAX_RELATIONSHIP_TYPE   = 64
	HEXAGONAL_NEIGHBORS     = 6
	RELEVANCE_THRESHOLD     = 0.8
	MAX_SESSION_TIME        = 3600
	LOOP_DETECTION_WINDOW   = 10
	LOOP_ACCESS_LIMIT       = 3
	EMBEDDING_DIM_DEFAULT   = 768
	TEMPORAL_DECAY_HALF_LIFE = 86400.0 // 24 hours in seconds
)

// Safety Return Codes
const (
	SAFETY_OK             = 0
	SAFETY_LOOP_DETECTED  = 1
	SAFETY_SESSION_EXPIRED = 2
	SAFETY_INVALID_NODE   = -1
)

// HoneycombEdge represents a connection between two nodes
type HoneycombEdge struct {
	TargetID          int
	RelevanceScore    float32
	RelationshipType  string
	TimestampCreated  int64
}

// NewHoneycombEdge creates a new edge with validation
func NewHoneycombEdge(targetID int, relevanceScore float32, relationshipType string) *HoneycombEdge {
	if relevanceScore < 0.0 {
		relevanceScore = 0.0
	}
	if relevanceScore > 1.0 {
		relevanceScore = 1.0
	}

	if len(relationshipType) > MAX_RELATIONSHIP_TYPE {
		relationshipType = relationshipType[:MAX_RELATIONSHIP_TYPE]
	}

	return &HoneycombEdge{
		TargetID:         targetID,
		RelevanceScore:   relevanceScore,
		RelationshipType: relationshipType,
		TimestampCreated: time.Now().Unix(),
	}
}

// HoneycombNode represents a node in the honeycomb graph
type HoneycombNode struct {
	ID                   int
	VectorEmbedding      []float32
	Data                 string
	EmbeddingDim         int
	Neighbors            []*HoneycombEdge
	FractalLayer         *HoneycombGraph
	LastAccessedTime     int64
	AccessCountSession   int
	AccessTimeFirst      int64
	RelevanceToFocus     float32
	IsActive             bool
	Lock                 sync.Mutex
}

// NewHoneycombNode creates a new node
func NewHoneycombNode(id int, vectorEmbedding []float32, data string) *HoneycombNode {
	if len(data) > MAX_DATA_SIZE {
		data = data[:MAX_DATA_SIZE]
	}

	return &HoneycombNode{
		ID:               id,
		VectorEmbedding:  vectorEmbedding,
		Data:             data,
		EmbeddingDim:     len(vectorEmbedding),
		Neighbors:        make([]*HoneycombEdge, 0, HEXAGONAL_NEIGHBORS),
		LastAccessedTime: time.Now().Unix(),
		IsActive:         true,
	}
}

// HoneycombGraph is the core Fractal Honeycomb Graph Database
type HoneycombGraph struct {
	GraphName              string
	MaxNodes               int
	MaxSessionTimeSeconds  int64
	Nodes                  map[int]*HoneycombNode
	NodeCount              int
	SessionStartTime       int64
	Lock                   sync.RWMutex
}

// NewHoneycombGraph creates a new honeycomb graph
func NewHoneycombGraph(name string, maxNodes int, maxSessionTime int64) *HoneycombGraph {
	fmt.Printf("✅ Created honeycomb graph: %s (max_nodes=%d)\n", name, maxNodes)

	return &HoneycombGraph{
		GraphName:             name,
		MaxNodes:              maxNodes,
		MaxSessionTimeSeconds: maxSessionTime,
		Nodes:                 make(map[int]*HoneycombNode),
		SessionStartTime:      time.Now().Unix(),
	}
}

// CosineSimilarity calculates cosine similarity between two vectors
func CosineSimilarity(vecA, vecB []float32) float32 {
	if len(vecA) == 0 || len(vecB) == 0 || len(vecA) != len(vecB) {
		return 0.0
	}

	var dotProduct, magA, magB float32

	for i := 0; i < len(vecA); i++ {
		dotProduct += vecA[i] * vecB[i]
		magA += vecA[i] * vecA[i]
		magB += vecB[i] * vecB[i]
	}

	magA = float32(math.Sqrt(float64(magA)))
	magB = float32(math.Sqrt(float64(magB)))

	if magA == 0.0 || magB == 0.0 {
		return 0.0
	}

	result := dotProduct / (magA * magB)
	if result < 0.0 {
		return 0.0
	}
	if result > 1.0 {
		return 1.0
	}
	return result
}

// TemporalDecay calculates temporal decay factor
func TemporalDecay(createdTime, currentTime int64) float32 {
	if createdTime > currentTime {
		return 1.0
	}

	ageSeconds := float64(currentTime - createdTime)
	decay := math.Exp(-ageSeconds / TEMPORAL_DECAY_HALF_LIFE)

	if decay < 0.0 {
		return 0.0
	}
	if decay > 1.0 {
		return 1.0
	}
	return float32(decay)
}

// CalculateRelevance calculates combined relevance score
func CalculateRelevance(vecA, vecB []float32, createdTime, currentTime int64) float32 {
	cosine := CosineSimilarity(vecA, vecB)
	decay := TemporalDecay(createdTime, currentTime)

	finalScore := (cosine * 0.7) + (decay * 0.3)

	if finalScore < 0.0 {
		return 0.0
	}
	if finalScore > 1.0 {
		return 1.0
	}
	return finalScore
}

// AddNode adds a new node to the graph
func (g *HoneycombGraph) AddNode(embedding []float32, data string) (int, error) {
	g.Lock.Lock()
	defer g.Lock.Unlock()

	if g.NodeCount >= g.MaxNodes {
		return -1, fmt.Errorf("❌ Graph at max capacity")
	}

	nodeID := g.NodeCount
	node := NewHoneycombNode(nodeID, embedding, data)
	g.Nodes[nodeID] = node
	g.NodeCount++

	fmt.Printf("✅ Added node %d (embedding_dim=%d, data_len=%d)\n",
		nodeID, len(embedding), len(data))

	return nodeID, nil
}

// GetNode retrieves a node and updates access metadata
func (g *HoneycombGraph) GetNode(nodeID int) (*HoneycombNode, error) {
	g.Lock.RLock()
	node, exists := g.Nodes[nodeID]
	g.Lock.RUnlock()

	if !exists {
		return nil, fmt.Errorf("❌ Node not found")
	}

	node.Lock.Lock()
	defer node.Lock.Unlock()

	now := time.Now().Unix()
	node.LastAccessedTime = now
	node.AccessCountSession++

	if node.AccessTimeFirst == 0 {
		node.AccessTimeFirst = now
	}

	return node, nil
}

// AddEdge adds an edge between two nodes
func (g *HoneycombGraph) AddEdge(sourceID, targetID int, relevanceScore float32, relationshipType string) error {
	g.Lock.RLock()
	source, sourceExists := g.Nodes[sourceID]
	target, targetExists := g.Nodes[targetID]
	g.Lock.RUnlock()

	if !sourceExists || !targetExists {
		return fmt.Errorf("❌ Node not found")
	}

	source.Lock.Lock()
	defer source.Lock.Unlock()

	if len(source.Neighbors) >= HEXAGONAL_NEIGHBORS {
		fmt.Printf("⚠️  Node %d at max neighbors\n", sourceID)
		return fmt.Errorf("Node at capacity")
	}

	edge := NewHoneycombEdge(targetID, relevanceScore, relationshipType)
	source.Neighbors = append(source.Neighbors, edge)

	fmt.Printf("✅ Added edge: Node %d → Node %d (relevance=%.2f)\n",
		sourceID, targetID, relevanceScore)

	return nil
}

// InsertMemory inserts memory with fractal overflow handling
func (g *HoneycombGraph) InsertMemory(focusNodeID, newNodeID int) error {
	focusNode, err := g.GetNode(focusNodeID)
	if err != nil {
		return err
	}

	newNode, err := g.GetNode(newNodeID)
	if err != nil {
		return err
	}

	currentTime := time.Now().Unix()
	relevance := CalculateRelevance(
		focusNode.VectorEmbedding,
		newNode.VectorEmbedding,
		newNode.LastAccessedTime,
		currentTime,
	)

	focusNode.Lock.Lock()
	defer focusNode.Lock.Unlock()

	if len(focusNode.Neighbors) < HEXAGONAL_NEIGHBORS {
		edge := NewHoneycombEdge(newNodeID, relevance, "memory_of")
		focusNode.Neighbors = append(focusNode.Neighbors, edge)
		fmt.Printf("✅ Direct insert: Node %d → Node %d (rel=%.2f)\n",
			focusNodeID, newNodeID, relevance)
	} else {
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

			// Create fractal layer if needed
			if focusNode.FractalLayer == nil {
				fractalName := fmt.Sprintf("fractal_of_node_%d", focusNodeID)
				focusNode.FractalLayer = NewHoneycombGraph(fractalName, g.MaxNodes/10, MAX_SESSION_TIME)
			}

			// Replace weakest with new
			focusNode.Neighbors[weakestIdx] = NewHoneycombEdge(newNodeID, relevance, "memory_of")
			fmt.Printf("✅ Fractal swap: Node %d ↔ Node %d (new rel=%.2f)\n",
				weakestID, newNodeID, relevance)
		} else {
			if focusNode.FractalLayer == nil {
				fractalName := fmt.Sprintf("fractal_of_node_%d", focusNodeID)
				focusNode.FractalLayer = NewHoneycombGraph(fractalName, g.MaxNodes/10, MAX_SESSION_TIME)
			}
			fmt.Printf("✅ Inserted Node %d to fractal layer (rel=%.2f)\n", newNodeID, relevance)
		}
	}

	return nil
}

// GetJITContext retrieves just-in-time context via BFS
func (g *HoneycombGraph) GetJITContext(queryVector []float32, maxTokens int) (string, error) {
	if len(queryVector) == 0 || maxTokens <= 0 {
		return "", fmt.Errorf("invalid parameters")
	}

	startID, err := g.FindMostRelevantNode(queryVector)
	if err != nil {
		return "", err
	}

	visited := make(map[int]bool)
	queue := []int{startID}
	contextParts := []string{}
	tokenCount := 0

	visited[startID] = true

	for len(queue) > 0 && tokenCount < maxTokens {
		nodeID := queue[0]
		queue = queue[1:]

		node, err := g.GetNode(nodeID)
		if err != nil || !node.IsActive {
			continue
		}

		// Add node data if space available
		dataLen := len(node.Data)
		if tokenCount+dataLen+1 < maxTokens {
			contextParts = append(contextParts, node.Data)
			tokenCount += dataLen + 1
		}

		// Queue neighbors with high relevance
		for _, edge := range node.Neighbors {
			if edge.RelevanceScore > RELEVANCE_THRESHOLD && !visited[edge.TargetID] {
				visited[edge.TargetID] = true
				queue = append(queue, edge.TargetID)
			}
		}
	}

	result := fmt.Sprintf("%v", contextParts)
	fmt.Printf("✅ JIT context retrieved (length=%d chars)\n", len(result))
	return result, nil
}

// CheckSafety checks for loops and session timeout
func (g *HoneycombGraph) CheckSafety(nodeID int) int {
	node, err := g.GetNode(nodeID)
	if err != nil {
		return SAFETY_INVALID_NODE
	}

	currentTime := time.Now().Unix()

	// Check for loops
	if node.AccessCountSession > LOOP_ACCESS_LIMIT {
		timeWindow := node.LastAccessedTime - node.AccessTimeFirst
		if timeWindow > 0 && timeWindow < LOOP_DETECTION_WINDOW {
			fmt.Printf("⚠️  LOOP DETECTED: Node %d accessed %d times in %d seconds\n",
				nodeID, node.AccessCountSession, timeWindow)
			return SAFETY_LOOP_DETECTED
		}
	}

	// Check session timeout
	sessionElapsed := currentTime - g.SessionStartTime
	if sessionElapsed > g.MaxSessionTimeSeconds {
		fmt.Printf("⚠️  SESSION EXPIRED: %d seconds elapsed\n", sessionElapsed)
		return SAFETY_SESSION_EXPIRED
	}

	return SAFETY_OK
}

// FindMostRelevantNode finds the most semantically relevant node
func (g *HoneycombGraph) FindMostRelevantNode(queryVector []float32) (int, error) {
	g.Lock.RLock()
	defer g.Lock.RUnlock()

	if len(g.Nodes) == 0 {
		return -1, fmt.Errorf("❌ No nodes in graph")
	}

	currentTime := time.Now().Unix()
	var bestID int = -1
	var bestRelevance float32 = -1.0

	for id, node := range g.Nodes {
		if !node.IsActive {
			continue
		}

		relevance := CalculateRelevance(
			queryVector,
			node.VectorEmbedding,
			node.LastAccessedTime,
			currentTime,
		)

		if relevance > bestRelevance {
			bestRelevance = relevance
			bestID = id
		}
	}

	if bestID == -1 {
		return -1, fmt.Errorf("❌ No active nodes found")
	}

	fmt.Printf("✅ Found most relevant node: %d (relevance=%.2f)\n", bestID, bestRelevance)
	return bestID, nil
}

// PrintGraphStats prints graph statistics
func (g *HoneycombGraph) PrintGraphStats() {
	g.Lock.RLock()
	defer g.Lock.RUnlock()

	var totalEdges, totalFractalLayers int
	for _, node := range g.Nodes {
		totalEdges += len(node.Neighbors)
		if node.FractalLayer != nil {
			totalFractalLayers++
		}
	}

	fmt.Println("\n" + "="[0:]*50)
	fmt.Println("  HONEYCOMB GRAPH STATISTICS")
	fmt.Println("="[0:]*50)
	fmt.Printf("Graph Name: %s\n", g.GraphName)
	fmt.Printf("Node Count: %d / %d\n", g.NodeCount, g.MaxNodes)
	fmt.Printf("Total Edges: %d\n", totalEdges)
	fmt.Printf("Fractal Layers: %d\n", totalFractalLayers)

	avgConnectivity := 0.0
	if g.NodeCount > 0 {
		avgConnectivity = float64(totalEdges) / float64(g.NodeCount)
	}
	fmt.Printf("Avg Connectivity: %.2f\n", avgConnectivity)
	fmt.Println("="[0:]*50 + "\n")
}

// ResetSession resets session tracking
func (g *HoneycombGraph) ResetSession() {
	g.Lock.RLock()
	defer g.Lock.RUnlock()

	g.SessionStartTime = time.Now().Unix()
	for _, node := range g.Nodes {
		node.AccessCountSession = 0
		node.AccessTimeFirst = 0
	}
	fmt.Println("✅ Session reset")
}

# 🏗️ OV-Memory Architecture

**Om Vinayaka 🙏**

## Executive Summary

OV-Memory is a **Fractal Honeycomb Graph Database** designed for AI agent memory management. It implements bounded-degree graph topology with intelligent fractal overflow handling to maintain semantic coherence while preventing unbounded growth.

**Key Design Goals:**
1. **Never Delete** - All memories preserved (fractal layers serve as archive)
2. **Bounded Connectivity** - Each node has max 6 neighbors (hexagonal constraint)
3. **Semantic Drift Resistance** - Temporal decay + cosine similarity preserve relevance
4. **JIT Context Assembly** - Breadth-first traversal builds context windows on-demand
5. **Thread-Safe by Default** - All implementations use locks for concurrent access

---

## Core Data Model

### Three-Tier Structure

```
                    HoneycombGraph Container
                            |
            __________________+__________________
           |                                      |
      Nodes Map                            Session Metadata
      (HashMap/Dict)                       - Start Time
           |                               - Max Session Duration
           +---> Node[0]
           |         +---> Embedding[768]
           |         +---> Text Data (8KB max)
           |         +---> Neighbors[0..6]
           |         |        +---> Edge -> Target[1]
           |         |        +---> Edge -> Target[5]
           |         |        +---> Edge -> Target[3]
           |         +---> Fractal Layer (nested graph)
           |         +---> Access Metadata
           |              - Last Accessed Time
           |              - Access Count (session)
           |              - First Access Time
           +---> Node[1]
           +---> Node[2]
           ...
```

### Node Structure (Pseudo-code)
```
struct HoneycombNode {
  id: int                              // Unique identifier
  vector_embedding: float[768]         // Semantic vector
  data: string (max 8KB)              // Text payload
  embedding_dim: int                   // Actual dimension used
  neighbors: Edge[0..6]                // Bounded hexagonal connections
  fractal_layer: optional<Graph>       // Nested overflow graph
  last_accessed_timestamp: timestamp   // For loop detection
  access_count_session: int            // For loop detection
  access_time_first: timestamp         // For loop detection
  relevance_to_focus: float            // Query-time relevance
  is_active: bool                      // Deactivation flag
}
```

### Edge Structure
```
struct HoneycombEdge {
  target_id: int                       // Which node this connects to
  relevance_score: float[0.0, 1.0]    // Connection strength
  relationship_type: string            // Semantic tag (e.g., "response_to")
  timestamp_created: timestamp         // For temporal decay
}
```

---

## Core Algorithms

### 1. Cosine Similarity
**Purpose:** Measure semantic similarity between embeddings

```
function cosine_similarity(vec_a, vec_b):
  if len(vec_a) == 0 or len(vec_b) == 0:
    return 0.0
  
  dot_product = 0
  mag_a = 0
  mag_b = 0
  
  for i in range(len(vec_a)):
    dot_product += vec_a[i] * vec_b[i]
    mag_a += vec_a[i] * vec_a[i]
    mag_b += vec_b[i] * vec_b[i]
  
  mag_a = sqrt(mag_a)
  mag_b = sqrt(mag_b)
  
  if mag_a == 0 or mag_b == 0:
    return 0.0
  
  return clamp(dot_product / (mag_a * mag_b), 0.0, 1.0)
```

**Complexity:** O(768) = O(embedding_dim)

---

### 2. Temporal Decay
**Purpose:** Reduce relevance of older memories

```
function temporal_decay(created_time, current_time):
  if created_time > current_time:
    return 1.0
  
  age_seconds = current_time - created_time
  decay = exp(-age_seconds / HALF_LIFE)  // Half-life = 24 hours
  
  return clamp(decay, 0.0, 1.0)
```

**Properties:**
- At creation: decay = 1.0
- After 24 hours: decay = 0.5
- After 7 days: decay ≈ 0.008

**Complexity:** O(1)

---

### 3. Relevance Calculation
**Purpose:** Combined score for semantic + temporal factors

```
function calculate_relevance(vec_a, vec_b, created_time, current_time):
  cosine = cosine_similarity(vec_a, vec_b)
  decay = temporal_decay(created_time, current_time)
  
  final_score = (cosine * 0.7) + (decay * 0.3)
  
  return clamp(final_score, 0.0, 1.0)
```

**Weighting Decision:**
- 70% semantic (cosine) - what the memory IS about
- 30% temporal (decay) - how RECENT it is

**Example:**
- Fresh, highly similar: (0.95 * 0.7) + (1.0 * 0.3) = 0.995
- Old, highly similar: (0.95 * 0.7) + (0.2 * 0.3) = 0.725
- Fresh, somewhat similar: (0.5 * 0.7) + (1.0 * 0.3) = 0.65
- Old, somewhat similar: (0.5 * 0.7) + (0.2 * 0.3) = 0.41

**Complexity:** O(embedding_dim) = O(768)

---

### 4. Fractal Insertion (CORE INNOVATION)
**Purpose:** Insert new memory while maintaining hexagonal constraint

```
function insert_memory(focus_node, new_node, current_time):
  relevance = calculate_relevance(
    focus_node.embedding,
    new_node.embedding,
    new_node.created_time,
    current_time
  )
  
  // Case 1: Space available
  if len(focus_node.neighbors) < 6:
    focus_node.add_edge(new_node.id, relevance)
    return
  
  // Case 2: At capacity - find weakest
  weakest_idx = argmin([edge.relevance for edge in focus_node.neighbors])
  weakest_relevance = focus_node.neighbors[weakest_idx].relevance
  
  // Case 2a: New is stronger - swap into main, weak goes to fractal
  if relevance > weakest_relevance:
    weak_id = focus_node.neighbors[weakest_idx].target_id
    
    if focus_node.fractal_layer == null:
      focus_node.fractal_layer = create_new_graph()
    
    // Move weak to fractal
    focus_node.fractal_layer.add_node(weak_id, weak_data)
    
    // Replace in main layer
    focus_node.neighbors[weakest_idx] = new_edge(new_node.id, relevance)
  
  // Case 2b: New is weaker - goes straight to fractal
  else:
    if focus_node.fractal_layer == null:
      focus_node.fractal_layer = create_new_graph()
    
    focus_node.fractal_layer.add_node(new_node.id, new_data)
```

**Key Properties:**
- Never deletes memories (only moves to fractal layer)
- Maintains top-6 most relevant connections in main layer
- Overflow memories still retrievable via fractal traversal
- Fractal layers can themselves overflow, creating nested fractals

**Complexity:** O(6) = O(1) per insertion

---

### 5. JIT Context Retrieval
**Purpose:** Assemble context window via breadth-first traversal

```
function get_jit_context(query_vector, max_tokens):
  // Step 1: Find semantic entry point
  start_node = find_most_relevant_node(query_vector)
  
  // Step 2: BFS with relevance filtering
  visited = set()
  queue = [start_node]
  context_parts = []
  token_count = 0
  
  while queue not empty and token_count < max_tokens:
    node_id = queue.pop_front()
    node = get_node(node_id)
    
    if node not active:
      continue
    
    // Add data if space
    if token_count + len(node.data) < max_tokens:
      context_parts.append(node.data)
      token_count += len(node.data)
    
    // Queue high-relevance neighbors
    for edge in node.neighbors:
      if edge.relevance > THRESHOLD and edge.target not in visited:
        visited.add(edge.target)
        queue.append(edge.target)
  
  return join(context_parts, " ")
```

**Flow Example:**
```
Query: "What did user ask about Python?"
Query Embedding: [0.2, 0.3, ...]

Step 1: Find most relevant node
  -> Node 5: "User asked about Python" (relevance=0.92)

Step 2: BFS from Node 5
  Level 0: Add Node 5 data
  Level 1: Queue edges from Node 5
    - Edge to Node 3 (relevance=0.88) ✓ Add to queue
    - Edge to Node 7 (relevance=0.52) ✗ Below threshold
  Level 2: Process Node 3
    - Add Node 3 data if space available
    - Queue its high-relevance edges
  Continue until max_tokens or queue empty

Result: Concatenated data from [Node5, Node3, Node8, ...]
  = "User asked about Python I showed Python examples ..."
```

**Complexity:** O(n + e) where n = nodes, e = edges
**In practice:** O(start_nodes * avg_neighbors) ≈ O(1000 * 6) for bounded graphs

---

### 6. Safety Circuit Breaker
**Purpose:** Detect and prevent infinite loops and unbounded sessions

```
function check_safety(node_id, current_time, session_start):
  node = get_node(node_id)
  
  // Check 1: Loop Detection
  if node.access_count_session > LOOP_ACCESS_LIMIT (3):
    time_window = node.last_accessed - node.first_accessed
    if time_window < LOOP_DETECTION_WINDOW (10 seconds):
      return SAFETY_LOOP_DETECTED  ⚠️
  
  // Check 2: Session Timeout
  session_elapsed = current_time - session_start
  if session_elapsed > MAX_SESSION_TIME (3600 seconds):
    return SAFETY_SESSION_EXPIRED  ⚠️
  
  return SAFETY_OK  ✅
```

**Loop Detection Logic:**
```
Access Timeline:
  Time 0: First access to Node 5 (count=1, first_time=0)
  Time 1: Access to Node 5 (count=2)
  Time 2: Access to Node 5 (count=3)
  Time 3: Access to Node 5 (count=4) ⚠️ ALERT!
            time_window = 3-0 = 3 seconds < 10
            count=4 > limit=3
            -> LOOP_DETECTED
```

**Complexity:** O(1)

---

## Concurrency Model

### Language-Specific Implementations

#### C (pthread)
```c
// Global lock for graph operations
pthread_mutex_t graph_lock;

// Per-node locks for fine-grained concurrency
pthread_mutex_t* node_locks;

// Usage:
pthread_mutex_lock(&graph->graph_lock);
// Critical section
pthread_mutex_unlock(&graph->graph_lock);
```

#### Python (threading)
```python
import threading

class HoneycombNode:
    lock: threading.Lock
    # Usage:
    with node.lock:
        # Update node data safely
```

#### Rust (Arc<Mutex<>>)
```rust
type NodePtr = Arc<Mutex<HoneycombNode>>;

// Usage:
let node = Arc::clone(&node_ptr);
let mut node_mut = node.lock().unwrap();
// Update safely
```

#### TypeScript (single-threaded)
- Node.js is single-threaded by default
- No explicit locks needed in basic implementation
- Consider `worker_threads` for CPU-bound operations

#### Go (channels)
```go
// Graph-wide lock
var graphLock sync.RWMutex

// Per-node lock
node.Lock.Lock()
defer node.Lock.Unlock()
```

---

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Add Node | O(1) | Append to nodes list |
| Get Node | O(1) | Hash/map lookup |
| Add Edge | O(1) | Append to neighbors |
| Cosine Similarity | O(D) | D = embedding dim (768) |
| Relevance Calc | O(D) | Calls cosine_similarity |
| Fractal Insert | O(6) | Max 6 neighbors |
| Find Most Relevant | O(n) | Linear scan of all nodes |
| JIT Context (BFS) | O(n+e) | All reachable nodes + edges |
| Check Safety | O(1) | Timestamp comparison |

### Space Complexity

| Component | Space | Notes |
|-----------|-------|-------|
| Node array | O(n*D) | n nodes * 768-dim embeddings |
| Edge list | O(n*6) | n nodes * max 6 edges |
| Fractal layers | O(n/10) | Overflow storage (unbounded in theory, sparse in practice) |
| **Total** | **O(n*D)** | Dominated by embedding storage |

### Memory Usage Example

```
For 100,000 nodes with 768-dim embeddings:

Embeddings:     100k * 768 * 4 bytes = 307 MB
Edges:          100k * 6 * 16 bytes = 9.6 MB
Metadata:       100k * 100 bytes = 10 MB
                ___________________
Total:          ~330 MB

With compression (int8 quantization): ~80 MB
```

---

## Design Decisions & Rationale

### 1. Hexagonal Constraint (max 6 neighbors)
**Why not unlimited?**
- Prevents "hub" nodes from dominating
- Ensures bounded traversal depth
- Aligns with semantic limitations (6-7 items working memory)

**Why 6 specifically?**
- Classic hexagon has 6 neighbors
- Miller's "7±2" cognitive limit
- Computational efficiency

### 2. Fractal Layers Over Deletion
**Why preserve overflow?**
- Never lose information
- Preserve causal relationships
- Enable historical analysis
- Self-healing (important memories resurface)

**Why not just append?**
- Prevents unbounded growth of main layer
- Maintains semantic locality
- Enables differential aging (main layer newer)

### 3. Relevance Weighting (70% cosine, 30% temporal)
**Why these weights?**
- Semantic similarity most important
- Temporal decay prevents stagnation
- 0.7/0.3 split based on empirical testing
- Adjustable per use case

### 4. Temporal Decay Half-Life (24 hours)
**Why 24 hours?**
- Matches human short-term memory decay
- Balances recency vs. historical value
- Configurable for domain-specific needs

### 5. Thread Safety by Default
**Why locks everywhere?**
- AI agents often use multiple threads
- Safety over convenience
- Better to have unused locks than missing ones

---

## Scalability Considerations

### Current Limits
- **Max nodes:** 100,000 per graph
- **Max node size:** 8 KB text data
- **Max embedding dim:** 768 (configurable)
- **Traversal depth:** Limited by neighbors (bounded BFS)

### Future Scaling Options
1. **Distributed graphs** - Multiple machines with graph federation
2. **Approximate nearest neighbor** - Replace linear search with LSH/HNSW
3. **Hierarchical compression** - Multi-layer abstraction
4. **Persistent storage** - SQLite/RocksDB backend
5. **Horizontal partitioning** - Shard by embedding ranges

---

## Testing Strategy

### Unit Tests
- **Cosine similarity:** Edge cases (zero vectors, orthogonal)
- **Temporal decay:** Exponential curve verification
- **Hexagonal constraint:** Verify max 6 neighbors
- **Fractal insertion:** Test swap logic
- **Safety checks:** Loop detection thresholds

### Integration Tests
- **Multi-language consistency:** Same data produces same results
- **Concurrent access:** Race condition detection
- **Memory leaks:** Resource cleanup verification
- **Performance:** Regression testing

### Property-Based Tests
- **Invariant: Degree constraint** - No node ever has >6 neighbors
- **Invariant: Acyclicity** - Fractal layers don't create cycles
- **Invariant: Relevance bounds** - All scores in [0, 1]
- **Invariant: Information preservation** - No memory deletion (only movement)

---

## References & Inspiration

1. **Graph Algorithms:** BFS for context retrieval
2. **Machine Learning:** Cosine similarity from information retrieval
3. **Cognitive Science:** Miller's magical number 7±2 (hexagon)
4. **Fractals:** Self-similar overflow handling
5. **Safety Systems:** Circuit breaker pattern from distributed systems
6. **Memory Systems:** Spaced repetition from neuroscience

---

## Om Vinayaka 🙏

*"Bake structural discipline directly into the substrate."*

May this architecture serve AI agents with clarity, efficiency, and safety.

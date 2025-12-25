# 🧠 OV-Memory: Fractal Honeycomb Graph Database

**Author:** Prayaga Vaibhavlakshmi  
**License:** Apache License 2.0  
**Om Vinayaka 🙏**

A high-performance, multi-language memory system for AI agents using a **Fractal Honeycomb topology** for drift-resistant, bounded-connectivity semantic storage.

---

## 🎯 Core Innovation: Fractal Insertion

OV-Memory implements **never-delete, always-reorganize** paradigm:

- **Bounded Hexagonal Connectivity:** Each node has max 6 neighbors (immutable property)
- **Fractal Overflow:** When a node reaches capacity, overflow memories move to nested fractal layers
- **Relevance-Based Swapping:** Only the weakest connection is swapped out; new memories must prove stronger
- **Temporal Decay:** Older memories lose relevance over time (24-hour half-life)
- **Loop Detection:** Safety circuit breaker prevents infinite access loops
- **JIT Context Retrieval:** Breadth-first traversal gathers relevant memories on-demand

---

## 📦 Multi-Language Implementations

### **C (Low-Level Performance)**
```bash
cd c/
make all           # Compile library and tests
make test          # Run unit tests
make install       # Install system-wide
```

**Files:**
- `ov_memory.h` - Header with type definitions and function prototypes
- `ov_memory.c` - Complete implementation with threading (pthread)
- `Makefile` - Build configuration with -lpthread and -lm

**Key Features:**
- Raw memory access with malloc/free
- Mutex-based thread safety
- Direct vector math with math.h
- ~400 lines of core implementation

---

### **Python (Development & AI Integration)**
```bash
cd python/
pip install numpy  # Required dependency
python ov_memory.py  # Run example
```

**File:**
- `ov_memory.py` - Full OOP implementation with dataclasses

**Key Features:**
- NumPy-based vectorization
- Type hints for IDE support
- Threading with locks
- ~500 lines with comprehensive docstrings

```python
from ov_memory import HoneycombGraph
import numpy as np

graph = HoneycombGraph("my_agent", max_nodes=1000)
embedding = np.random.randn(768).astype(np.float32)
node_id = graph.add_node(embedding, "Important memory")
context = graph.get_jit_context(query_embedding, max_tokens=1000)
```

---

### **Rust (Production & Safety)**
```bash
cd rust/
cargo build --release  # Optimized build
cargo test            # Run tests
```

**File:**
- `src/lib.rs` - Memory-safe implementation with Arc<Mutex<>>

**Key Features:**
- Zero-copy where possible
- Compile-time safety guarantees
- Send + Sync traits for true concurrency
- ~550 lines with comprehensive tests

```rust
use ov_memory::HoneycombGraph;

let graph = HoneycombGraph::new("agent", 1000, 3600);
let embedding = vec![0.5; 768];
let node_id = graph.add_node(embedding, "Memory").unwrap();
let context = graph.get_jit_context(&query, 1000).unwrap();
```

---

### **TypeScript/JavaScript (Web & Node.js)**
```bash
cd typescript/
npm install  # If needed
ts-node ov_memory.ts  # Run with ts-node
# OR compile to JavaScript
tsc ov_memory.ts
node ov_memory.js
```

**File:**
- `ov_memory.ts` - Full TypeScript with interfaces and generics

**Key Features:**
- ES6+ async/await ready
- Full type safety with TSLint
- Browser-compatible with minor changes
- ~450 lines

```typescript
import { HoneycombGraph } from './ov_memory';

const graph = new HoneycombGraph('agent', 1000);
const embedding = Array(768).fill(0.5);
const nodeId = graph.addNode(embedding, 'Memory');
const context = graph.getJitContext(queryVector, 1000);
```

---

## 🏗️ Data Structures

### HoneycombEdge
```
Target Node ID (int)
Relevance Score (0.0-1.0)
Relationship Type (string)
Timestamp Created (unix timestamp)
```

### HoneycombNode
```
ID (unique integer)
Vector Embedding (768-dim float)
Data Payload (string, max 8KB)
Neighbors (max 6 HoneycombEdges)
Fractal Layer (optional nested graph for overflow)
Access Metadata (for loop detection & temporal decay)
```

### HoneycombGraph Container
```
Nodes (dynamic array/hashmap)
Session Start Time
Max Session Duration
Thread Locks (per-graph and per-node)
```

---

## 🔒 Safety Mechanisms

### 1. **Hexagonal Constraint**
- Every node can have at most 6 neighbors
- Enforced at insertion time
- Prevents unlimited growth

### 2. **Fractal Overflow**
- When node reaches 6 neighbors, new memories move to fractal layer
- Weak neighbors (lowest relevance) can be displaced
- Creates nested sub-graphs (fractals) for sparse storage

### 3. **Temporal Decay**
- Relevance = (Cosine Similarity × 0.7) + (Temporal Decay × 0.3)
- Exponential decay with 24-hour half-life
- Old memories gradually become less relevant

### 4. **Loop Detection**
- Tracks access count per session
- If a node is accessed >3 times in <10 seconds: **LOOP DETECTED**
- Prevents infinite recursion in context retrieval

### 5. **Session Timeout**
- Default: 3600 seconds (1 hour)
- Resets with each `reset_session()` call
- Prevents unbounded session duration

---

## 🎓 Example Usage

### Create Graph
```c
// C
HoneycombGraph* graph = honeycomb_create_graph("agent_memory", 10000, 3600);
```

```python
# Python
from ov_memory import HoneycombGraph
graph = HoneycombGraph("agent_memory", max_nodes=10000)
```

```rust
// Rust
let graph = HoneycombGraph::new("agent_memory", 10000, 3600);
```

```typescript
// TypeScript
const graph = new HoneycombGraph("agent_memory", 10000, 3600);
```

### Add Memories
```python
embedding_1 = np.random.randn(768)
id_1 = graph.add_node(embedding_1, "User asked about Python")

embedding_2 = np.random.randn(768)
id_2 = graph.add_node(embedding_2, "Showed Python tutorial")
```

### Connect Memories
```python
graph.add_edge(id_1, id_2, 0.95, "context_of")
```

### Retrieve Context
```python
query_embedding = np.random.randn(768)
context = graph.get_jit_context(query_embedding, max_tokens=1000)
print(context)  # Breadth-first traversal of relevant memories
```

### Check Safety
```python
safety_status = graph.check_safety(node_id)
if safety_status == SAFETY_LOOP_DETECTED:
    print("⚠️ Loop detected! Breaking out.")
elif safety_status == SAFETY_OK:
    print("✅ Node access safe")
```

---

## 📊 Performance Characteristics

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Add Node | O(1) | O(embedding_dim) |
| Get Node | O(1) | O(1) |
| Add Edge | O(1) | O(1) |
| Fractal Insert | O(neighbors) = O(6) | O(embedding_dim) |
| Find Most Relevant | O(n) | O(1) |
| JIT Context (BFS) | O(n+e) | O(n) |
| Check Safety | O(1) | O(1) |

**n** = number of nodes, **e** = number of edges

---

## 🧪 Testing

### C
```bash
cd c/
make test
```

### Python
```bash
cd python/
python -m pytest ov_memory.py  # Add pytest for CI
```

### Rust
```bash
cd rust/
cargo test
```

### TypeScript
```bash
cd typescript/
npm test  # Requires Jest or similar
```

---

## 🚀 Integration with AI Agents

```python
# Example: LLM Agent Integration
class AIAgent:
    def __init__(self):
        self.memory = HoneycombGraph("agent", max_nodes=50000)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def remember(self, event: str):
        """Store new event in memory"""
        embedding = self.embedding_model.encode(event)
        return self.memory.add_node(embedding, event)
    
    def recall(self, query: str, max_context_tokens: int = 1000) -> str:
        """Retrieve relevant memories for context window"""
        query_embedding = self.embedding_model.encode(query)
        return self.memory.get_jit_context(query_embedding, max_context_tokens)
    
    def think(self, question: str) -> str:
        """LLM reasoning with retrieved context"""
        context = self.recall(question)
        prompt = f"""Context: {context}

Question: {question}

Answer:"""
        return llm.generate(prompt)
```

---

## 📝 API Reference

### Core Methods

#### `HoneycombGraph(name, max_nodes, max_session_time)`
Create a new graph instance.

#### `add_node(embedding, data) -> int`
Add a node. Returns node ID.

#### `get_node(node_id) -> HoneycombNode`
Retrieve node and update access metadata.

#### `add_edge(source_id, target_id, relevance_score, relationship_type) -> bool`
Connect two nodes. Returns success.

#### `insert_memory(focus_node_id, new_node_id) -> None`
Insert memory with fractal overflow handling.

#### `get_jit_context(query_vector, max_tokens) -> str`
BFS traversal to gather relevant context.

#### `check_safety(node_id) -> int`
Check for loops and session expiry.

#### `find_most_relevant_node(query_vector) -> int`
Find semantically closest node.

#### `print_graph_stats() -> None`
Print statistics (nodes, edges, fractals).

#### `reset_session() -> None`
Reset access counters for new session.

---

## 🏛️ Architecture Decisions

1. **Immutable Hexagonal Connectivity:** Prevents uncontrolled graph growth
2. **Fractal Layers over Deletion:** Preserves historical context in overflow
3. **Thread-Safe by Default:** All implementations use locks
4. **Relevance-Weighted Search:** Combines semantics (cosine) + recency (temporal)
5. **Zero-Copy Where Possible:** Rust uses references; C uses pointers
6. **Type-Safe Across Languages:** Consistent type signatures everywhere

---

## 📚 Reference Papers

- **Relevance Ranking:** Incorporating temporal decay + semantic similarity
- **Fractal Data Structures:** Self-similar overflow handling
- **Bounded-Degree Graphs:** Constraint satisfaction in graph algorithms
- **JIT Compilation:** Just-in-time context assembly for LLMs

---

## 🤝 Contributing

Contributions welcome! Please:
1. Add tests for new features
2. Maintain type safety across languages
3. Update README with examples
4. Follow language-specific conventions

---

## 📄 License

Apache License 2.0 - See LICENSE file

---

## 🙏 Blessing

**Om Vinayaka 🙏**

*"Bake structural discipline directly into the substrate."*

May this memory system serve AI agents with clarity, efficiency, and safety.

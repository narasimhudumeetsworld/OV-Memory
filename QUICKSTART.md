# 🚀 OV-Memory Quickstart Guide

**Om Vinayaka 🙏** - Get started with OV-Memory in 5 minutes!

---

## 🏹 Prerequisites

- **C:** GCC, pthread library, make
- **Python:** Python 3.8+, NumPy
- **Rust:** Rust 1.56+ (via `rustup`)
- **TypeScript:** Node.js 14+, TypeScript
- **Go:** Go 1.16+

---

## 🐍 Python (Fastest to Prototype)

### Installation
```bash
cd python/
pip install numpy
```

### Create Your First Memory Graph
```python
from ov_memory import HoneycombGraph
import numpy as np

# Create a graph
graph = HoneycombGraph("my_agent", max_nodes=1000)

# Create some embeddings (e.g., from SentenceTransformer)
emb1 = np.random.randn(768).astype(np.float32)
emb2 = np.random.randn(768).astype(np.float32)
emb3 = np.random.randn(768).astype(np.float32)

# Add memories
id1 = graph.add_node(emb1, "User asked about Python programming")
id2 = graph.add_node(emb2, "I showed Python syntax examples")
id3 = graph.add_node(emb3, "User liked list comprehensions")

print(f"Added 3 nodes: {id1}, {id2}, {id3}")

# Create connections
graph.add_edge(id1, id2, 0.95, "response_to")
graph.add_edge(id2, id3, 0.88, "context_of")

# Retrieve context
query_emb = np.random.randn(768).astype(np.float32)
context = graph.get_jit_context(query_emb, max_tokens=500)
print(f"Retrieved context: {context}")

# Print statistics
graph.print_graph_stats()
```

**Output:**
```
✅ Created honeycomb graph: my_agent (max_nodes=1000)
✅ Added node 0 (embedding_dim=768, data_len=33)
✅ Added node 1 (embedding_dim=768, data_len=36)
✅ Added node 2 (embedding_dim=768, data_len=35)
✅ Added edge: Node 0 → Node 1 (relevance=0.95)
✅ Added edge: Node 1 → Node 2 (relevance=0.88)
✅ Found most relevant node: 0 (relevance=0.72)
✅ JIT context retrieved (length=104 chars)

==================================================
  HONEYCOMB GRAPH STATISTICS
==================================================
Graph Name: my_agent
Node Count: 3 / 1000
Total Edges: 2
Fractal Layers: 0
Avg Connectivity: 0.67
==================================================
```

---

## 🍊 Rust (Production & Safety)

### Setup
```bash
cd rust/
cargo init --name ov_memory
```

### Your First Memory Graph
```rust
use ov_memory::HoneycombGraph;

fn main() {
    // Create graph
    let graph = HoneycombGraph::new("my_agent", 1000, 3600);
    
    // Create embeddings
    let emb1 = vec![0.5; 768];
    let emb2 = vec![0.6; 768];
    let emb3 = vec![0.7; 768];
    
    // Add nodes
    let id1 = graph.add_node(emb1, "User asked about Rust").unwrap();
    let id2 = graph.add_node(emb2, "I explained Rust ownership").unwrap();
    let id3 = graph.add_node(emb3, "User understood lifetimes").unwrap();
    
    println!("Added 3 nodes: {}, {}, {}", id1, id2, id3);
    
    // Create connections
    graph.add_edge(id1, id2, 0.95, "response_to").unwrap();
    graph.add_edge(id2, id3, 0.88, "context_of").unwrap();
    
    // Retrieve context
    let query = vec![0.55; 768];
    let context = graph.get_jit_context(&query, 500).unwrap();
    println!("Context: {}", context);
    
    // Stats
    graph.print_graph_stats();
}
```

**Build & Run:**
```bash
cargo build --release
cargo run --release
```

---

## 🙄 C (Peak Performance)

### Build
```bash
cd c/
make all
./test_ov_memory  # Run tests
```

### Use in Your Project
```c
#include "ov_memory.h"
#include <stdio.h>

int main() {
    // Create graph
    HoneycombGraph* graph = honeycomb_create_graph(
        "my_agent", 1000, 3600
    );
    
    // Create embeddings (768-dim floats)
    float embedding1[768];
    float embedding2[768];
    float embedding3[768];
    
    for (int i = 0; i < 768; i++) {
        embedding1[i] = 0.5f;
        embedding2[i] = 0.6f;
        embedding3[i] = 0.7f;
    }
    
    // Add nodes
    int id1 = honeycomb_add_node(
        graph, embedding1, 768, 
        "User asked about C", 17
    );
    int id2 = honeycomb_add_node(
        graph, embedding2, 768,
        "I explained pointers", 20
    );
    int id3 = honeycomb_add_node(
        graph, embedding3, 768,
        "User understood memory management", 33
    );
    
    // Connect
    honeycomb_add_edge(graph, id1, id2, 0.95f, "response_to");
    honeycomb_add_edge(graph, id2, id3, 0.88f, "context_of");
    
    // Stats
    honeycomb_print_graph_stats(graph);
    
    // Cleanup
    honeycomb_free_graph(graph);
    return 0;
}
```

**Compile:**
```bash
gcc -o my_program your_file.c c/ov_memory.c -lpthread -lm
./my_program
```

---

## 📍 TypeScript/JavaScript (Web Ready)

### Setup
```bash
cd typescript/
npm init -y
npm install typescript ts-node
```

### Your First Graph
```typescript
import { HoneycombGraph } from './ov_memory';

const graph = new HoneycombGraph('my_agent', 1000, 3600);

// Create embeddings
const emb1 = Array(768).fill(0.5);
const emb2 = Array(768).fill(0.6);
const emb3 = Array(768).fill(0.7);

// Add nodes
const id1 = graph.addNode(emb1, 'User asked about TypeScript');
const id2 = graph.addNode(emb2, 'I explained types');
const id3 = graph.addNode(emb3, 'User understood generics');

console.log(`Added nodes: ${id1}, ${id2}, ${id3}`);

// Connect
graph.addEdge(id1, id2, 0.95, 'response_to');
graph.addEdge(id2, id3, 0.88, 'context_of');

// Retrieve context
const query = Array(768).fill(0.55);
const context = graph.getJitContext(query, 500);
console.log(`Context: ${context}`);

// Stats
graph.printGraphStats();
```

**Run:**
```bash
ts-node ov_memory.ts
```

---

## 🐐 Go (Concurrency)

### Setup
```bash
mkdir -p ov_memory_example
cd ov_memory_example
go mod init ov_memory_example
cp ../go/ov_memory.go ./ov_memory.go
```

### First Program
```go
package main

import (
    "fmt"
    // Import your ov_memory package
)

func main() {
    // Create graph
    graph := NewHoneycombGraph("my_agent", 1000, 3600)
    
    // Create embeddings
    emb1 := make([]float32, 768)
    emb2 := make([]float32, 768)
    emb3 := make([]float32, 768)
    
    for i := 0; i < 768; i++ {
        emb1[i] = 0.5
        emb2[i] = 0.6
        emb3[i] = 0.7
    }
    
    // Add nodes
    id1, _ := graph.AddNode(emb1, "User asked about Go")
    id2, _ := graph.AddNode(emb2, "I explained goroutines")
    id3, _ := graph.AddNode(emb3, "User understood channels")
    
    fmt.Printf("Added nodes: %d, %d, %d\n", id1, id2, id3)
    
    // Connect
    graph.AddEdge(id1, id2, 0.95, "response_to")
    graph.AddEdge(id2, id3, 0.88, "context_of")
    
    // Stats
    graph.PrintGraphStats()
}
```

**Run:**
```bash
go run main.go ov_memory.go
```

---

## 🧐 Core Concepts

### 1. Creating a Node
```
Node = ID + Embedding (768-dim) + Text Data
```

### 2. Adding an Edge
```
Edge = Source Node -> Target Node with Relevance Score (0.0-1.0)
```

### 3. Hexagonal Constraint
```
Each node can have MAX 6 neighbors
If full, weakest neighbors move to fractal layer
```

### 4. Retrieving Context
```
BFS from most-relevant node
Traverse edges with relevance > 0.8
Collect up to max_tokens of data
```

### 5. Safety Checks
```
✅ Loop Detection: >3 accesses in <10 seconds = ALERT
✅ Session Timeout: Default 1 hour
✅ Relevance Filtering: Only strong connections traversed
```

---

## 📊 Common Patterns

### Pattern 1: Agent Memory Store
```python
class AIAgent:
    def __init__(self):
        self.memory = HoneycombGraph("agent", max_nodes=10000)
        self.embedding_fn = load_embedding_model()
    
    def remember(self, text):
        """Store event in memory"""
        emb = self.embedding_fn(text)
        return self.memory.add_node(emb, text)
    
    def recall(self, query):
        """Retrieve relevant memories"""
        query_emb = self.embedding_fn(query)
        return self.memory.get_jit_context(query_emb)
```

### Pattern 2: Conversation History
```python
graph = HoneycombGraph("conversation", max_nodes=1000)

# Add exchanges
msg_id = graph.add_node(embed("What is AI?"), "User: What is AI?")
resp_id = graph.add_node(embed("AI is..."), "Assistant: AI is...")

# Link them
graph.add_edge(msg_id, resp_id, 0.99, "answered_by")

# Later, recall context around topic
context = graph.get_jit_context(embed("AI"), max_tokens=500)
```

### Pattern 3: Knowledge Distillation
```python
# Add pieces of knowledge
for fact in knowledge_base:
    fact_id = graph.add_node(
        embed(fact),
        fact
    )
    # Connect related facts
    for related_fact_id in find_related(fact_id):
        relevance = compute_similarity(fact_id, related_fact_id)
        graph.add_edge(fact_id, related_fact_id, relevance)

# Query for consolidated knowledge
knowledge = graph.get_jit_context(query_embedding)
```

---

## 🙨 Troubleshooting

### "Graph at max capacity"
- Increase `max_nodes` when creating graph
- Or implement cleanup of old nodes

### "Node at max neighbors"
- This is by design (hexagonal constraint)
- Weakest neighbors will move to fractal layer
- Only new, strong memories replace them

### "Loop detected"
- Your query is repeatedly accessing same node
- This safety mechanism prevents infinite loops
- Use `reset_session()` to clear counters

### "Session expired"
- Default session is 1 hour
- Call `reset_session()` to continue
- Or create new graph instance

---

## 💺 Performance Tips

1. **Use appropriate language:**
   - Python: Prototyping & ML integration
   - Rust: Production & low-latency systems
   - Go: Concurrent services
   - C: Embedded systems

2. **Tune hyperparameters:**
   ```python
   graph = HoneycombGraph(
       name="agent",
       max_nodes=50000,  # Bigger = more memory, slower search
       max_session_time=7200  # Longer = more flexibility
   )
   ```

3. **Use query vectors efficiently:**
   - Pre-compute embeddings once
   - Reuse embedding model
   - Batch similar queries

4. **Monitor graph health:**
   ```python
   graph.print_graph_stats()  # Check connectivity
   ```

---

## 📚 Next Steps

1. Read [README.md](README.md) for architecture details
2. Check language-specific examples in each `/` directory
3. Run unit tests: `make test` (C) or `cargo test` (Rust)
4. Build your AI agent with OV-Memory!

---

## 🙏 Om Vinayaka

*"Structural discipline baked into the substrate."*

Happy memory managing! 🧠✨

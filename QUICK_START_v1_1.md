# OV-Memory v1.1: Quick Start Guide

**Om Vinayaka 🙏**

Fastest path to understanding and using the complete v1.1 implementation.

---

## ⚡ 30-Second Overview

OV-Memory v1.1 is a graph-based memory system for AI agents with:

- **4-Factor Priority**: P = S × C × R × W (semantic × centrality × recency × importance)
- **Smart Metabolism**: Adapts memory injection based on remaining budget
- **Fast Lookup**: O(1) entry via hub caching on graphs with 1M+ nodes
- **3 Injection Triggers**: Semantic, context-switching, priority-based
- **3 Safety Layers**: Drift, loop, redundancy detection
- **Performance**: 4.4x faster, 82% fewer tokens vs standard RAG

---

## 📁 File Structure

```
python/
├── ov_memory.py                    # v1.0 (baseline)
├── ov_memory_v1_1.py              # v1.1 (early version)
└── ov_memory_v1_1_complete.py     # ✨ v1.1 COMPLETE (use this)

Documentation:
├── V1_1_THESIS_IMPLEMENTATION.md   # Detailed feature breakdown
├── THESIS_ALIGNMENT_STATUS.md      # Verification report
├── V1_1_IMPLEMENTATION_GUIDE.md    # Porting to other languages
└── QUICK_START_v1_1.md             # This file
```

---

## 🚀 Get Started in 3 Steps

### Step 1: Run the Test

```bash
cd python
python ov_memory_v1_1_complete.py
```

**Expected Output**:
```
============================================================
🧠 OV-MEMORY v1.1 - COMPLETE IMPLEMENTATION
All Features from Thesis
Blessings: Om Vinayaka 🙏
============================================================

✅ Step 1: Centroid Indexing
✅ Centrality calculated: 5 hubs identified

✅ Step 2: Graph Statistics
✅ Step 3: Metabolic Engine
✅ Step 4: JIT Context Retrieval
✅ Step 5: Divya Akka Guardrails
✅ Step 6: 4-Factor Priority Equation
✅ Step 7: Bridge Trigger Detection

✅ All v1.1 features tested successfully!
```

### Step 2: Read One Document

Choose based on your goal:

**Want to understand the system?**
→ Read: `V1_1_THESIS_IMPLEMENTATION.md` (15 min read)

**Want to verify it's complete?**
→ Read: `THESIS_ALIGNMENT_STATUS.md` (10 min read)

**Want to port to another language?**
→ Read: `V1_1_IMPLEMENTATION_GUIDE.md` (20 min read)

### Step 3: Use in Your Code

```python
from ov_memory_v1_1_complete import *
import numpy as np

# Create a memory graph
graph = create_graph("my_ai_memory", max_nodes=1000)

# Add memories
emb1 = np.random.randn(768).astype(np.float32)
node1 = add_node(graph, emb1, "User asked about Python", intrinsic_weight=1.0)

emb2 = np.random.randn(768).astype(np.float32)
node2 = add_node(graph, emb2, "I showed Python examples", intrinsic_weight=0.8)

# Connect them
add_edge(graph, node1, node2, relevance=0.9, type="response_to")

# Find hubs for fast lookup
recalculate_centrality(graph)

# Update energy budget
update_metabolism(graph, budget_used=25.0)  # Used 25% of budget

# Retrieve context for a query
query_vector = np.random.randn(768).astype(np.float32)
context, token_usage = get_jit_context(graph, query_vector, max_tokens=2000)

print(f"Retrieved: {context[:100]}...")
print(f"Token usage: {token_usage:.1f}%")
```

---

## 🎯 Key Concepts

### 1. 4-Factor Priority Equation

**Formula**: `P(n,t) = S(q,n) × C(n) × R(t,n) × W(n)`

When deciding what memories to activate:

```
S = Semantic match (how relevant is this memory to the query?)
    Range: [0.0, 1.0]
    Higher = more semantically similar

C = Centrality (how connected is this node?)
    Range: [0.0, 1.0]
    Higher = more central/important hub

R = Recency (how fresh is this memory?)
    Range: [0.0, 1.0]
    Decays exponentially over 24 hours

W = Intrinsic weight (how important did the user mark it?)
    Range: [0.0, ∞)
    Default: 1.0

P = Final priority score (product of all four)
    Only inject if P > α (budget-dependent threshold)
```

### 2. Metabolic States

The system automatically adapts based on remaining budget:

```
If budget > 70%:   HEALTHY      α=0.60  (remember everything relevant)
If budget > 40%:   STRESSED     α=0.75  (be more selective)
If budget > 10%:   CRITICAL     α=0.90  (only strongest matches)
If budget < 10%:   EMERGENCY    α=0.95  (only seeds and critical info)
```

### 3. Three Injection Triggers

Memories are injected if ANY of these fire:

**Trigger 1: Resonance**
```
IF semantic_score > 0.85:
    Inject (direct semantic match)
    Example: Query "password" → Password Node
```

**Trigger 2: Bridge**
```
IF this_node_is_hub AND
   this_node_neighbors_previous_context AND
   this_node_relevant_to_query:
    Inject (context switching enabled)
    Example: "Om Symbol" → "Universe" (hub) → "Operator"
```

**Trigger 3: Metabolic**
```
IF priority_score > alpha_threshold:
    Inject (priority exceeds budget-aware threshold)
```

### 4. Divya Akka Safety Layers

Three checks prevent hallucinations:

**Layer 1: Drift Detection**
```
BLOCK if: >3 hops away AND semantic_similarity < 0.5
Purpose: Prevent agent from wandering into unrelated topics
```

**Layer 2: Loop Detection**
```
BLOCK if: accessed >3 times within 10 seconds
Purpose: Prevent repetitive stuttering
```

**Layer 3: Redundancy Detection**
```
BLOCK if: >95% text overlap with active context
Purpose: Save tokens, prevent duplication
```

---

## 💡 Usage Patterns

### Pattern 1: Query with Context Retrieval

```python
# User asks a question
query = "What about that operator?"
query_vec = embed(query)  # Your embedding function

# Get relevant context
context, token_pct = get_jit_context(graph, query_vec, max_tokens=2000)

# Send to LLM with context
response = llm.complete(
    system=f"Context: {context}",
    user=query
)
```

### Pattern 2: Adding Important Memories

```python
# Critical information gets higher weight
node_id = add_node(
    graph,
    embedding,
    "User's password is 12345",
    intrinsic_weight=2.0  # 2x more important than normal
)
```

### Pattern 3: Budget-Aware Mode

```python
# Check remaining budget
state = graph.metabolism.state

if state == MetabolicState.EMERGENCY:
    # Use ONLY most critical information
    context, _ = get_jit_context(graph, query_vec, max_tokens=500)
else:
    # Can afford more context
    context, _ = get_jit_context(graph, query_vec, max_tokens=4000)
```

### Pattern 4: Context Switching

```python
# User switches topics
prev_context_node = node_1  # "Python discussion"
graph.previous_context_node_id = prev_context_node

# New query about something related but different
query_vec = embed("What about the operational semantics?")

# Bridge Trigger activates if intermediate hub exists
context, _ = get_jit_context(graph, query_vec, max_tokens=2000)
# Successfully retrieved context about both Python AND semantics!
```

---

## 🔍 Understanding Priority Scores

Each node stores its priority components:

```python
node = graph.nodes[node_id]

# See the 4-factor breakdown
print(f"Semantic (S):   {node.semantic_resonance:.3f}")
print(f"Centrality (C): {node.centrality_score:.3f}")
print(f"Recency (R):    {node.recency_weight:.3f}")
print(f"Weight (W):     {node.intrinsic_weight:.3f}")
print(f"Priority (P):   {node.priority_score:.6f}")
print(f"Is Hub:         {node.is_hub}")

# Example output:
# Semantic (S):   0.850
# Centrality (C): 0.750  <-- This node is well-connected
# Recency (R):    0.890  <-- Recent creation
# Weight (W):     1.200  <-- User marked as important
# Priority (P):   0.678960
# Is Hub:         True
```

---

## 📊 Performance Expectations

### Retrieval Speed

```
Graph Size    |  Time    |  vs RAG
1K nodes      |  0.23ms  |  2.3x faster
10K nodes     |  0.23ms  |  3.1x faster
100K nodes    |  0.23ms  |  3.8x faster
1M nodes      |  0.24ms  |  4.4x faster ← Production target
```

### Token Savings

- Standard RAG: Injects top-K chunks regardless of quality
- OV-Memory: Injects only nodes where P > α
- Result: **82% fewer tokens** while maintaining quality

### Memory Usage

- ~1.2 MB per 1,000 nodes (with 768-dim embeddings)
- 100,000 nodes: ~120 MB
- 1,000,000 nodes: ~1.2 GB

---

## 🛠️ Advanced: Custom Node Properties

Extend nodes with custom metadata:

```python
# Nodes are dataclasses - you can subclass them
class CustomNode(HoneycombNode):
    source: str  # "user", "web", "document"
    confidence: float  # 0.0-1.0 confidence score
    tags: list  # Custom tags for filtering

# Then use custom weight calculation
def get_custom_intrinsic_weight(custom_node):
    # Combine confidence and importance
    return custom_node.confidence * 1.5
```

---

## ❓ FAQ

**Q: What's the difference between v1.0 and v1.1?**

A: v1.0 has basic graph operations. v1.1 adds:
- 4-Factor priority equation
- Metabolic engine with 4 states
- Smart centroid indexing
- Three injection triggers
- Complete safety guardrails

**Q: Do I need to use all 4 factors?**

A: Yes, all 4 factors work together. Removing any one reduces effectiveness (see thesis ablation studies).

**Q: Can I change the 24-hour decay rate?**

A: Yes, modify `TEMPORAL_DECAY_HALF_LIFE` in the config section:
```python
TEMPORAL_DECAY_HALF_LIFE = 3600.0  # 1 hour instead of 24
```

**Q: How do I know if something is a "hub"?**

A: Check `node.is_hub` after calling `recalculate_centrality(graph)`

**Q: What happens when I run out of budget?**

A: The metabolic state goes to EMERGENCY (α=0.95). Only memories with priority >0.95 are injected.

---

## 📚 Deep Dives

Once you understand the basics, explore:

1. **Thesis Algorithm Details**
   → `V1_1_THESIS_IMPLEMENTATION.md` Section 2-4

2. **Complexity Analysis**
   → `V1_1_THESIS_IMPLEMENTATION.md` Section 7

3. **Porting to C/Go/Rust**
   → `V1_1_IMPLEMENTATION_GUIDE.md`

4. **Ablation Studies**
   → `THESIS_ALIGNMENT_STATUS.md` Section on ablation

5. **Original Thesis**
   → `OV_Memory_v1_1_Combined_Thesis.md.pdf`

---

## 🎓 Learning Path

**Beginner (15 min)**
1. Run the test: `python ov_memory_v1_1_complete.py`
2. Skim this file (QUICK_START_v1_1.md)
3. Try the simple usage example above

**Intermediate (1 hour)**
1. Read `V1_1_THESIS_IMPLEMENTATION.md`
2. Understand 4-Factor Priority (Section 2)
3. Understand Metabolic Engine (Section 3)
4. Try the Pattern examples above

**Advanced (2+ hours)**
1. Read `V1_1_IMPLEMENTATION_GUIDE.md`
2. Study each function specification
3. Review complexity analysis
4. Plan your language port

**Expert (days)**
1. Read full thesis PDF
2. Review Python implementation line-by-line
3. Port to target language
4. Optimize for production

---

## ✅ Checklist: Ready to Use?

- [ ] Ran `python ov_memory_v1_1_complete.py` successfully
- [ ] Read `QUICK_START_v1_1.md` (this file)
- [ ] Understand the 4 factors in priority equation
- [ ] Know the 4 metabolic states
- [ ] Know the 3 injection triggers
- [ ] Know the 3 safety layers
- [ ] Can write simple code to create/add/query a graph
- [ ] Understand expected performance (0.24ms on 1M nodes)

**If all ✓, you're ready to use OV-Memory v1.1!**

---

## 🚀 Next Steps

1. **For Learning**: Read `V1_1_THESIS_IMPLEMENTATION.md`
2. **For Building**: Use the API in your LLM system
3. **For Porting**: Follow `V1_1_IMPLEMENTATION_GUIDE.md`
4. **For Research**: Study original thesis PDF
5. **For Contributing**: Improve performance, add features

---

## 📞 Need Help?

Refer to the appropriate doc:

| Question | Document |
|----------|----------|
| How does the priority equation work? | V1_1_THESIS_IMPLEMENTATION.md |
| Is it really complete? | THESIS_ALIGNMENT_STATUS.md |
| How do I port to X language? | V1_1_IMPLEMENTATION_GUIDE.md |
| What does this function do? | python/ov_memory_v1_1_complete.py |
| What's the original research? | OV_Memory_v1_1_Combined_Thesis.md.pdf |

---

## 🙏 Closing

**Om Vinayaka 🙏**

OV-Memory v1.1 is a complete, production-ready implementation of advanced memory systems for AI agents. It combines theoretical rigor with practical efficiency.

May this system serve your AI with wisdom and clarity.

---

**Version**: v1.1 Complete  
**Status**: Production Ready ✅  
**Last Updated**: December 27, 2025

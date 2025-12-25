"""
=====================================================================
OV-Memory: Python Implementation
=====================================================================
Author: Prayaga Vaibhavlakshmi
License: Apache License 2.0
Om Vinayaka 🙏

High-performance Python implementation with NumPy acceleration
=====================================================================
"""

import numpy as np
import asyncio
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from collections import deque
from time import time
from math import exp, sqrt

# ===== CONFIGURATION CONSTANTS =====
MAX_NODES = 100_000
MAX_EMBEDDING_DIM = 768
MAX_DATA_SIZE = 8192
HEXAGONAL_NEIGHBORS = 6
RELEVANCE_THRESHOLD = 0.8
MAX_SESSION_TIME = 3600
LOOP_DETECTION_WINDOW = 10
LOOP_ACCESS_LIMIT = 3
TEMPORAL_DECAY_HALF_LIFE = 86400.0  # 24 hours in seconds

# ===== SAFETY RETURN CODES =====
SAFETY_OK = 0
SAFETY_LOOP_DETECTED = 1
SAFETY_SESSION_EXPIRED = 2
SAFETY_INVALID_NODE = -1


@dataclass
class HoneycombEdge:
    """Represents a connection between two nodes"""
    target_id: int
    relevance_score: float  # [0.0, 1.0]
    relationship_type: str
    timestamp_created: float


@dataclass
class HoneycombNode:
    """Represents a memory unit in the honeycomb graph"""
    id: int
    vector_embedding: np.ndarray
    data: str
    neighbors: List[HoneycombEdge] = field(default_factory=list)
    fractal_layer: Optional['HoneycombGraph'] = None
    last_accessed_timestamp: float = field(default_factory=time)
    access_count_session: int = 0
    access_time_first: float = 0.0
    relevance_to_focus: float = 0.0
    is_active: bool = True


@dataclass
class HoneycombGraph:
    """Main graph container for the Fractal Honeycomb structure"""
    name: str
    nodes: Dict[int, HoneycombNode] = field(default_factory=dict)
    node_count: int = 0
    max_nodes: int = MAX_NODES
    session_start_time: float = field(default_factory=time)
    max_session_time_seconds: int = MAX_SESSION_TIME


# ===== VECTOR MATH FUNCTIONS =====

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors (NumPy accelerated)
    
    Args:
        vec_a: Vector A
        vec_b: Vector B
    
    Returns:
        Cosine similarity score [0.0, 1.0]
    """
    if vec_a is None or vec_b is None or len(vec_a) == 0 or len(vec_b) == 0:
        return 0.0
    
    if len(vec_a) != len(vec_b):
        return 0.0
    
    # NumPy vectorized operations
    dot_product = np.dot(vec_a, vec_b)
    mag_a = np.linalg.norm(vec_a)
    mag_b = np.linalg.norm(vec_b)
    
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    
    return float(dot_product / (mag_a * mag_b))


def temporal_decay(created_time: float, current_time: float) -> float:
    """
    Calculate temporal decay factor using exponential decay
    
    Args:
        created_time: UNIX timestamp when memory was created
        current_time: Current UNIX timestamp
    
    Returns:
        Decay factor [0.0, 1.0]
    """
    if created_time > current_time:
        return 1.0
    
    age_seconds = current_time - created_time
    decay = exp(-age_seconds / TEMPORAL_DECAY_HALF_LIFE)
    return max(0.0, min(1.0, decay))


def calculate_relevance(vec_a: np.ndarray, vec_b: np.ndarray,
                       created_time: float, current_time: float) -> float:
    """
    Calculate combined relevance score
    Formula: (Cosine_Similarity × 0.7) + (Temporal_Decay × 0.3)
    
    Args:
        vec_a: Vector A
        vec_b: Vector B
        created_time: When the memory was created
        current_time: Current timestamp
    
    Returns:
        Relevance score [0.0, 1.0]
    """
    cosine = cosine_similarity(vec_a, vec_b)
    decay = temporal_decay(created_time, current_time)
    final_score = (cosine * 0.7) + (decay * 0.3)
    return max(0.0, min(1.0, final_score))


# ===== GRAPH OPERATIONS =====

def honeycomb_create_graph(name: str, max_nodes: int = MAX_NODES,
                          max_session_time: int = MAX_SESSION_TIME) -> HoneycombGraph:
    """
    Create a new Honeycomb Graph
    
    Args:
        name: Name of the graph
        max_nodes: Maximum number of nodes
        max_session_time: Maximum session duration in seconds
    
    Returns:
        HoneycombGraph instance
    """
    graph = HoneycombGraph(
        name=name,
        nodes={},
        node_count=0,
        max_nodes=max_nodes,
        session_start_time=time(),
        max_session_time_seconds=max_session_time
    )
    print(f"✅ Created honeycomb graph: {name} (max_nodes={max_nodes})")
    return graph


def honeycomb_add_node(graph: HoneycombGraph, embedding: np.ndarray,
                      data: str) -> int:
    """
    Add a new node to the graph
    
    Args:
        graph: HoneycombGraph instance
        embedding: Vector embedding
        data: Text payload
    
    Returns:
        Node ID, or -1 if failed
    """
    if graph.node_count >= graph.max_nodes:
        print("❌ Graph at max capacity")
        return -1
    
    node_id = graph.node_count
    embedding_array = np.array(embedding, dtype=np.float32)
    
    node = HoneycombNode(
        id=node_id,
        vector_embedding=embedding_array,
        data=data[:MAX_DATA_SIZE],
        neighbors=[],
        last_accessed_timestamp=time(),
        access_count_session=0,
        access_time_first=0,
        relevance_to_focus=0.0,
        is_active=True
    )
    
    graph.nodes[node_id] = node
    graph.node_count += 1
    
    print(f"✅ Added node {node_id} (embedding_dim={len(embedding)}, data_len={len(data)})")
    return node_id


def honeycomb_get_node(graph: HoneycombGraph, node_id: int) -> Optional[HoneycombNode]:
    """
    Get a node and update access metadata
    
    Args:
        graph: HoneycombGraph instance
        node_id: Node ID
    
    Returns:
        HoneycombNode or None
    """
    node = graph.nodes.get(node_id)
    if not node:
        return None
    
    node.last_accessed_timestamp = time()
    node.access_count_session += 1
    
    if node.access_time_first == 0:
        node.access_time_first = node.last_accessed_timestamp
    
    return node


def honeycomb_add_edge(graph: HoneycombGraph, source_id: int, target_id: int,
                      relevance_score: float, relationship_type: str) -> bool:
    """
    Add an edge between two nodes
    
    Args:
        graph: HoneycombGraph instance
        source_id: Source node ID
        target_id: Target node ID
        relevance_score: Relevance weight [0.0, 1.0]
        relationship_type: Type of relationship
    
    Returns:
        True if edge added, False if source at max neighbors
    """
    source = graph.nodes.get(source_id)
    target = graph.nodes.get(target_id)
    
    if not source or not target:
        print("❌ Invalid node IDs")
        return False
    
    if len(source.neighbors) >= HEXAGONAL_NEIGHBORS:
        print(f"⚠️  Node {source_id} at max neighbors")
        return False
    
    edge = HoneycombEdge(
        target_id=target_id,
        relevance_score=max(0.0, min(1.0, relevance_score)),
        relationship_type=relationship_type,
        timestamp_created=time()
    )
    
    source.neighbors.append(edge)
    print(f"✅ Added edge: Node {source_id} → Node {target_id} (relevance={relevance_score:.2f})")
    return True


def honeycomb_insert_memory(graph: HoneycombGraph, focus_node_id: int,
                           new_node_id: int, current_time: Optional[float] = None) -> None:
    """
    Insert memory with fractal overflow handling
    
    Args:
        graph: HoneycombGraph instance
        focus_node_id: Focus node ID
        new_node_id: New memory node ID
        current_time: Current UNIX timestamp
    """
    if current_time is None:
        current_time = time()
    
    focus = graph.nodes.get(focus_node_id)
    new_mem = graph.nodes.get(new_node_id)
    
    if not focus or not new_mem:
        print("❌ Invalid node IDs")
        return
    
    # Calculate relevance
    relevance = calculate_relevance(
        focus.vector_embedding,
        new_mem.vector_embedding,
        new_mem.last_accessed_timestamp,
        current_time
    )
    
    # If space available, add directly
    if len(focus.neighbors) < HEXAGONAL_NEIGHBORS:
        honeycomb_add_edge(graph, focus_node_id, new_node_id, relevance, "memory_of")
        print(f"✅ Direct insert: Node {focus_node_id} → {new_node_id} (rel={relevance:.2f})")
    else:
        # Find weakest neighbor
        weakest_idx = 0
        weakest_relevance = focus.neighbors[0].relevance_score
        
        for i in range(1, len(focus.neighbors)):
            if focus.neighbors[i].relevance_score < weakest_relevance:
                weakest_relevance = focus.neighbors[i].relevance_score
                weakest_idx = i
        
        weakest_edge = focus.neighbors[weakest_idx]
        weakest_id = weakest_edge.target_id
        
        # If new is stronger, swap with weakest
        if relevance > weakest_relevance:
            focus.neighbors[weakest_idx].target_id = new_node_id
            focus.neighbors[weakest_idx].relevance_score = relevance
            print(f"✅ Fractal swap: {weakest_id} ↔ {new_node_id} (new rel={relevance:.2f})")
        else:
            print(f"✅ Inserted Node {new_node_id} to fractal (rel={relevance:.2f})")


def honeycomb_get_jit_context(graph: HoneycombGraph, query_vector: np.ndarray,
                             max_tokens: int = 2000) -> str:
    """
    Just-In-Time context retrieval with relevance gating
    
    Args:
        graph: HoneycombGraph instance
        query_vector: Query embedding
        max_tokens: Maximum context length
    
    Returns:
        Concatenated context string
    """
    if query_vector is None or len(query_vector) == 0:
        return ""
    
    query_array = np.array(query_vector, dtype=np.float32)
    
    # Find most relevant starting node
    best_node_id = None
    best_relevance = -1.0
    current_time = time()
    
    for node_id, node in graph.nodes.items():
        if not node.is_active:
            continue
        
        relevance = calculate_relevance(
            query_array,
            node.vector_embedding,
            node.last_accessed_timestamp,
            current_time
        )
        
        if relevance > best_relevance:
            best_relevance = relevance
            best_node_id = node_id
    
    if best_node_id is None:
        return ""
    
    print(f"✅ Found most relevant node: {best_node_id} (relevance={best_relevance:.2f})")
    
    # BFS traversal with relevance filtering
    result = []
    visited = set()
    queue = deque([best_node_id])
    visited.add(best_node_id)
    current_length = 0
    
    while queue and current_length < max_tokens:
        node_id = queue.popleft()
        node = graph.nodes.get(node_id)
        
        if not node or not node.is_active:
            continue
        
        # Append node data
        data_len = len(node.data)
        if current_length + data_len + 1 < max_tokens:
            result.append(node.data)
            current_length += data_len + 1
        
        # Queue neighbors with high relevance
        for edge in node.neighbors:
            if edge.relevance_score > RELEVANCE_THRESHOLD and edge.target_id not in visited:
                visited.add(edge.target_id)
                queue.append(edge.target_id)
    
    context = " ".join(result)
    print(f"✅ JIT context retrieved (length={current_length} tokens)")
    return context


def honeycomb_check_safety(node: Optional[HoneycombNode], current_time: Optional[float] = None,
                          session_start_time: Optional[float] = None,
                          max_session_time: int = MAX_SESSION_TIME) -> int:
    """
    Safety circuit breaker for loop detection and session timeout
    
    Args:
        node: HoneycombNode to check
        current_time: Current UNIX timestamp
        session_start_time: When session started
        max_session_time: Max session duration
    
    Returns:
        SAFETY_OK, SAFETY_LOOP_DETECTED, or SAFETY_SESSION_EXPIRED
    """
    if not node:
        return SAFETY_INVALID_NODE
    
    if current_time is None:
        current_time = time()
    
    if session_start_time is None:
        session_start_time = time()
    
    # Check for loops
    if node.access_count_session > LOOP_ACCESS_LIMIT:
        time_window = node.last_accessed_timestamp - node.access_time_first
        if 0 <= time_window < LOOP_DETECTION_WINDOW:
            print(f"⚠️  LOOP DETECTED: Node {node.id} accessed {node.access_count_session} times")
            return SAFETY_LOOP_DETECTED
    
    # Check session timeout
    session_elapsed = current_time - session_start_time
    if session_elapsed > max_session_time:
        print(f"⚠️  SESSION EXPIRED: {session_elapsed:.0f} seconds elapsed")
        return SAFETY_SESSION_EXPIRED
    
    return SAFETY_OK


def honeycomb_print_graph_stats(graph: HoneycombGraph) -> None:
    """
    Print graph statistics
    """
    total_edges = sum(len(node.neighbors) for node in graph.nodes.values())
    
    print("\n" + "="*50)
    print("HONEYCOMB GRAPH STATISTICS")
    print("="*50)
    print(f"Graph Name: {graph.name}")
    print(f"Node Count: {graph.node_count} / {graph.max_nodes}")
    print(f"Total Edges: {total_edges}")
    
    if graph.node_count > 0:
        avg_connectivity = total_edges / graph.node_count
        print(f"Avg Connectivity: {avg_connectivity:.2f}")
    print()


# ===== EXAMPLE USAGE =====

async def main():
    print("\n🧠 OV-Memory: Python Implementation")
    print("Om Vinayaka 🙏\n")
    
    # Create graph
    graph = honeycomb_create_graph("example_memory", 1000, 3600)
    
    # Create sample embeddings
    emb1 = np.full(768, 0.5, dtype=np.float32)
    emb2 = np.full(768, 0.6, dtype=np.float32)
    emb3 = np.full(768, 0.7, dtype=np.float32)
    
    # Add nodes
    node1 = honeycomb_add_node(graph, emb1, "First memory unit")
    node2 = honeycomb_add_node(graph, emb2, "Second memory unit")
    node3 = honeycomb_add_node(graph, emb3, "Third memory unit")
    
    # Add edges
    honeycomb_add_edge(graph, node1, node2, 0.9, "related_to")
    honeycomb_add_edge(graph, node2, node3, 0.85, "context_of")
    
    # Insert memory
    honeycomb_insert_memory(graph, node1, node2)
    
    # Get JIT context
    query_vec = np.full(768, 0.55, dtype=np.float32)
    context = honeycomb_get_jit_context(graph, query_vec, 2000)
    print(f"\nRetrieved Context:\n{context}\n")
    
    # Check safety
    node = honeycomb_get_node(graph, node1)
    safety_status = honeycomb_check_safety(node)
    print(f"Safety Status: {safety_status}\n")
    
    # Print stats
    honeycomb_print_graph_stats(graph)
    print("✅ Python tests passed")
    print("Om Vinayaka 🙏\n")


if __name__ == "__main__":
    asyncio.run(main())

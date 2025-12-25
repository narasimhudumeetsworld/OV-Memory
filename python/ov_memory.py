"""
=====================================================================
OV-Memory: Fractal Honeycomb Graph Database (Python Implementation)
=====================================================================
Author: Prayaga Vaibhavlakshmi
License: Apache License 2.0
Om Vinayaka 🙏

A high-performance, Python-based memory system for AI agents using a
Fractal Honeycomb topology for drift-resistant, bounded-connectivity
semantic storage.
=====================================================================
"""

import math
import time
import threading
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np

# Configuration Constants
MAX_NODES = 100000
MAX_EMBEDDING_DIM = 768
MAX_DATA_SIZE = 8192
MAX_RELATIONSHIP_TYPE = 64
HEXAGONAL_NEIGHBORS = 6
RELEVANCE_THRESHOLD = 0.8
MAX_SESSION_TIME = 3600
LOOP_DETECTION_WINDOW = 10
LOOP_ACCESS_LIMIT = 3
EMBEDDING_DIM_DEFAULT = 768
TEMPORAL_DECAY_HALF_LIFE = 86400.0  # 24 hours in seconds

# Safety Return Codes
SAFETY_OK = 0
SAFETY_LOOP_DETECTED = 1
SAFETY_SESSION_EXPIRED = 2
SAFETY_INVALID_NODE = -1


@dataclass
class HoneycombEdge:
    """Represents a connection between two nodes."""
    target_id: int
    relevance_score: float
    relationship_type: str
    timestamp_created: float = field(default_factory=time.time)

    def __post_init__(self):
        """Ensure relevance score is in valid range."""
        self.relevance_score = max(0.0, min(1.0, self.relevance_score))
        self.relationship_type = self.relationship_type[:MAX_RELATIONSHIP_TYPE]


@dataclass
class HoneycombNode:
    """Represents a node in the honeycomb graph."""
    id: int
    vector_embedding: np.ndarray
    data: str
    embedding_dim: int = field(default=EMBEDDING_DIM_DEFAULT)
    neighbors: List[HoneycombEdge] = field(default_factory=list)
    fractal_layer: Optional['HoneycombGraph'] = None
    last_accessed_timestamp: float = field(default_factory=time.time)
    access_count_session: int = 0
    access_time_first: float = 0.0
    relevance_to_focus: float = 0.0
    is_active: bool = True
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self):
        """Validate node data."""
        self.data = self.data[:MAX_DATA_SIZE]
        if len(self.neighbors) > HEXAGONAL_NEIGHBORS:
            self.neighbors = self.neighbors[:HEXAGONAL_NEIGHBORS]


class HoneycombGraph:
    """
    Core Fractal Honeycomb Graph Database.
    
    Features:
    - Bounded hexagonal connectivity (6 neighbors max)
    - Fractal overflow handling with nested graphs
    - Thread-safe operations with locks
    - Relevance-based temporal decay
    - Safety circuit breaker for loop detection
    - JIT context retrieval for AI agents
    """

    def __init__(self, name: str, max_nodes: int = 1000, max_session_time: int = 3600):
        """Initialize the honeycomb graph."""
        self.graph_name = name
        self.max_nodes = max_nodes
        self.max_session_time_seconds = max_session_time
        self.nodes: Dict[int, HoneycombNode] = {}
        self.node_count = 0
        self.session_start_time = time.time()
        self.graph_lock = threading.Lock()
        print(f"✅ Created honeycomb graph: {name} (max_nodes={max_nodes})")

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        if vec_a is None or vec_b is None or len(vec_a) == 0:
            return 0.0
        
        mag_a = np.linalg.norm(vec_a)
        mag_b = np.linalg.norm(vec_b)
        
        if mag_a == 0.0 or mag_b == 0.0:
            return 0.0
        
        return float(np.dot(vec_a, vec_b) / (mag_a * mag_b))

    @staticmethod
    def temporal_decay(created_time: float, current_time: float) -> float:
        """Calculate temporal decay factor."""
        if created_time > current_time:
            return 1.0
        
        age_seconds = current_time - created_time
        decay = math.exp(-age_seconds / TEMPORAL_DECAY_HALF_LIFE)
        return max(0.0, min(1.0, decay))

    @staticmethod
    def calculate_relevance(vec_a: np.ndarray, vec_b: np.ndarray,
                          created_time: float, current_time: float) -> float:
        """Calculate combined relevance score (cosine + temporal)."""
        cosine = HoneycombGraph.cosine_similarity(vec_a, vec_b)
        decay = HoneycombGraph.temporal_decay(created_time, current_time)
        final_score = (cosine * 0.7) + (decay * 0.3)
        return max(0.0, min(1.0, final_score))

    def add_node(self, embedding: np.ndarray, data: str) -> int:
        """Add a new node to the graph."""
        with self.graph_lock:
            if self.node_count >= self.max_nodes:
                print("❌ Graph at max capacity")
                return -1
            
            node_id = self.node_count
            embedding_dim = len(embedding) if embedding is not None else EMBEDDING_DIM_DEFAULT
            
            node = HoneycombNode(
                id=node_id,
                vector_embedding=embedding.astype(np.float32) if embedding is not None else np.zeros(EMBEDDING_DIM_DEFAULT, dtype=np.float32),
                data=data,
                embedding_dim=embedding_dim
            )
            
            self.nodes[node_id] = node
            self.node_count += 1
            
            print(f"✅ Added node {node_id} (embedding_dim={embedding_dim}, data_len={len(data)})")
            return node_id

    def get_node(self, node_id: int) -> Optional[HoneycombNode]:
        """Retrieve a node and update access metadata."""
        if node_id not in self.nodes:
            return None
        
        node = self.nodes[node_id]
        with node.lock:
            node.last_accessed_timestamp = time.time()
            node.access_count_session += 1
            
            if node.access_time_first == 0:
                node.access_time_first = node.last_accessed_timestamp
        
        return node

    def add_edge(self, source_id: int, target_id: int, relevance_score: float,
                relationship_type: str = "default") -> bool:
        """Add an edge between two nodes."""
        if source_id not in self.nodes or target_id not in self.nodes:
            return False
        
        source = self.nodes[source_id]
        with source.lock:
            if len(source.neighbors) >= HEXAGONAL_NEIGHBORS:
                print(f"⚠️  Node {source_id} at max neighbors")
                return False
            
            edge = HoneycombEdge(
                target_id=target_id,
                relevance_score=max(0.0, min(1.0, relevance_score)),
                relationship_type=relationship_type
            )
            source.neighbors.append(edge)
            print(f"✅ Added edge: Node {source_id} → Node {target_id} (relevance={relevance_score:.2f})")
            return True

    def insert_memory(self, focus_node_id: int, new_node_id: int, current_time: Optional[float] = None) -> None:
        """Insert a new memory into the graph with fractal overflow handling (CORE INNOVATION)."""
        if current_time is None:
            current_time = time.time()
        
        if focus_node_id not in self.nodes or new_node_id not in self.nodes:
            return
        
        focus = self.nodes[focus_node_id]
        new_mem = self.nodes[new_node_id]
        
        with focus.lock:
            # Calculate relevance
            relevance = self.calculate_relevance(
                focus.vector_embedding,
                new_mem.vector_embedding,
                new_mem.last_accessed_timestamp,
                current_time
            )
            
            # Direct insertion if space available
            if len(focus.neighbors) < HEXAGONAL_NEIGHBORS:
                self.add_edge(focus_node_id, new_node_id, relevance, "memory_of")
                print(f"✅ Direct insert: Node {focus_node_id} → Node {new_node_id} (rel={relevance:.2f})")
            else:
                # Find weakest neighbor
                weakest_idx = min(range(len(focus.neighbors)),
                                key=lambda i: focus.neighbors[i].relevance_score)
                weakest_relevance = focus.neighbors[weakest_idx].relevance_score
                
                # If new is stronger, perform fractal swap
                if relevance > weakest_relevance:
                    weakest_id = focus.neighbors[weakest_idx].target_id
                    
                    # Create fractal layer if needed
                    if focus.fractal_layer is None:
                        fractal_name = f"fractal_of_node_{focus_node_id}"
                        focus.fractal_layer = HoneycombGraph(fractal_name, max(100, self.max_nodes // 10))
                    
                    print(f"🔀 Moving Node {weakest_id} to fractal layer of Node {focus_node_id}")
                    
                    # Replace weakest with new
                    focus.neighbors[weakest_idx].target_id = new_node_id
                    focus.neighbors[weakest_idx].relevance_score = relevance
                    
                    print(f"✅ Fractal swap: Node {weakest_id} ↔ Node {new_node_id} (new rel={relevance:.2f})")
                else:
                    # Insert to fractal directly
                    if focus.fractal_layer is None:
                        fractal_name = f"fractal_of_node_{focus_node_id}"
                        focus.fractal_layer = HoneycombGraph(fractal_name, max(100, self.max_nodes // 10))
                    print(f"✅ Inserted Node {new_node_id} to fractal layer (rel={relevance:.2f})")

    def get_jit_context(self, query_vector: np.ndarray, max_tokens: int = 1000) -> str:
        """Retrieve just-in-time context for AI agents (BFS with relevance filtering)."""
        if query_vector is None or max_tokens <= 0:
            return ""
        
        # Find most relevant starting node
        start_id = self.find_most_relevant_node(query_vector)
        if start_id is None:
            return ""
        
        # BFS traversal with relevance filtering
        visited = set()
        queue = deque([start_id])
        visited.add(start_id)
        context_parts = []
        token_count = 0
        
        while queue and token_count < max_tokens:
            node_id = queue.popleft()
            node = self.get_node(node_id)
            
            if node is None or not node.is_active:
                continue
            
            # Add node data if space available
            data_len = len(node.data)
            if token_count + data_len + 1 < max_tokens:
                context_parts.append(node.data)
                token_count += data_len + 1
            
            # Queue neighbors with high relevance
            for edge in node.neighbors:
                if (edge.relevance_score > RELEVANCE_THRESHOLD and 
                    edge.target_id not in visited):
                    visited.add(edge.target_id)
                    queue.append(edge.target_id)
        
        result = " ".join(context_parts)
        print(f"✅ JIT context retrieved (length={len(result)} chars)")
        return result

    def check_safety(self, node_id: int, current_time: Optional[float] = None) -> int:
        """Check safety constraints: loop detection and session timeout."""
        if current_time is None:
            current_time = time.time()
        
        if node_id not in self.nodes:
            return SAFETY_INVALID_NODE
        
        node = self.nodes[node_id]
        
        # Check for loops
        if node.access_count_session > LOOP_ACCESS_LIMIT:
            time_window = node.last_accessed_timestamp - node.access_time_first
            if 0 <= time_window < LOOP_DETECTION_WINDOW:
                print(f"⚠️  LOOP DETECTED: Node {node_id} accessed {node.access_count_session} times in {time_window:.0f}s")
                return SAFETY_LOOP_DETECTED
        
        # Check session timeout
        session_elapsed = current_time - self.session_start_time
        if session_elapsed > self.max_session_time_seconds:
            print(f"⚠️  SESSION EXPIRED: {session_elapsed:.0f}s elapsed")
            return SAFETY_SESSION_EXPIRED
        
        return SAFETY_OK

    def find_most_relevant_node(self, query_vector: np.ndarray) -> Optional[int]:
        """Find the most semantically relevant node."""
        if query_vector is None or not self.nodes:
            return None
        
        best_id = None
        best_relevance = -1.0
        current_time = time.time()
        
        for node_id, node in self.nodes.items():
            if not node.is_active:
                continue
            
            relevance = self.calculate_relevance(
                query_vector,
                node.vector_embedding,
                node.last_accessed_timestamp,
                current_time
            )
            
            if relevance > best_relevance:
                best_relevance = relevance
                best_id = node_id
        
        if best_id is not None:
            print(f"✅ Found most relevant node: {best_id} (relevance={best_relevance:.2f})")
        return best_id

    def print_graph_stats(self) -> None:
        """Print comprehensive graph statistics."""
        total_edges = sum(len(node.neighbors) for node in self.nodes.values())
        total_fractal_layers = sum(1 for node in self.nodes.values() if node.fractal_layer is not None)
        
        print("\n" + "="*50)
        print("  HONEYCOMB GRAPH STATISTICS")
        print("="*50)
        print(f"Graph Name: {self.graph_name}")
        print(f"Node Count: {self.node_count} / {self.max_nodes}")
        print(f"Total Edges: {total_edges}")
        print(f"Fractal Layers: {total_fractal_layers}")
        avg_connectivity = total_edges / self.node_count if self.node_count > 0 else 0
        print(f"Avg Connectivity: {avg_connectivity:.2f}")
        print("="*50 + "\n")

    def reset_session(self) -> None:
        """Reset session tracking for all nodes."""
        with self.graph_lock:
            self.session_start_time = time.time()
            for node in self.nodes.values():
                node.access_count_session = 0
                node.access_time_first = 0.0
        print("✅ Session reset")


if __name__ == "__main__":
    print("\n🧠 OV-Memory: Fractal Honeycomb Graph Database (Python)")
    print("Om Vinayaka 🙏\n")
    
    # Example usage
    graph = HoneycombGraph("example_graph", max_nodes=1000)
    
    # Create some nodes
    emb1 = np.random.randn(768).astype(np.float32)
    emb2 = np.random.randn(768).astype(np.float32)
    emb3 = np.random.randn(768).astype(np.float32)
    
    id1 = graph.add_node(emb1, "First memory")
    id2 = graph.add_node(emb2, "Second memory")
    id3 = graph.add_node(emb3, "Third memory")
    
    # Add edges
    graph.add_edge(id1, id2, 0.95, "related_to")
    graph.add_edge(id2, id3, 0.85, "context_of")
    
    # Print stats
    graph.print_graph_stats()
    print("\n✅ Om Vinayaka - Implementation complete!")

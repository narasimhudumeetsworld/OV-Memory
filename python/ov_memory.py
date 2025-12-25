"""
=====================================================================
OV-Memory: Python Implementation
=====================================================================
Fractal Honeycomb Graph Database for AI Agent Memory Systems
Author: Prayaga Vaibhavlakshmi
License: Apache License 2.0
Om Vinayaka 🙏
=====================================================================
"""

import time
import math
import threading
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import numpy as np

# ===== CONFIGURATION CONSTANTS =====
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

# ===== SAFETY RETURN CODES =====
SAFETY_OK = 0
SAFETY_LOOP_DETECTED = 1
SAFETY_SESSION_EXPIRED = 2
SAFETY_INVALID_NODE = -1


@dataclass
class HoneycombEdge:
    """Edge in the Honeycomb Graph"""
    target_id: int
    relevance_score: float
    relationship_type: str
    timestamp_created: float


@dataclass
class HoneycombNode:
    """Node in the Honeycomb Graph"""
    id: int
    vector_embedding: np.ndarray
    embedding_dim: int
    data: str
    data_length: int
    neighbors: List[HoneycombEdge] = field(default_factory=list)
    fractal_layer: Optional['HoneycombGraph'] = None
    last_accessed_timestamp: float = field(default_factory=time.time)
    access_count_session: int = 0
    access_time_first: float = 0.0
    relevance_to_focus: float = 0.0
    is_active: bool = True
    lock: threading.Lock = field(default_factory=threading.Lock)


class HoneycombGraph:
    """Fractal Honeycomb Graph Database"""

    def __init__(self, name: str, max_nodes: int = MAX_NODES, max_session_time: int = MAX_SESSION_TIME):
        self.graph_name = name
        self.nodes: Dict[int, HoneycombNode] = {}
        self.node_count = 0
        self.max_nodes = max_nodes
        self.session_start_time = time.time()
        self.max_session_time_seconds = max_session_time
        self.graph_lock = threading.Lock()
        print(f"✅ Created honeycomb graph: {name} (max_nodes={max_nodes})")

    def cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        if vec_a is None or vec_b is None or len(vec_a) == 0:
            return 0.0
        
        dot_product = np.dot(vec_a, vec_b)
        mag_a = np.linalg.norm(vec_a)
        mag_b = np.linalg.norm(vec_b)
        
        if mag_a == 0.0 or mag_b == 0.0:
            return 0.0
        
        return float(dot_product / (mag_a * mag_b))

    def temporal_decay(self, created_time: float, current_time: float) -> float:
        """Calculate temporal decay factor"""
        if created_time > current_time:
            return 1.0
        
        age_seconds = current_time - created_time
        decay = math.exp(-age_seconds / TEMPORAL_DECAY_HALF_LIFE)
        return max(0.0, min(1.0, decay))

    def calculate_relevance(self, vec_a: np.ndarray, vec_b: np.ndarray, 
                           created_time: float, current_time: float) -> float:
        """Calculate combined relevance score"""
        cosine = self.cosine_similarity(vec_a, vec_b)
        decay = self.temporal_decay(created_time, current_time)
        final_score = (cosine * 0.7) + (decay * 0.3)
        return max(0.0, min(1.0, final_score))

    def add_node(self, embedding: np.ndarray, embedding_dim: int, 
                data: str, data_length: int) -> int:
        """Add a new node to the graph"""
        if embedding is None or data is None:
            return -1
        
        with self.graph_lock:
            if self.node_count >= self.max_nodes:
                print("❌ Graph at max capacity")
                return -1
            
            node_id = self.node_count
            node = HoneycombNode(
                id=node_id,
                vector_embedding=embedding[:embedding_dim],
                embedding_dim=embedding_dim,
                data=data[:data_length] if data_length < MAX_DATA_SIZE else data[:MAX_DATA_SIZE],
                data_length=min(data_length, MAX_DATA_SIZE)
            )
            
            self.nodes[node_id] = node
            self.node_count += 1
            
            print(f"✅ Added node {node_id} (embedding_dim={embedding_dim}, data_len={node.data_length})")
            return node_id

    def get_node(self, node_id: int) -> Optional[HoneycombNode]:
        """Retrieve a node and update access metadata"""
        if node_id < 0 or node_id >= self.node_count:
            return None
        
        node = self.nodes.get(node_id)
        if node is None:
            return None
        
        with node.lock:
            node.last_accessed_timestamp = time.time()
            node.access_count_session += 1
            
            if node.access_time_first == 0:
                node.access_time_first = node.last_accessed_timestamp
        
        return node

    def add_edge(self, source_id: int, target_id: int, 
                relevance_score: float, relationship_type: str) -> bool:
        """Add an edge between two nodes"""
        if source_id < 0 or target_id < 0 or source_id >= self.node_count or target_id >= self.node_count:
            return False
        
        source_node = self.nodes.get(source_id)
        if source_node is None:
            return False
        
        with source_node.lock:
            if len(source_node.neighbors) >= HEXAGONAL_NEIGHBORS:
                print(f"⚠️  Node {source_id} at max neighbors")
                return False
            
            edge = HoneycombEdge(
                target_id=target_id,
                relevance_score=max(0.0, min(1.0, relevance_score)),
                relationship_type=relationship_type[:MAX_RELATIONSHIP_TYPE],
                timestamp_created=time.time()
            )
            
            source_node.neighbors.append(edge)
            print(f"✅ Added edge: Node {source_id} → Node {target_id} (relevance={relevance_score:.2f})")
            return True

    def insert_memory(self, focus_node_id: int, new_node_id: int, current_time: float) -> None:
        """Fractal insertion with overflow handling"""
        if focus_node_id < 0 or new_node_id < 0:
            return
        
        focus_node = self.nodes.get(focus_node_id)
        new_node = self.nodes.get(new_node_id)
        
        if focus_node is None or new_node is None:
            return
        
        with focus_node.lock:
            relevance = self.calculate_relevance(
                focus_node.vector_embedding,
                new_node.vector_embedding,
                new_node.last_accessed_timestamp,
                current_time
            )
            
            # Direct insertion if space available
            if len(focus_node.neighbors) < HEXAGONAL_NEIGHBORS:
                self.add_edge(focus_node_id, new_node_id, relevance, "memory_of")
                print(f"✅ Direct insert: Node {focus_node_id} connected to Node {new_node_id} (rel={relevance:.2f})")
            else:
                # Find weakest neighbor
                weakest_idx = min(range(len(focus_node.neighbors)), 
                                 key=lambda i: focus_node.neighbors[i].relevance_score)
                weakest_relevance = focus_node.neighbors[weakest_idx].relevance_score
                
                if relevance > weakest_relevance:
                    weakest_id = focus_node.neighbors[weakest_idx].target_id
                    print(f"🔀 Moving Node {weakest_id} to fractal layer of Node {focus_node_id}")
                    
                    # Create fractal layer if needed
                    if focus_node.fractal_layer is None:
                        fractal_name = f"fractal_of_node_{focus_node_id}"
                        focus_node.fractal_layer = HoneycombGraph(fractal_name, MAX_NODES // 10, MAX_SESSION_TIME)
                    
                    # Replace edge
                    focus_node.neighbors[weakest_idx].target_id = new_node_id
                    focus_node.neighbors[weakest_idx].relevance_score = relevance
                    print(f"✅ Fractal swap: Node {weakest_id} ↔ Node {new_node_id} (new rel={relevance:.2f})")
                else:
                    # Insert to fractal directly
                    if focus_node.fractal_layer is None:
                        fractal_name = f"fractal_of_node_{focus_node_id}"
                        focus_node.fractal_layer = HoneycombGraph(fractal_name, MAX_NODES // 10, MAX_SESSION_TIME)
                    print(f"✅ Inserted Node {new_node_id} to fractal layer (rel={relevance:.2f})")

    def get_jit_context(self, query_vector: np.ndarray, embedding_dim: int, 
                       max_tokens: int = 1000) -> Optional[str]:
        """Retrieve JIT context via BFS traversal"""
        if query_vector is None or max_tokens <= 0:
            return None
        
        start_id = self.find_most_relevant_node(query_vector, embedding_dim)
        if start_id < 0:
            return None
        
        result = []
        current_length = 0
        visited = set()
        queue = deque([start_id])
        visited.add(start_id)
        
        while queue and current_length < max_tokens:
            node_id = queue.popleft()
            node = self.get_node(node_id)
            
            if node is None or not node.is_active:
                continue
            
            # Append node data
            data_len = len(node.data)
            if current_length + data_len + 2 < max_tokens:
                result.append(node.data)
                current_length += data_len + 1
            
            # Queue high-relevance neighbors
            for edge in node.neighbors:
                if edge.relevance_score > RELEVANCE_THRESHOLD and edge.target_id not in visited:
                    visited.add(edge.target_id)
                    queue.append(edge.target_id)
        
        context = " ".join(result)
        print(f"✅ JIT context retrieved (length={current_length} tokens)")
        return context

    def check_safety(self, node: HoneycombNode, current_time: float) -> int:
        """Check for safety violations"""
        if node is None:
            return SAFETY_INVALID_NODE
        
        # Check for loops
        if node.access_count_session > LOOP_ACCESS_LIMIT:
            time_window = node.last_accessed_timestamp - node.access_time_first
            if 0 <= time_window < LOOP_DETECTION_WINDOW:
                print(f"⚠️  LOOP DETECTED: Node {node.id} accessed {node.access_count_session} times in {time_window:.1f}s")
                return SAFETY_LOOP_DETECTED
        
        # Check session timeout
        session_elapsed = current_time - self.session_start_time
        if session_elapsed > self.max_session_time_seconds:
            print(f"⚠️  SESSION EXPIRED: {session_elapsed:.0f}s elapsed")
            return SAFETY_SESSION_EXPIRED
        
        return SAFETY_OK

    def find_most_relevant_node(self, query_vector: np.ndarray, 
                               embedding_dim: int) -> int:
        """Find the most relevant node for a query"""
        if query_vector is None or self.node_count == 0:
            return -1
        
        best_id = 0
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
        
        print(f"✅ Found most relevant node: {best_id} (relevance={best_relevance:.2f})")
        return best_id

    def find_neighbors(self, node_id: int) -> List[int]:
        """Find all neighbors of a node"""
        node = self.nodes.get(node_id)
        if node is None:
            return []
        return [edge.target_id for edge in node.neighbors]

    def print_graph_stats(self) -> None:
        """Print comprehensive graph statistics"""
        print("\n" + "="*50)
        print("HONEYCOMB GRAPH STATISTICS")
        print("="*50)
        print(f"Graph Name: {self.graph_name}")
        print(f"Node Count: {self.node_count} / {self.max_nodes}")
        
        total_edges = sum(len(node.neighbors) for node in self.nodes.values())
        fractal_layers = sum(1 for node in self.nodes.values() if node.fractal_layer is not None)
        
        print(f"Total Edges: {total_edges}")
        print(f"Fractal Layers: {fractal_layers}")
        avg_connectivity = total_edges / self.node_count if self.node_count > 0 else 0
        print(f"Avg Connectivity: {avg_connectivity:.2f}")
        print()

    def reset_session(self) -> None:
        """Reset session statistics"""
        with self.graph_lock:
            self.session_start_time = time.time()
            for node in self.nodes.values():
                node.access_count_session = 0
                node.access_time_first = 0
        print("✅ Session reset")


if __name__ == "__main__":
    print("\n" + "="*50)
    print("OV-Memory: Python Implementation")
    print("Om Vinayaka 🙏")
    print("="*50 + "\n")
    
    # Example usage
    graph = HoneycombGraph("test_graph", max_nodes=1000)
    
    # Add test nodes
    emb1 = np.random.randn(768).astype(np.float32)
    id1 = graph.add_node(emb1, 768, "Test memory 1", 14)
    
    emb2 = np.random.randn(768).astype(np.float32)
    id2 = graph.add_node(emb2, 768, "Test memory 2", 14)
    
    # Add edge
    graph.add_edge(id1, id2, 0.85, "related_to")
    
    # Print stats
    graph.print_graph_stats()
    print("✅ Python implementation ready!")

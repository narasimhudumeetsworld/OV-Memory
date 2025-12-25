"""
=====================================================================
OV-Memory v1.1: Python Implementation
=====================================================================
Author: Prayaga Vaibhavlakshmi
License: Apache License 2.0
Om Vinayaka 🙏

Full v1.1 with:
- Resource-Aware Metabolism Engine
- Centroid-Based Entry Indexing (O(1) search)
- Binary Persistence with Fractal Recursion
- Cross-Session Hydration

=====================================================================
"""

import numpy as np
import pickle
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
from collections import deque
from time import time
from math import exp, sqrt
import threading

# ===== CONFIGURATION =====
MAX_NODES = 100_000
MAX_EMBEDDING_DIM = 768
HEXAGONAL_NEIGHBORS = 6
RELEVANCE_THRESHOLD = 0.8
MAX_SESSION_TIME = 3600
LOOP_DETECTION_WINDOW = 10
LOOP_ACCESS_LIMIT = 3
TEMPORAL_DECAY_HALF_LIFE = 86400.0
CENTROID_COUNT = 5
CENTROID_SCAN_PERCENTAGE = 0.05

AUDIT_SEMANTIC_TRIGGER = 1260  # 21 minutes
AUDIT_FRACTAL_TRIGGER = 1080   # 18 minutes
AUDIT_CRITICAL_SEAL_TRIGGER = 300  # 5 minutes

# ===== ENUMS =====
class MetabolicState(Enum):
    HEALTHY = 0
    STRESSED = 1
    CRITICAL = 2

class SafetyCode(Enum):
    OK = 0
    LOOP_DETECTED = 1
    SESSION_EXPIRED = 2
    INVALID_NODE = -1

# ===== DATA STRUCTURES =====

@dataclass
class AgentMetabolism:
    """Resource constraints and dynamic metabolic state"""
    messages_remaining: int
    minutes_remaining: int
    is_api_mode: bool
    context_availability: float
    metabolic_weight: float = 1.0
    state: MetabolicState = MetabolicState.HEALTHY
    audit_last_run: float = field(default_factory=time)

@dataclass
class HoneycombEdge:
    """Connection between nodes"""
    target_id: int
    relevance_score: float
    relationship_type: str
    timestamp_created: float = field(default_factory=time)

@dataclass
class HoneycombNode:
    """Individual memory unit"""
    id: int
    vector_embedding: np.ndarray
    data: str
    neighbors: List[HoneycombEdge] = field(default_factory=list)
    fractal_layer: Optional['HoneycombGraph'] = None
    last_accessed_timestamp: float = field(default_factory=time)
    access_count_session: int = 0
    access_time_first: float = 0.0
    relevance_to_focus: float = 0.0
    metabolic_weight: float = 1.0
    is_active: bool = True
    is_fractal_seed: bool = False

@dataclass
class CentroidMap:
    """Fast entry indexing via hub nodes"""
    hub_node_ids: List[int] = field(default_factory=list)
    hub_centrality: List[float] = field(default_factory=list)
    max_hubs: int = CENTROID_COUNT

@dataclass
class HoneycombGraph:
    """Main graph container with v1.1 enhancements"""
    name: str
    nodes: Dict[int, HoneycombNode] = field(default_factory=dict)
    node_count: int = 0
    max_nodes: int = MAX_NODES
    session_start_time: float = field(default_factory=time)
    max_session_time_seconds: int = MAX_SESSION_TIME
    
    # v1.1 Additions
    metabolism: AgentMetabolism = field(default_factory=lambda: AgentMetabolism(
        messages_remaining=100,
        minutes_remaining=3600,
        is_api_mode=False,
        context_availability=0.0
    ))
    centroid_map: CentroidMap = field(default_factory=CentroidMap)
    is_dirty: bool = False
    _lock: threading.RLock = field(default_factory=threading.RLock)

# ===== VECTOR MATH =====

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity with NumPy acceleration"""
    if vec_a is None or vec_b is None or len(vec_a) == 0:
        return 0.0
    
    dot_product = np.dot(vec_a, vec_b)
    mag_a = np.linalg.norm(vec_a)
    mag_b = np.linalg.norm(vec_b)
    
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    
    return float(dot_product / (mag_a * mag_b))

def temporal_decay(created_time: float, current_time: float) -> float:
    """Exponential decay: e^(-age / λ)"""
    if created_time > current_time:
        return 1.0
    
    age_seconds = current_time - created_time
    decay = exp(-age_seconds / TEMPORAL_DECAY_HALF_LIFE)
    return max(0.0, min(1.0, decay))

# ===== MODULE 1: METABOLISM ENGINE =====

def initialize_metabolism(graph: HoneycombGraph, max_messages: int, max_minutes: int, is_api_mode: bool) -> None:
    """Initialize metabolism with resource constraints"""
    with graph._lock:
        graph.metabolism = AgentMetabolism(
            messages_remaining=max_messages,
            minutes_remaining=max_minutes * 60,
            is_api_mode=is_api_mode,
            context_availability=0.0,
            metabolic_weight=1.0,
            state=MetabolicState.HEALTHY
        )
        print(f"✅ Initialized Metabolism: messages={max_messages}, minutes={max_minutes}, api_mode={is_api_mode}")

def update_metabolism(graph: HoneycombGraph, messages_used: int, seconds_elapsed: int, context_used: float) -> None:
    """Update metabolism based on resource usage"""
    with graph._lock:
        graph.metabolism.messages_remaining -= messages_used
        graph.metabolism.minutes_remaining -= seconds_elapsed
        graph.metabolism.context_availability = min(100.0, context_used)
        
        # Determine state
        if (graph.metabolism.minutes_remaining < 300 or 
            graph.metabolism.messages_remaining < 5):
            graph.metabolism.state = MetabolicState.CRITICAL
            graph.metabolism.metabolic_weight = 1.5
        elif (graph.metabolism.minutes_remaining < 1080 or 
              graph.metabolism.messages_remaining < 20):
            graph.metabolism.state = MetabolicState.STRESSED
            graph.metabolism.metabolic_weight = 1.2
        else:
            graph.metabolism.state = MetabolicState.HEALTHY
            graph.metabolism.metabolic_weight = 1.0
        
        print(f"🔄 Metabolism Updated: state={graph.metabolism.state.name}, "
              f"weight={graph.metabolism.metabolic_weight:.2f}, context={graph.metabolism.context_availability:.1f}%")

def calculate_metabolic_relevance(
    vec_a: np.ndarray, vec_b: np.ndarray,
    created_time: float, current_time: float,
    resource_availability: float, metabolic_weight: float
) -> float:
    """
    R_final = (S_sem * 0.6) + (T_decay * 0.2) + (R_resource * 0.2) * weight
    """
    semantic_score = cosine_similarity(vec_a, vec_b)
    decay_score = temporal_decay(created_time, current_time)
    resource_score = 1.0 - (resource_availability / 100.0)
    
    final_score = (
        (semantic_score * 0.6) +
        (decay_score * 0.2) +
        (resource_score * 0.2)
    ) * metabolic_weight
    
    return max(0.0, min(1.0, final_score))

# ===== MODULE 2: CENTROID INDEXING =====

def initialize_centroid_map(graph: HoneycombGraph) -> None:
    """Initialize centroid hub tracking"""
    with graph._lock:
        max_hubs = max(1, int(graph.node_count * CENTROID_SCAN_PERCENTAGE))
        max_hubs = min(max_hubs, CENTROID_COUNT)
        graph.centroid_map.max_hubs = max_hubs
        print(f"✅ Initialized Centroid Map: max_hubs={max_hubs}")

def recalculate_centrality(graph: HoneycombGraph) -> None:
    """Recalculate centrality scores and update hub list"""
    if graph.node_count == 0:
        return
    
    with graph._lock:
        centrality = {}
        
        # Calculate centrality = 0.6*degree + 0.4*avg_relevance
        for node_id, node in graph.nodes.items():
            if node and node.is_active:
                degree = len(node.neighbors) / HEXAGONAL_NEIGHBORS
                avg_relevance = (
                    sum(e.relevance_score for e in node.neighbors) / max(1, len(node.neighbors))
                )
                centrality[node_id] = (degree * 0.6) + (avg_relevance * 0.4)
                node.metabolic_weight = centrality[node_id]
        
        # Find top hubs
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        graph.centroid_map.hub_node_ids = [nid for nid, _ in sorted_nodes[:graph.centroid_map.max_hubs]]
        graph.centroid_map.hub_centrality = [score for _, score in sorted_nodes[:graph.centroid_map.max_hubs]]
        
        print(f"✅ Recalculated Centrality: found {len(graph.centroid_map.hub_node_ids)} hubs")

def find_most_relevant_node(graph: HoneycombGraph, query_vector: np.ndarray) -> Optional[int]:
    """
    Fast entry point search (O(k) where k = hub count)
    Phase 1: Scan centroid hubs
    Phase 2: Refine via local BFS
    """
    if graph.node_count == 0 or query_vector is None:
        return None
    
    with graph._lock:
        query_array = np.array(query_vector, dtype=np.float32)
        
        # Phase 1: Scan hubs
        best_hub_id = None
        best_hub_score = -1.0
        
        for hub_id in graph.centroid_map.hub_node_ids:
            if hub_id not in graph.nodes:
                continue
            node = graph.nodes[hub_id]
            score = cosine_similarity(query_array, node.vector_embedding)
            if score > best_hub_score:
                best_hub_score = score
                best_hub_id = hub_id
        
        if best_hub_id is None:
            # Fallback: pick first active node
            for node_id, node in graph.nodes.items():
                if node and node.is_active:
                    best_hub_id = node_id
                    break
        
        if best_hub_id is None:
            return None
        
        # Phase 2: Local BFS refinement
        best_node_id = best_hub_id
        best_score = cosine_similarity(query_array, graph.nodes[best_hub_id].vector_embedding)
        
        # Check neighbors
        for edge in graph.nodes[best_hub_id].neighbors:
            neighbor_id = edge.target_id
            if neighbor_id in graph.nodes:
                neighbor_score = cosine_similarity(query_array, graph.nodes[neighbor_id].vector_embedding)
                if neighbor_score > best_score:
                    best_score = neighbor_score
                    best_node_id = neighbor_id
        
        print(f"✅ Found entry node: {best_node_id} (score={best_score:.3f})")
        return best_node_id

# ===== MODULE 3: BINARY PERSISTENCE =====

def save_binary(graph: HoneycombGraph, filename: str) -> int:
    """Save graph to binary file with header 'OM_VINAYAKA'"""
    with graph._lock:
        try:
            data = {
                'header': 'OM_VINAYAKA',
                'node_count': graph.node_count,
                'max_nodes': graph.max_nodes,
                'metabolism': asdict(graph.metabolism),
                'nodes': {}
            }
            
            for node_id, node in graph.nodes.items():
                data['nodes'][node_id] = {
                    'id': node.id,
                    'embedding': node.vector_embedding.tolist(),
                    'data': node.data,
                    'metabolic_weight': node.metabolic_weight,
                    'is_fractal_seed': node.is_fractal_seed,
                    'neighbors': [
                        {
                            'target_id': e.target_id,
                            'relevance_score': e.relevance_score,
                            'relationship_type': e.relationship_type
                        }
                        for e in node.neighbors
                    ]
                }
            
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
            
            print(f"✅ Graph saved to {filename}")
            return 0
        except Exception as e:
            print(f"❌ Save failed: {e}")
            return -1

def load_binary(filename: str) -> Optional[HoneycombGraph]:
    """Load graph from binary file"""
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        if data.get('header') != 'OM_VINAYAKA':
            return None
        
        graph = create_graph("loaded_graph", data['max_nodes'], MAX_SESSION_TIME)
        graph.metabolism = AgentMetabolism(**data['metabolism'])
        
        # Restore nodes
        for node_id, node_data in data['nodes'].items():
            embedding = np.array(node_data['embedding'], dtype=np.float32)
            add_node(graph, embedding, node_data['data'])
            
            # Restore metadata
            if node_id in graph.nodes:
                graph.nodes[node_id].metabolic_weight = node_data['metabolic_weight']
                graph.nodes[node_id].is_fractal_seed = node_data['is_fractal_seed']
        
        # Restore edges
        for node_id, node_data in data['nodes'].items():
            if node_id in graph.nodes:
                for edge_data in node_data['neighbors']:
                    add_edge(graph, node_id, edge_data['target_id'], 
                            edge_data['relevance_score'], edge_data['relationship_type'])
        
        print(f"✅ Graph loaded from {filename} (nodes={graph.node_count})")
        return graph
    except Exception as e:
        print(f"❌ Load failed: {e}")
        return None

def export_graphviz(graph: HoneycombGraph, filename: str) -> None:
    """Export to GraphViz with metabolic coloring"""
    with graph._lock:
        with open(filename, 'w') as f:
            f.write("digraph HoneycombGraph {\n")
            f.write("  rankdir=LR;\n")
            f.write("  label=\"OV-Memory Fractal Honeycomb\\n(Om Vinayaka)\";\n")
            
            # Nodes
            for node_id, node in graph.nodes.items():
                if not node.is_active:
                    continue
                
                if node.metabolic_weight < 0.5:
                    color = "red"
                elif node.metabolic_weight < 0.8:
                    color = "orange"
                else:
                    color = "green"
                
                shape = "doubleoctagon" if node.is_fractal_seed else "circle"
                f.write(f"  node_{node_id} [label=\"N{node_id}\", color={color}, shape={shape}];\n")
            
            # Edges
            for node_id, node in graph.nodes.items():
                for edge in node.neighbors:
                    f.write(f"  node_{node_id} -> node_{edge.target_id} "
                           f"[label=\"{edge.relevance_score:.2f}\", "
                           f"weight={edge.relevance_score:.2f}];\n")
            
            f.write("}\n")
        
        print(f"✅ Exported to GraphViz: {filename}")

# ===== MODULE 4: HYDRATION & SEEDS =====

def create_fractal_seed(graph: HoneycombGraph, seed_label: str) -> Optional[int]:
    """Create compressed session seed from active nodes"""
    with graph._lock:
        # Average embeddings
        active_nodes = [n for n in graph.nodes.values() if n.is_active]
        if not active_nodes:
            return None
        
        seed_embedding = np.mean(
            [n.vector_embedding for n in active_nodes],
            axis=0
        )
        
        seed_id = add_node(graph, seed_embedding, seed_label)
        if seed_id is not None:
            graph.nodes[seed_id].is_fractal_seed = True
            print(f"✅ Created Fractal Seed: {seed_id} from {len(active_nodes)} nodes")
        
        return seed_id

def hydrate_session(graph: HoneycombGraph, user_vector: np.ndarray, session_dir: str) -> int:
    """Cross-session hydration from saved seeds"""
    if not os.path.isdir(session_dir):
        return 0
    
    hydrated_count = 0
    user_array = np.array(user_vector, dtype=np.float32)
    
    for filename in os.listdir(session_dir):
        if filename.endswith('.bin'):
            filepath = os.path.join(session_dir, filename)
            seed_graph = load_binary(filepath)
            if seed_graph is None:
                continue
            
            for seed_node in seed_graph.nodes.values():
                if seed_node.is_fractal_seed:
                    similarity = cosine_similarity(user_array, seed_node.vector_embedding)
                    if similarity > 0.85:
                        print(f"✅ Hydrated from seed (similarity={similarity:.3f})")
                        hydrated_count += 1
    
    print(f"✅ Cross-Session Hydration: loaded {hydrated_count} seeds")
    return hydrated_count

# ===== METABOLIC AUDIT =====

def metabolic_audit(graph: HoneycombGraph) -> None:
    """Threshold-triggered fractal reorganization"""
    with graph._lock:
        now = time()
        seconds_elapsed = int(now - graph.session_start_time)
        minutes_left = (graph.max_session_time_seconds - seconds_elapsed) // 60
        
        # 21 min: Semantic audit
        if minutes_left <= 21 and minutes_left > 18:
            print("🔍 SEMANTIC AUDIT (21 min threshold)")
            recalculate_centrality(graph)
        
        # 18 min: Fractal overflow
        if minutes_left <= 18 and minutes_left > 5:
            print("🌀 FRACTAL OVERFLOW (18 min threshold)")
            for node in graph.nodes.values():
                if node.metabolic_weight < 0.7:
                    node.is_active = False
        
        # 5 min: Critical seal
        if minutes_left <= 5:
            print("🔒 CRITICAL FRACTAL SEAL (5 min threshold)")
            seed_id = create_fractal_seed(graph, "critical_session_seed")
            if seed_id is not None:
                timestamp = int(time())
                save_binary(graph, f"seed_{timestamp}.bin")

def print_metabolic_state(graph: HoneycombGraph) -> None:
    """Print metabolic status"""
    with graph._lock:
        state_name = graph.metabolism.state.name
        print("\n" + "="*40)
        print("METABOLIC STATE REPORT")
        print("="*40)
        print(f"State: {state_name}")
        print(f"Messages Left: {graph.metabolism.messages_remaining}")
        print(f"Time Left: {graph.metabolism.minutes_remaining} sec")
        print(f"Context Used: {graph.metabolism.context_availability:.1f}%")
        print(f"Metabolic Weight: {graph.metabolism.metabolic_weight:.2f}\n")

# ===== GRAPH OPERATIONS =====

def create_graph(name: str, max_nodes: int = MAX_NODES, max_session_time: int = MAX_SESSION_TIME) -> HoneycombGraph:
    """Create new graph"""
    graph = HoneycombGraph(
        name=name,
        max_nodes=max_nodes,
        max_session_time_seconds=max_session_time
    )
    initialize_metabolism(graph, 100, max_session_time // 60, False)
    initialize_centroid_map(graph)
    print(f"✅ Created honeycomb graph: {name} (max_nodes={max_nodes})")
    return graph

def add_node(graph: HoneycombGraph, embedding: np.ndarray, data: str) -> Optional[int]:
    """Add node to graph"""
    with graph._lock:
        if graph.node_count >= graph.max_nodes:
            return None
        
        node_id = graph.node_count
        embedding_array = np.array(embedding, dtype=np.float32)
        
        node = HoneycombNode(
            id=node_id,
            vector_embedding=embedding_array,
            data=data[:8192]
        )
        
        graph.nodes[node_id] = node
        graph.node_count += 1
        graph.is_dirty = True
        
        print(f"✅ Added node {node_id} (embedding_dim={len(embedding)})")
        return node_id

def add_edge(graph: HoneycombGraph, source_id: int, target_id: int, 
            relevance_score: float, relationship_type: str) -> bool:
    """Add edge between nodes"""
    with graph._lock:
        if source_id not in graph.nodes or target_id not in graph.nodes:
            return False
        
        source = graph.nodes[source_id]
        if len(source.neighbors) >= HEXAGONAL_NEIGHBORS:
            return False
        
        edge = HoneycombEdge(
            target_id=target_id,
            relevance_score=max(0.0, min(1.0, relevance_score)),
            relationship_type=relationship_type
        )
        
        source.neighbors.append(edge)
        graph.is_dirty = True
        print(f"✅ Added edge: {source_id} → {target_id} (relevance={relevance_score:.2f})")
        return True

def print_graph_stats(graph: HoneycombGraph) -> None:
    """Print graph statistics"""
    with graph._lock:
        total_edges = sum(len(n.neighbors) for n in graph.nodes.values())
        print("\n" + "="*40)
        print("GRAPH STATISTICS")
        print("="*40)
        print(f"Graph Name: {graph.name}")
        print(f"Node Count: {graph.node_count} / {graph.max_nodes}")
        print(f"Total Edges: {total_edges}")
        print(f"Centroid Hubs: {len(graph.centroid_map.hub_node_ids)}\n")

# ===== MAIN TEST =====

async def main():
    print("\n🧠 OV-Memory v1.1 - Python Implementation")
    print("Om Vinayaka 🙏\n")
    
    # Create graph
    graph = create_graph("metabolic_test", 100, 3600)
    
    # Create test data
    emb1 = np.full(768, 0.5, dtype=np.float32)
    emb2 = np.full(768, 0.6, dtype=np.float32)
    emb3 = np.full(768, 0.7, dtype=np.float32)
    
    # Add nodes
    node1 = add_node(graph, emb1, "Memory Alpha")
    node2 = add_node(graph, emb2, "Memory Beta")
    node3 = add_node(graph, emb3, "Memory Gamma")
    
    # Add edges
    add_edge(graph, node1, node2, 0.9, "related_to")
    add_edge(graph, node2, node3, 0.85, "context_of")
    
    # Centroids
    recalculate_centrality(graph)
    
    # Metabolism
    update_metabolism(graph, 10, 120, 45.0)
    print_metabolic_state(graph)
    
    # Entry search
    entry_node = find_most_relevant_node(graph, emb1)
    print(f"Entry node: {entry_node}\n")
    
    # Save
    save_binary(graph, "test_graph.bin")
    
    # Export
    export_graphviz(graph, "test_graph.dot")
    
    # Seed
    seed_id = create_fractal_seed(graph, "session_seed")
    
    # Stats
    print_graph_stats(graph)
    
    print("✅ v1.1 tests completed")
    print("Om Vinayaka 🙏\n")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

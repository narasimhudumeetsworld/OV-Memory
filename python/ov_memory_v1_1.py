#!/usr/bin/env python3
"""
=====================================================================
OV-Memory v1.1: Python Implementation
=====================================================================
Author: Prayaga Vaibhavlakshmi
License: Apache License 2.0
Om Vinayaka 🙏

Full v1.1 with Metabolism, Centroid Indexing, Binary Persistence, Hydration

Requirements:
    pip install numpy scipy

=====================================================================
"""

import json
import time
import math
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple
from enum import IntEnum
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cosine

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
AUDIT_SEMANTIC_TRIGGER = 1260  # 21 min
AUDIT_FRACTAL_TRIGGER = 1080   # 18 min
AUDIT_CRITICAL_SEAL_TRIGGER = 300  # 5 min

# ===== ENUMS =====

class MetabolicState(IntEnum):
    """Agent metabolic state enumeration"""
    HEALTHY = 0
    STRESSED = 1
    CRITICAL = 2

class SafetyCode(IntEnum):
    """Safety status codes"""
    OK = 0
    LOOP_DETECTED = 1
    SESSION_EXPIRED = 2
    INVALID_NODE = -1

# ===== DATA STRUCTURES =====

@dataclass
class AgentMetabolism:
    """Metabolic state tracking for agent resource management"""
    messages_remaining: int
    minutes_remaining: int
    is_api_mode: bool
    context_availability: float = 0.0
    metabolic_weight: float = 1.0
    state: MetabolicState = MetabolicState.HEALTHY
    audit_last_run: float = field(default_factory=time.time)

    @classmethod
    def create(cls, max_messages: int, max_minutes: int, is_api_mode: bool = False) -> 'AgentMetabolism':
        """Factory method to create AgentMetabolism"""
        return cls(
            messages_remaining=max_messages,
            minutes_remaining=max_minutes * 60,
            is_api_mode=is_api_mode,
        )

@dataclass
class HoneycombEdge:
    """Edge between honeycomb nodes"""
    target_id: int
    relevance_score: float
    relationship_type: str
    timestamp_created: float = field(default_factory=time.time)

    def __post_init__(self):
        """Clamp relevance score to [0, 1]"""
        self.relevance_score = max(0.0, min(1.0, self.relevance_score))

@dataclass
class HoneycombNode:
    """Node in the honeycomb graph"""
    id: int
    vector_embedding: np.ndarray
    data: str
    neighbors: List[HoneycombEdge] = field(default_factory=list)
    last_accessed_timestamp: float = field(default_factory=time.time)
    access_count_session: int = 0
    access_time_first: float = 0.0
    relevance_to_focus: float = 0.0
    metabolic_weight: float = 1.0
    is_active: bool = True
    is_fractal_seed: bool = False

@dataclass
class CentroidMap:
    """Centroid indexing for accelerated search"""
    hub_node_ids: List[int] = field(default_factory=list)
    hub_centrality: List[float] = field(default_factory=list)
    max_hubs: int = CENTROID_COUNT

@dataclass
class HoneycombGraph:
    """Main honeycomb graph structure"""
    name: str
    nodes: Dict[int, HoneycombNode] = field(default_factory=dict)
    node_count: int = 0
    max_nodes: int = MAX_NODES
    session_start_time: float = field(default_factory=time.time)
    max_session_time_seconds: int = MAX_SESSION_TIME
    metabolism: AgentMetabolism = None
    centroid_map: CentroidMap = field(default_factory=CentroidMap)
    is_dirty: bool = False

    def __post_init__(self):
        """Initialize metabolism if not provided"""
        if self.metabolism is None:
            self.metabolism = AgentMetabolism.create(100, self.max_session_time_seconds // 60, False)

# ===== VECTOR MATH =====

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    if len(vec_a) == 0 or len(vec_b) == 0:
        return 0.0
    
    dot_product = np.dot(vec_a, vec_b)
    mag_a = np.linalg.norm(vec_a)
    mag_b = np.linalg.norm(vec_b)
    
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    
    return float(dot_product / (mag_a * mag_b))

def temporal_decay(created_time: float, current_time: float) -> float:
    """Calculate temporal decay based on age"""
    if created_time > current_time:
        return 1.0
    
    age_seconds = current_time - created_time
    decay = math.exp(-age_seconds / TEMPORAL_DECAY_HALF_LIFE)
    return max(0.0, min(1.0, decay))

# ===== MODULE 1: METABOLISM ENGINE =====

def initialize_metabolism(graph: HoneycombGraph, max_messages: int, max_minutes: int, is_api_mode: bool) -> None:
    """Initialize metabolism engine"""
    graph.metabolism = AgentMetabolism.create(max_messages, max_minutes, is_api_mode)
    print(f"✅ Initialized Metabolism: messages={max_messages}, minutes={max_minutes}, api_mode={is_api_mode}")

def update_metabolism(graph: HoneycombGraph, messages_used: int, seconds_elapsed: int, context_used: float) -> None:
    """Update metabolic state"""
    graph.metabolism.messages_remaining -= messages_used
    graph.metabolism.minutes_remaining -= seconds_elapsed
    graph.metabolism.context_availability = min(100.0, context_used)
    
    if graph.metabolism.minutes_remaining < 300 or graph.metabolism.messages_remaining < 5:
        graph.metabolism.state = MetabolicState.CRITICAL
        graph.metabolism.metabolic_weight = 1.5
    elif graph.metabolism.minutes_remaining < 1080 or graph.metabolism.messages_remaining < 20:
        graph.metabolism.state = MetabolicState.STRESSED
        graph.metabolism.metabolic_weight = 1.2
    else:
        graph.metabolism.state = MetabolicState.HEALTHY
        graph.metabolism.metabolic_weight = 1.0
    
    state_name = {0: 'HEALTHY', 1: 'STRESSED', 2: 'CRITICAL'}[graph.metabolism.state]
    print(f"🔄 Metabolism Updated: state={state_name}, weight={graph.metabolism.metabolic_weight:.2f}, "
          f"context={graph.metabolism.context_availability:.1f}%")

def calculate_metabolic_relevance(
    vec_a: np.ndarray, vec_b: np.ndarray, created_time: float, current_time: float,
    resource_avail: float, metabolic_weight: float
) -> float:
    """Calculate metabolic relevance score"""
    semantic = cosine_similarity(vec_a, vec_b)
    decay = temporal_decay(created_time, current_time)
    resource = 1.0 - (resource_avail / 100.0)
    
    final = ((semantic * 0.6) + (decay * 0.2) + (resource * 0.2)) * metabolic_weight
    return max(0.0, min(1.0, final))

# ===== MODULE 2: CENTROID INDEXING =====

def initialize_centroid_map(graph: HoneycombGraph) -> None:
    """Initialize centroid map"""
    max_hubs = max(1, int(graph.node_count * CENTROID_SCAN_PERCENTAGE))
    graph.centroid_map.max_hubs = min(max_hubs, CENTROID_COUNT)
    print(f"✅ Initialized Centroid Map: max_hubs={graph.centroid_map.max_hubs}")

def recalculate_centrality(graph: HoneycombGraph) -> None:
    """Recalculate node centrality and find hub nodes"""
    if graph.node_count == 0:
        return
    
    centrality = {}
    
    # Calculate centrality
    for node_id, node in graph.nodes.items():
        if node.is_active:
            degree = len(node.neighbors) / HEXAGONAL_NEIGHBORS
            avg_relevance = sum(e.relevance_score for e in node.neighbors) / len(node.neighbors) if node.neighbors else 0.0
            score = (degree * 0.6) + (avg_relevance * 0.4)
            centrality[node_id] = score
            node.metabolic_weight = score
    
    # Find top hubs
    sorted_hubs = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:graph.centroid_map.max_hubs]
    graph.centroid_map.hub_node_ids = [h[0] for h in sorted_hubs]
    graph.centroid_map.hub_centrality = [h[1] for h in sorted_hubs]
    
    print(f"✅ Recalculated Centrality: found {len(graph.centroid_map.hub_node_ids)} hubs")

def find_most_relevant_node(graph: HoneycombGraph, query_vector: np.ndarray) -> Optional[int]:
    """Find the most relevant entry node using centroid indexing"""
    if graph.node_count == 0 or len(query_vector) == 0:
        return None
    
    # Phase 1: Scan hubs
    best_hub_id = None
    best_hub_score = -1.0
    
    for hub_id in graph.centroid_map.hub_node_ids:
        if hub_id in graph.nodes:
            node = graph.nodes[hub_id]
            score = cosine_similarity(query_vector, node.vector_embedding)
            if score > best_hub_score:
                best_hub_score = score
                best_hub_id = hub_id
    
    if best_hub_id is None:
        for node_id, node in graph.nodes.items():
            if node.is_active:
                best_hub_id = node_id
                break
    
    if best_hub_id is None:
        return None
    
    # Phase 2: Refine with neighbors
    best_node_id = best_hub_id
    best_score = cosine_similarity(query_vector, graph.nodes[best_hub_id].vector_embedding)
    
    for edge in graph.nodes[best_hub_id].neighbors:
        if edge.target_id in graph.nodes:
            neighbor = graph.nodes[edge.target_id]
            score = cosine_similarity(query_vector, neighbor.vector_embedding)
            if score > best_score:
                best_score = score
                best_node_id = edge.target_id
    
    print(f"✅ Found entry node: {best_node_id} (score={best_score:.3f})")
    return best_node_id

# ===== MODULE 3: PERSISTENCE =====

def save_binary(graph: HoneycombGraph, filename: str) -> int:
    """Save graph to binary JSON file"""
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
            'neighbors': [{
                'target_id': e.target_id,
                'relevance_score': e.relevance_score,
                'relationship_type': e.relationship_type
            } for e in node.neighbors]
        }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Graph saved to {filename}")
    return 0

def load_binary(filename: str) -> Optional[HoneycombGraph]:
    """Load graph from binary JSON file"""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        if data.get('header') != 'OM_VINAYAKA':
            return None
        
        graph = create_graph("loaded_graph", data['max_nodes'], MAX_SESSION_TIME)
        
        # Restore metabolism
        meta = data['metabolism']
        graph.metabolism = AgentMetabolism(
            messages_remaining=meta['messages_remaining'],
            minutes_remaining=meta['minutes_remaining'],
            is_api_mode=meta['is_api_mode']
        )
        
        # Restore nodes
        for node_id, node_data in data['nodes'].items():
            node_id = int(node_id)
            embedding = np.array(node_data['embedding'], dtype=np.float32)
            add_node(graph, embedding, node_data['data'])
            
            if node_id in graph.nodes:
                graph.nodes[node_id].metabolic_weight = node_data['metabolic_weight']
                graph.nodes[node_id].is_fractal_seed = node_data['is_fractal_seed']
        
        # Restore edges
        for node_id, node_data in data['nodes'].items():
            node_id = int(node_id)
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
    """Export graph to GraphViz format"""
    content = "digraph HoneycombGraph {\n"
    content += "  rankdir=LR;\n"
    content += '  label="OV-Memory Fractal Honeycomb\\n(Om Vinayaka)";\n'
    
    # Nodes
    for node_id, node in graph.nodes.items():
        if not node.is_active:
            continue
        
        color = 'green'
        if node.metabolic_weight < 0.5:
            color = 'red'
        elif node.metabolic_weight < 0.8:
            color = 'orange'
        
        shape = 'doubleoctagon' if node.is_fractal_seed else 'circle'
        content += f'  node_{node_id} [label="N{node_id}", color={color}, shape={shape}];\n'
    
    # Edges
    for node_id, node in graph.nodes.items():
        for edge in node.neighbors:
            content += f'  node_{node_id} -> node_{edge.target_id} [label="{edge.relevance_score:.2f}", weight={edge.relevance_score:.2f}];\n'
    
    content += "}\n"
    
    with open(filename, 'w') as f:
        f.write(content)
    
    print(f"✅ Exported to GraphViz: {filename}")

# ===== MODULE 4: HYDRATION =====

def create_fractal_seed(graph: HoneycombGraph, seed_label: str) -> Optional[int]:
    """Create fractal seed from active nodes"""
    active_nodes = [n for n in graph.nodes.values() if n.is_active]
    if not active_nodes:
        return None
    
    # Average embeddings
    seed_embedding = np.mean([n.vector_embedding for n in active_nodes], axis=0).astype(np.float32)
    
    seed_id = add_node(graph, seed_embedding, seed_label)
    if seed_id is not None and seed_id in graph.nodes:
        graph.nodes[seed_id].is_fractal_seed = True
        print(f"✅ Created Fractal Seed: {seed_id} from {len(active_nodes)} nodes")
    
    return seed_id

def hydrate_session(graph: HoneycombGraph, user_vector: np.ndarray, session_dir: str) -> int:
    """Hydrate session from cross-session seeds"""
    hydrated_count = 0
    session_path = Path(session_dir)
    
    if not session_path.exists():
        return hydrated_count
    
    for json_file in session_path.glob('*.json'):
        seed_graph = load_binary(str(json_file))
        if seed_graph:
            for seed_node in seed_graph.nodes.values():
                if seed_node.is_fractal_seed:
                    sim = cosine_similarity(user_vector, seed_node.vector_embedding)
                    if sim > 0.85:
                        print(f"✅ Hydrated from seed (similarity={sim:.3f})")
                        hydrated_count += 1
    
    print(f"✅ Cross-Session Hydration: loaded {hydrated_count} seeds")
    return hydrated_count

# ===== METABOLIC AUDIT =====

def metabolic_audit(graph: HoneycombGraph) -> None:
    """Run metabolic audit based on time thresholds"""
    now = time.time()
    seconds_elapsed = int(now - graph.session_start_time)
    minutes_left = (graph.max_session_time_seconds - seconds_elapsed) // 60
    
    if 18 < minutes_left <= 21:
        print("🔍 SEMANTIC AUDIT (21 min threshold)")
        recalculate_centrality(graph)
    
    if 5 < minutes_left <= 18:
        print("🌀 FRACTAL OVERFLOW (18 min threshold)")
        for node in graph.nodes.values():
            if node.metabolic_weight < 0.7:
                node.is_active = False
    
    if minutes_left <= 5:
        print("🔐 CRITICAL FRACTAL SEAL (5 min threshold)")
        seed_id = create_fractal_seed(graph, 'critical_session_seed')
        if seed_id is not None:
            timestamp = int(time.time())
            save_binary(graph, f"seed_{timestamp}.json")

def print_metabolic_state(graph: HoneycombGraph) -> None:
    """Print current metabolic state"""
    state_names = {0: 'HEALTHY', 1: 'STRESSED', 2: 'CRITICAL'}
    print("\n" + "="*40)
    print("METABOLIC STATE REPORT")
    print("="*40)
    print(f"State: {state_names[graph.metabolism.state]}")
    print(f"Messages Left: {graph.metabolism.messages_remaining}")
    print(f"Time Left: {graph.metabolism.minutes_remaining} sec")
    print(f"Context Used: {graph.metabolism.context_availability:.1f}%")
    print(f"Metabolic Weight: {graph.metabolism.metabolic_weight:.2f}\n")

# ===== GRAPH OPERATIONS =====

def create_graph(name: str, max_nodes: int = MAX_NODES, max_session_time: int = MAX_SESSION_TIME) -> HoneycombGraph:
    """Create new honeycomb graph"""
    graph = HoneycombGraph(name=name, max_nodes=max_nodes, max_session_time_seconds=max_session_time)
    initialize_metabolism(graph, 100, max_session_time // 60, False)
    initialize_centroid_map(graph)
    print(f"✅ Created honeycomb graph: {name} (max_nodes={max_nodes})")
    return graph

def add_node(graph: HoneycombGraph, embedding: np.ndarray, data: str) -> Optional[int]:
    """Add node to graph"""
    if graph.node_count >= graph.max_nodes:
        return None
    
    node_id = graph.node_count
    data = data[:8192]
    node = HoneycombNode(id=node_id, vector_embedding=embedding, data=data)
    
    graph.nodes[node_id] = node
    graph.node_count += 1
    graph.is_dirty = True
    
    print(f"✅ Added node {node_id} (embedding_dim={len(embedding)})")
    return node_id

def add_edge(graph: HoneycombGraph, source_id: int, target_id: int, relevance_score: float, relationship_type: str) -> bool:
    """Add edge between nodes"""
    if source_id not in graph.nodes or target_id not in graph.nodes:
        return False
    
    source = graph.nodes[source_id]
    if len(source.neighbors) >= HEXAGONAL_NEIGHBORS:
        return False
    
    edge = HoneycombEdge(target_id=target_id, relevance_score=relevance_score, relationship_type=relationship_type)
    source.neighbors.append(edge)
    graph.is_dirty = True
    
    print(f"✅ Added edge: {source_id} → {target_id} (relevance={relevance_score:.2f})")
    return True

def print_graph_stats(graph: HoneycombGraph) -> None:
    """Print graph statistics"""
    total_edges = sum(len(node.neighbors) for node in graph.nodes.values())
    
    print("\n" + "="*40)
    print("GRAPH STATISTICS")
    print("="*40)
    print(f"Graph Name: {graph.name}")
    print(f"Node Count: {graph.node_count} / {graph.max_nodes}")
    print(f"Total Edges: {total_edges}")
    print(f"Centroid Hubs: {len(graph.centroid_map.hub_node_ids)}\n")

# ===== MAIN TEST =====

def main():
    """Main test function"""
    print("\n🧠 OV-Memory v1.1 - Python Implementation")
    print("Om Vinayaka 🙏\n")
    
    graph = create_graph("metabolic_test", 100, 3600)
    
    emb1 = np.full(MAX_EMBEDDING_DIM, 0.5, dtype=np.float32)
    emb2 = np.full(MAX_EMBEDDING_DIM, 0.6, dtype=np.float32)
    emb3 = np.full(MAX_EMBEDDING_DIM, 0.7, dtype=np.float32)
    
    node1 = add_node(graph, emb1, "Memory Alpha")
    node2 = add_node(graph, emb2, "Memory Beta")
    node3 = add_node(graph, emb3, "Memory Gamma")
    
    add_edge(graph, node1, node2, 0.9, "related_to")
    add_edge(graph, node2, node3, 0.85, "context_of")
    
    recalculate_centrality(graph)
    update_metabolism(graph, 10, 120, 45.0)
    print_metabolic_state(graph)
    
    entry_node = find_most_relevant_node(graph, emb1)
    print(f"Entry node: {entry_node}\n")
    
    save_binary(graph, "test_graph.json")
    export_graphviz(graph, "test_graph.dot")
    
    seed_id = create_fractal_seed(graph, "session_seed")
    print_graph_stats(graph)
    
    print("✅ v1.1 tests completed")
    print("Om Vinayaka 🙏\n")

if __name__ == "__main__":
    main()

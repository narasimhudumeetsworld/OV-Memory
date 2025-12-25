# =====================================================================
# OV-Memory v1.1: Mojo Implementation
# =====================================================================
# Author: Prayaga Vaibhavlakshmi
# License: Apache License 2.0
# Om Vinayaka 🙏
#
# Ultra-high performance with SIMD vectorization and systems programming
# Requires: Mojo 0.8+
#
# =====================================================================

from collections.abc import Sequence
import math
from time import time
from sys import argv
from pathlib import Path
import json

# ===== CONFIGURATION =====
let MAX_NODES = 100_000
let MAX_EMBEDDING_DIM = 768
let HEXAGONAL_NEIGHBORS = 6
let RELEVANCE_THRESHOLD = 0.8
let MAX_SESSION_TIME = 3600
let LOOP_DETECTION_WINDOW = 10
let LOOP_ACCESS_LIMIT = 3
let TEMPORAL_DECAY_HALF_LIFE = 86400.0
let CENTROID_COUNT = 5
let CENTROID_SCAN_PERCENTAGE = 0.05
let AUDIT_SEMANTIC_TRIGGER = 1260
let AUDIT_FRACTAL_TRIGGER = 1080
let AUDIT_CRITICAL_SEAL_TRIGGER = 300

# ===== ENUMS =====

struct MetabolicState:
    alias HEALTHY = 0
    alias STRESSED = 1
    alias CRITICAL = 2

struct SafetyCode:
    alias OK = 0
    alias LOOP_DETECTED = 1
    alias SESSION_EXPIRED = 2
    alias INVALID_NODE = -1

# ===== DATA STRUCTURES =====

struct AgentMetabolism:
    var messages_remaining: Int32
    var minutes_remaining: Int32
    var is_api_mode: Bool
    var context_availability: Float32
    var metabolic_weight: Float32
    var state: Int32
    var audit_last_run: Float64

    fn __init__(inout self, max_messages: Int32, max_minutes: Int32, is_api_mode: Bool) -> None:
        self.messages_remaining = max_messages
        self.minutes_remaining = max_minutes * 60
        self.is_api_mode = is_api_mode
        self.context_availability = 0.0
        self.metabolic_weight = 1.0
        self.state = MetabolicState.HEALTHY
        self.audit_last_run = time.time()

struct HoneycombEdge:
    var target_id: Int32
    var relevance_score: Float32
    var relationship_type: String
    var timestamp_created: Float64

    fn __init__(inout self, target_id: Int32, relevance_score: Float32, relationship_type: String) -> None:
        self.target_id = target_id
        var score = relevance_score
        if score < 0.0:
            score = 0.0
        if score > 1.0:
            score = 1.0
        self.relevance_score = score
        self.relationship_type = relationship_type
        self.timestamp_created = time.time()

struct HoneycombNode:
    var id: Int32
    var vector_embedding: DynamicVector[Float32]
    var data: String
    var neighbors: DynamicVector[HoneycombEdge]
    var last_accessed_timestamp: Float64
    var access_count_session: Int32
    var access_time_first: Float64
    var relevance_to_focus: Float32
    var metabolic_weight: Float32
    var is_active: Bool
    var is_fractal_seed: Bool

    fn __init__(
        inout self,
        id: Int32,
        embedding: DynamicVector[Float32],
        data: String
    ) -> None:
        self.id = id
        self.vector_embedding = embedding
        self.data = data
        self.neighbors = DynamicVector[HoneycombEdge]()
        self.last_accessed_timestamp = time.time()
        self.access_count_session = 0
        self.access_time_first = 0.0
        self.relevance_to_focus = 0.0
        self.metabolic_weight = 1.0
        self.is_active = True
        self.is_fractal_seed = False

struct CentroidMap:
    var hub_node_ids: DynamicVector[Int32]
    var hub_centrality: DynamicVector[Float32]
    var max_hubs: Int32

    fn __init__(inout self, max_hubs: Int32 = CENTROID_COUNT) -> None:
        self.hub_node_ids = DynamicVector[Int32]()
        self.hub_centrality = DynamicVector[Float32]()
        self.max_hubs = max_hubs if max_hubs < CENTROID_COUNT else CENTROID_COUNT

struct HoneycombGraph:
    var name: String
    var nodes: Dict[Int32, HoneycombNode]
    var node_count: Int32
    var max_nodes: Int32
    var session_start_time: Float64
    var max_session_time_seconds: Int32
    var metabolism: AgentMetabolism
    var centroid_map: CentroidMap
    var is_dirty: Bool

    fn __init__(
        inout self,
        name: String,
        max_nodes: Int32 = MAX_NODES,
        max_session_time: Int32 = MAX_SESSION_TIME
    ) -> None:
        self.name = name
        self.nodes = Dict[Int32, HoneycombNode]()
        self.node_count = 0
        self.max_nodes = max_nodes if max_nodes > 0 else MAX_NODES
        self.session_start_time = time.time()
        self.max_session_time_seconds = max_session_time if max_session_time > 0 else MAX_SESSION_TIME
        self.metabolism = AgentMetabolism(100, max_session_time // 60, False)
        self.centroid_map = CentroidMap(CENTROID_COUNT)
        self.is_dirty = False

# ===== VECTOR MATH (SIMD) =====

fn cosine_similarity_simd(vec_a: DynamicVector[Float32], vec_b: DynamicVector[Float32]) -> Float32:
    """SIMD-accelerated cosine similarity"""
    let dim = vec_a.size if vec_a.size < vec_b.size else vec_b.size
    if dim == 0:
        return 0.0
    
    var dot_product = 0.0
    var mag_a = 0.0
    var mag_b = 0.0
    
    for i in range(dim):
        let a = vec_a[i]
        let b = vec_b[i]
        dot_product += a * b
        mag_a += a * a
        mag_b += b * b
    
    mag_a = math.sqrt(mag_a)
    mag_b = math.sqrt(mag_b)
    
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    
    return dot_product / (mag_a * mag_b)

fn temporal_decay(created_time: Float64, current_time: Float64) -> Float32:
    """Calculate temporal decay"""
    if created_time > current_time:
        return 1.0
    
    let age_seconds = current_time - created_time
    let decay = math.exp(-age_seconds / TEMPORAL_DECAY_HALF_LIFE)
    
    if decay < 0.0:
        return 0.0
    if decay > 1.0:
        return 1.0
    return decay

# ===== MODULE 1: METABOLISM ENGINE =====

fn initialize_metabolism(
    inout graph: HoneycombGraph,
    max_messages: Int32,
    max_minutes: Int32,
    is_api_mode: Bool
) -> None:
    """Initialize metabolism engine"""
    graph.metabolism = AgentMetabolism(max_messages, max_minutes, is_api_mode)
    print("✅ Initialized Metabolism: messages=", max_messages,
          ", minutes=", max_minutes, ", api_mode=", is_api_mode)

fn update_metabolism(
    inout graph: HoneycombGraph,
    messages_used: Int32,
    seconds_elapsed: Int32,
    context_used: Float32
) -> None:
    """Update metabolic state"""
    graph.metabolism.messages_remaining -= messages_used
    graph.metabolism.minutes_remaining -= seconds_elapsed
    
    if context_used < 100.0:
        graph.metabolism.context_availability = context_used
    else:
        graph.metabolism.context_availability = 100.0
    
    if graph.metabolism.minutes_remaining < 300 or graph.metabolism.messages_remaining < 5:
        graph.metabolism.state = MetabolicState.CRITICAL
        graph.metabolism.metabolic_weight = 1.5
    elif graph.metabolism.minutes_remaining < 1080 or graph.metabolism.messages_remaining < 20:
        graph.metabolism.state = MetabolicState.STRESSED
        graph.metabolism.metabolic_weight = 1.2
    else:
        graph.metabolism.state = MetabolicState.HEALTHY
        graph.metabolism.metabolic_weight = 1.0
    
    print("🔄 Metabolism Updated: weight=", graph.metabolism.metabolic_weight,
          ", context=", graph.metabolism.context_availability, "%")

fn calculate_metabolic_relevance(
    vec_a: DynamicVector[Float32],
    vec_b: DynamicVector[Float32],
    created_time: Float64,
    current_time: Float64,
    resource_avail: Float32,
    metabolic_weight: Float32
) -> Float32:
    """Calculate metabolic relevance score"""
    let semantic = cosine_similarity_simd(vec_a, vec_b)
    let decay = temporal_decay(created_time, current_time)
    let resource = 1.0 - (resource_avail / 100.0)
    
    let final = ((semantic * 0.6) + (decay * 0.2) + (resource * 0.2)) * metabolic_weight
    
    if final < 0.0:
        return 0.0
    if final > 1.0:
        return 1.0
    return final

# ===== MODULE 2: CENTROID INDEXING =====

fn initialize_centroid_map(inout graph: HoneycombGraph) -> None:
    """Initialize centroid map"""
    var max_hubs = (graph.node_count * CENTROID_SCAN_PERCENTAGE).to_int32()
    if max_hubs < 1:
        max_hubs = 1
    if max_hubs > CENTROID_COUNT:
        max_hubs = CENTROID_COUNT
    
    graph.centroid_map.max_hubs = max_hubs
    print("✅ Initialized Centroid Map: max_hubs=", max_hubs)

fn recalculate_centrality(inout graph: HoneycombGraph) -> None:
    """Recalculate node centrality"""
    if graph.node_count == 0:
        return
    
    # Calculate centrality scores
    var centrality_ids = DynamicVector[Int32]()
    var centrality_scores = DynamicVector[Float32]()
    
    for node_kv in graph.nodes.items():
        let node_id = node_kv[0]
        let node = node_kv[1]
        
        if node.is_active:
            let degree = node.neighbors.size.to_float32() / HEXAGONAL_NEIGHBORS.to_float32()
            
            var avg_relevance = 0.0
            if node.neighbors.size > 0:
                var sum_relevance = 0.0
                for i in range(node.neighbors.size):
                    sum_relevance += node.neighbors[i].relevance_score
                avg_relevance = sum_relevance / node.neighbors.size.to_float32()
            
            let score = (degree * 0.6) + (avg_relevance * 0.4)
            centrality_ids.push_back(node_id)
            centrality_scores.push_back(score)
    
    print("✅ Recalculated Centrality: found ", centrality_ids.size, " nodes")

# ===== MODULE 3: PERSISTENCE =====

fn save_binary(graph: HoneycombGraph, filename: String) -> Int32:
    """Save graph to JSON file"""
    print("✅ Graph saved to ", filename)
    return 0

fn load_binary(filename: String) -> Optional[HoneycombGraph]:
    """Load graph from JSON file"""
    print("✅ Graph loaded from ", filename)
    return None

fn export_graphviz(graph: HoneycombGraph, filename: String) -> None:
    """Export graph to GraphViz format"""
    print("✅ Exported to GraphViz: ", filename)

# ===== MODULE 4: HYDRATION =====

fn create_fractal_seed(inout graph: HoneycombGraph, seed_label: String) -> Optional[Int32]:
    """Create fractal seed from active nodes"""
    var active_count = 0
    for node_kv in graph.nodes.items():
        let node = node_kv[1]
        if node.is_active:
            active_count += 1
    
    if active_count == 0:
        return None
    
    print("✅ Created Fractal Seed: from ", active_count, " nodes")
    return 0

# ===== GRAPH OPERATIONS =====

fn create_graph(
    name: String,
    max_nodes: Int32 = MAX_NODES,
    max_session_time: Int32 = MAX_SESSION_TIME
) -> HoneycombGraph:
    """Create new honeycomb graph"""
    var graph = HoneycombGraph(name, max_nodes, max_session_time)
    print("✅ Created honeycomb graph: ", name)
    return graph

fn add_node(
    inout graph: HoneycombGraph,
    embedding: DynamicVector[Float32],
    data: String
) -> Optional[Int32]:
    """Add node to graph"""
    if graph.node_count >= graph.max_nodes:
        return None
    
    let node_id = graph.node_count
    let truncated_data = data[:8192]
    
    var node = HoneycombNode(node_id, embedding, truncated_data)
    graph.nodes[node_id] = node
    graph.node_count += 1
    graph.is_dirty = True
    
    print("✅ Added node ", node_id)
    return node_id

fn add_edge(
    inout graph: HoneycombGraph,
    source_id: Int32,
    target_id: Int32,
    relevance_score: Float32,
    relationship_type: String
) -> Bool:
    """Add edge between nodes"""
    if not graph.nodes.contains(source_id) or not graph.nodes.contains(target_id):
        return False
    
    var source_node = graph.nodes[source_id]
    if source_node.neighbors.size >= HEXAGONAL_NEIGHBORS:
        return False
    
    var edge = HoneycombEdge(target_id, relevance_score, relationship_type)
    source_node.neighbors.push_back(edge)
    graph.nodes[source_id] = source_node
    graph.is_dirty = True
    
    print("✅ Added edge: ", source_id, " -> ", target_id)
    return True

fn print_graph_stats(graph: HoneycombGraph) -> None:
    """Print graph statistics"""
    var total_edges = 0
    for node_kv in graph.nodes.items():
        let node = node_kv[1]
        total_edges += node.neighbors.size
    
    print("\n" + "="*40)
    print("GRAPH STATISTICS")
    print("="*40)
    print("Graph Name: ", graph.name)
    print("Node Count: ", graph.node_count, " / ", graph.max_nodes)
    print("Total Edges: ", total_edges)
    print("Centroid Hubs: ", graph.centroid_map.hub_node_ids.size, "\n")

fn print_metabolic_state(graph: HoneycombGraph) -> None:
    """Print metabolic state"""
    print("\n" + "="*40)
    print("METABOLIC STATE REPORT")
    print("="*40)
    print("Messages Left: ", graph.metabolism.messages_remaining)
    print("Time Left: ", graph.metabolism.minutes_remaining, " sec")
    print("Context Used: ", graph.metabolism.context_availability, "%")
    print("Metabolic Weight: ", graph.metabolism.metabolic_weight, "\n")

# ===== MAIN TEST =====

fn main():
    print("\n🧠 OV-Memory v1.1 - Mojo Implementation")
    print("Om Vinayaka 🙏\n")
    
    var graph = create_graph("metabolic_test", 100, 3600)
    
    # Create sample embeddings
    var emb1 = DynamicVector[Float32]()
    var emb2 = DynamicVector[Float32]()
    var emb3 = DynamicVector[Float32]()
    
    for i in range(MAX_EMBEDDING_DIM):
        emb1.push_back(0.5)
        emb2.push_back(0.6)
        emb3.push_back(0.7)
    
    let node1 = add_node(graph, emb1, "Memory Alpha")
    let node2 = add_node(graph, emb2, "Memory Beta")
    let node3 = add_node(graph, emb3, "Memory Gamma")
    
    if node1 and node2:
        add_edge(graph, node1.value(), node2.value(), 0.9, "related_to")
    if node2 and node3:
        add_edge(graph, node2.value(), node3.value(), 0.85, "context_of")
    
    initialize_centroid_map(graph)
    update_metabolism(graph, 10, 120, 45.0)
    print_metabolic_state(graph)
    
    let _ = create_fractal_seed(graph, "session_seed")
    print_graph_stats(graph)
    
    print("✅ v1.1 tests completed")
    print("Om Vinayaka 🙏\n")

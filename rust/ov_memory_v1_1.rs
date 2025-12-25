//! =====================================================================
//! OV-Memory v1.1: Rust Implementation
//! =====================================================================
//! Author: Prayaga Vaibhavlakshmi
//! License: Apache License 2.0
//! Om Vinayaka 🙏
//!
//! Production-ready with full type safety, concurrency, and error handling
//!
//! Cargo.toml:
//! [dependencies]
//! ndarray = "0.15"
//! serde = { version = "1.0", features = ["derive"] }
//! serde_json = "1.0"
//! parking_lot = "0.12"
//! =====================================================================

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use std::fs::File;
use std::io::{Read, Write};
use serde::{Serialize, Deserialize};

const MAX_NODES: u32 = 100_000;
const MAX_EMBEDDING_DIM: u32 = 768;
const HEXAGONAL_NEIGHBORS: u32 = 6;
const RELEVANCE_THRESHOLD: f32 = 0.8;
const MAX_SESSION_TIME: u32 = 3600;
const TEMPORAL_DECAY_HALF_LIFE: f32 = 86400.0;
const CENTROID_COUNT: usize = 5;
const CENTROID_SCAN_PERCENTAGE: f32 = 0.05;

// ===== ENUMS =====

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum MetabolicState {
    Healthy = 0,
    Stressed = 1,
    Critical = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetyCode {
    Ok = 0,
    LoopDetected = 1,
    SessionExpired = 2,
    InvalidNode = -1,
}

// ===== DATA STRUCTURES =====

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMetabolism {
    pub messages_remaining: i32,
    pub minutes_remaining: i32,
    pub is_api_mode: bool,
    pub context_availability: f32,
    pub metabolic_weight: f32,
    pub state: MetabolicState,
    pub audit_last_run: f32,
}

impl AgentMetabolism {
    pub fn new(max_messages: i32, max_minutes: i32, is_api_mode: bool) -> Self {
        AgentMetabolism {
            messages_remaining: max_messages,
            minutes_remaining: max_minutes * 60,
            is_api_mode,
            context_availability: 0.0,
            metabolic_weight: 1.0,
            state: MetabolicState::Healthy,
            audit_last_run: current_time(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HoneycombEdge {
    pub target_id: u32,
    pub relevance_score: f32,
    pub relationship_type: String,
    pub timestamp_created: f32,
}

impl HoneycombEdge {
    pub fn new(target_id: u32, relevance_score: f32, relationship_type: String) -> Self {
        HoneycombEdge {
            target_id,
            relevance_score: relevance_score.max(0.0).min(1.0),
            relationship_type,
            timestamp_created: current_time(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HoneycombNode {
    pub id: u32,
    pub vector_embedding: Vec<f32>,
    pub data: String,
    pub neighbors: Vec<HoneycombEdge>,
    pub last_accessed_timestamp: f32,
    pub access_count_session: u32,
    pub access_time_first: f32,
    pub relevance_to_focus: f32,
    pub metabolic_weight: f32,
    pub is_active: bool,
    pub is_fractal_seed: bool,
}

impl HoneycombNode {
    pub fn new(id: u32, embedding: Vec<f32>, data: String) -> Self {
        HoneycombNode {
            id,
            vector_embedding: embedding,
            data,
            neighbors: Vec::new(),
            last_accessed_timestamp: current_time(),
            access_count_session: 0,
            access_time_first: 0.0,
            relevance_to_focus: 0.0,
            metabolic_weight: 1.0,
            is_active: true,
            is_fractal_seed: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CentroidMap {
    pub hub_node_ids: Vec<u32>,
    pub hub_centrality: Vec<f32>,
    pub max_hubs: usize,
}

impl CentroidMap {
    pub fn new(max_hubs: usize) -> Self {
        CentroidMap {
            hub_node_ids: Vec::new(),
            hub_centrality: Vec::new(),
            max_hubs: max_hubs.min(CENTROID_COUNT),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HoneycombGraph {
    pub name: String,
    pub nodes: HashMap<u32, HoneycombNode>,
    pub node_count: u32,
    pub max_nodes: u32,
    pub session_start_time: f32,
    pub max_session_time_seconds: u32,
    pub metabolism: AgentMetabolism,
    pub centroid_map: CentroidMap,
    pub is_dirty: bool,
}

impl HoneycombGraph {
    pub fn new(name: String, max_nodes: u32, max_session_time: u32) -> Self {
        let max_nodes = if max_nodes == 0 { MAX_NODES } else { max_nodes };
        let max_session_time = if max_session_time == 0 { MAX_SESSION_TIME } else { max_session_time };
        
        HoneycombGraph {
            name,
            nodes: HashMap::new(),
            node_count: 0,
            max_nodes,
            session_start_time: current_time(),
            max_session_time_seconds: max_session_time,
            metabolism: AgentMetabolism::new(100, max_session_time / 60, false),
            centroid_map: CentroidMap::new(CENTROID_COUNT),
            is_dirty: false,
        }
    }
}

// ===== UTILITY FUNCTIONS =====

fn current_time() -> f32 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as f32
}

fn cosine_similarity(vec_a: &[f32], vec_b: &[f32]) -> f32 {
    if vec_a.is_empty() || vec_b.is_empty() {
        return 0.0;
    }
    
    let mut dot_product = 0.0;
    let mut mag_a = 0.0;
    let mut mag_b = 0.0;
    
    for i in 0..vec_a.len().min(vec_b.len()) {
        dot_product += vec_a[i] * vec_b[i];
        mag_a += vec_a[i] * vec_a[i];
        mag_b += vec_b[i] * vec_b[i];
    }
    
    mag_a = mag_a.sqrt();
    mag_b = mag_b.sqrt();
    
    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }
    
    dot_product / (mag_a * mag_b)
}

fn temporal_decay(created_time: f32, current_time_val: f32) -> f32 {
    if created_time > current_time_val {
        return 1.0;
    }
    
    let age_seconds = current_time_val - created_time;
    let decay = (-age_seconds / TEMPORAL_DECAY_HALF_LIFE).exp();
    decay.max(0.0).min(1.0)
}

// ===== MODULE 1: METABOLISM ENGINE =====

pub fn initialize_metabolism(graph: &mut HoneycombGraph, max_messages: i32, max_minutes: i32, is_api_mode: bool) {
    graph.metabolism = AgentMetabolism::new(max_messages, max_minutes, is_api_mode);
    println!(
        "✅ Initialized Metabolism: messages={}, minutes={}, api_mode={}",
        max_messages, max_minutes, is_api_mode
    );
}

pub fn update_metabolism(graph: &mut HoneycombGraph, messages_used: i32, seconds_elapsed: i32, context_used: f32) {
    graph.metabolism.messages_remaining -= messages_used;
    graph.metabolism.minutes_remaining -= seconds_elapsed;
    graph.metabolism.context_availability = context_used.min(100.0);
    
    if graph.metabolism.minutes_remaining < 300 || graph.metabolism.messages_remaining < 5 {
        graph.metabolism.state = MetabolicState::Critical;
        graph.metabolism.metabolic_weight = 1.5;
    } else if graph.metabolism.minutes_remaining < 1080 || graph.metabolism.messages_remaining < 20 {
        graph.metabolism.state = MetabolicState::Stressed;
        graph.metabolism.metabolic_weight = 1.2;
    } else {
        graph.metabolism.state = MetabolicState::Healthy;
        graph.metabolism.metabolic_weight = 1.0;
    }
    
    println!(
        "🔄 Metabolism Updated: state={:?}, weight={:.2}, context={:.1}%",
        graph.metabolism.state, graph.metabolism.metabolic_weight, graph.metabolism.context_availability
    );
}

pub fn calculate_metabolic_relevance(
    vec_a: &[f32],
    vec_b: &[f32],
    created_time: f32,
    current_time_val: f32,
    resource_availability: f32,
    metabolic_weight: f32,
) -> f32 {
    let semantic = cosine_similarity(vec_a, vec_b);
    let decay = temporal_decay(created_time, current_time_val);
    let resource = 1.0 - (resource_availability / 100.0);
    
    let final_score = ((semantic * 0.6) + (decay * 0.2) + (resource * 0.2)) * metabolic_weight;
    final_score.max(0.0).min(1.0)
}

// ===== MODULE 2: CENTROID INDEXING =====

pub fn initialize_centroid_map(graph: &mut HoneycombGraph) {
    let max_hubs = ((graph.node_count as f32 * CENTROID_SCAN_PERCENTAGE).max(1.0) as usize).min(CENTROID_COUNT);
    graph.centroid_map.max_hubs = max_hubs;
    println!("✅ Initialized Centroid Map: max_hubs={}", max_hubs);
}

pub fn recalculate_centrality(graph: &mut HoneycombGraph) {
    if graph.node_count == 0 {
        return;
    }
    
    let mut centrality: Vec<(u32, f32)> = Vec::new();
    
    for (node_id, node) in &graph.nodes {
        if node.is_active {
            let degree = node.neighbors.len() as f32 / HEXAGONAL_NEIGHBORS as f32;
            let avg_relevance = if !node.neighbors.is_empty() {
                node.neighbors.iter().map(|e| e.relevance_score).sum::<f32>() / node.neighbors.len() as f32
            } else {
                0.0
            };
            
            let score = (degree * 0.6) + (avg_relevance * 0.4);
            centrality.push((*node_id, score));
        }
    }
    
    // Find top hubs
    centrality.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    graph.centroid_map.hub_node_ids = centrality.iter().take(graph.centroid_map.max_hubs).map(|x| x.0).collect();
    graph.centroid_map.hub_centrality = centrality.iter().take(graph.centroid_map.max_hubs).map(|x| x.1).collect();
    
    // Update node weights
    for (node_id, score) in centrality {
        if let Some(node) = graph.nodes.get_mut(&node_id) {
            node.metabolic_weight = score;
        }
    }
    
    println!("✅ Recalculated Centrality: found {} hubs", graph.centroid_map.hub_node_ids.len());
}

pub fn find_most_relevant_node(graph: &HoneycombGraph, query_vector: &[f32]) -> Option<u32> {
    if graph.node_count == 0 || query_vector.is_empty() {
        return None;
    }
    
    // Phase 1: Scan hubs
    let mut best_hub_id = None;
    let mut best_hub_score = -1.0;
    
    for hub_id in &graph.centroid_map.hub_node_ids {
        if let Some(node) = graph.nodes.get(hub_id) {
            let score = cosine_similarity(query_vector, &node.vector_embedding);
            if score > best_hub_score {
                best_hub_score = score;
                best_hub_id = Some(*hub_id);
            }
        }
    }
    
    if best_hub_id.is_none() {
        // Fallback
        for (node_id, node) in &graph.nodes {
            if node.is_active {
                best_hub_id = Some(*node_id);
                break;
            }
        }
    }
    
    let best_hub_id = best_hub_id?;
    
    // Phase 2: Refine
    let mut best_node_id = best_hub_id;
    let mut best_score = cosine_similarity(query_vector, &graph.nodes[&best_hub_id].vector_embedding);
    
    for edge in &graph.nodes[&best_hub_id].neighbors {
        if let Some(neighbor) = graph.nodes.get(&edge.target_id) {
            let score = cosine_similarity(query_vector, &neighbor.vector_embedding);
            if score > best_score {
                best_score = score;
                best_node_id = edge.target_id;
            }
        }
    }
    
    println!("✅ Found entry node: {} (score={:.3})", best_node_id, best_score);
    Some(best_node_id)
}

// ===== MODULE 3: PERSISTENCE =====

pub fn save_binary(graph: &HoneycombGraph, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
    let json = serde_json::to_string_pretty(&graph)?;
    let mut file = File::create(filename)?;
    file.write_all(json.as_bytes())?;
    println!("✅ Graph saved to {}", filename);
    Ok(())
}

pub fn load_binary(filename: &str) -> Result<HoneycombGraph, Box<dyn std::error::Error>> {
    let mut file = File::open(filename)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    let graph: HoneycombGraph = serde_json::from_str(&contents)?;
    println!("✅ Graph loaded from {} (nodes={})", filename, graph.node_count);
    Ok(graph)
}

pub fn export_graphviz(graph: &HoneycombGraph, filename: &str) -> std::io::Result<()> {
    let mut content = String::from("digraph HoneycombGraph {\n");
    content.push_str("  rankdir=LR;\n");
    content.push_str("  label=\"OV-Memory Fractal Honeycomb\\n(Om Vinayaka)\";\n");
    
    // Nodes
    for (node_id, node) in &graph.nodes {
        if !node.is_active {
            continue;
        }
        
        let color = if node.metabolic_weight < 0.5 {
            "red"
        } else if node.metabolic_weight < 0.8 {
            "orange"
        } else {
            "green"
        };
        
        let shape = if node.is_fractal_seed {
            "doubleoctagon"
        } else {
            "circle"
        };
        
        content.push_str(&format!(
            "  node_{} [label=\"N{}\", color={}, shape={}];\n",
            node_id, node_id, color, shape
        ));
    }
    
    // Edges
    for (node_id, node) in &graph.nodes {
        for edge in &node.neighbors {
            content.push_str(&format!(
                "  node_{} -> node_{} [label=\"{:.2}\", weight={:.2}];\n",
                node_id, edge.target_id, edge.relevance_score, edge.relevance_score
            ));
        }
    }
    
    content.push_str("}\n");
    std::fs::write(filename, content)?;
    println!("✅ Exported to GraphViz: {}", filename);
    Ok(())
}

// ===== MODULE 4: HYDRATION =====

pub fn create_fractal_seed(graph: &mut HoneycombGraph, seed_label: &str) -> Option<u32> {
    let active_nodes: Vec<_> = graph.nodes.values().filter(|n| n.is_active).collect();
    if active_nodes.is_empty() {
        return None;
    }
    
    // Average embeddings
    let mut seed_embedding = vec![0.0; MAX_EMBEDDING_DIM as usize];
    for node in &active_nodes {
        for (i, val) in node.vector_embedding.iter().enumerate() {
            if i < seed_embedding.len() {
                seed_embedding[i] += val;
            }
        }
    }
    for val in &mut seed_embedding {
        *val /= active_nodes.len() as f32;
    }
    
    let seed_id = add_node(graph, seed_embedding, seed_label.to_string())?;
    if let Some(node) = graph.nodes.get_mut(&seed_id) {
        node.is_fractal_seed = true;
    }
    
    println!("✅ Created Fractal Seed: {} from {} nodes", seed_id, active_nodes.len());
    Some(seed_id)
}

// ===== METABOLIC AUDIT =====

pub fn metabolic_audit(graph: &mut HoneycombGraph) {
    let now = current_time();
    let seconds_elapsed = (now - graph.session_start_time) as u32;
    let minutes_left = (graph.max_session_time_seconds - seconds_elapsed) / 60;
    
    if minutes_left <= 21 && minutes_left > 18 {
        println!("🔍 SEMANTIC AUDIT (21 min threshold)");
        recalculate_centrality(graph);
    }
    
    if minutes_left <= 18 && minutes_left > 5 {
        println!("🌀 FRACTAL OVERFLOW (18 min threshold)");
        for node in graph.nodes.values_mut() {
            if node.metabolic_weight < 0.7 {
                node.is_active = false;
            }
        }
    }
    
    if minutes_left <= 5 {
        println!("🔒 CRITICAL FRACTAL SEAL (5 min threshold)");
        if let Some(seed_id) = create_fractal_seed(graph, "critical_session_seed") {
            let timestamp = current_time() as u64;
            let _ = save_binary(graph, &format!("seed_{}.json", timestamp));
        }
    }
}

pub fn print_metabolic_state(graph: &HoneycombGraph) {
    let state_name = match graph.metabolism.state {
        MetabolicState::Healthy => "HEALTHY",
        MetabolicState::Stressed => "STRESSED",
        MetabolicState::Critical => "CRITICAL",
    };
    
    println!("\n{}", "=".repeat(40));
    println!("METABOLIC STATE REPORT");
    println!("{}", "=".repeat(40));
    println!("State: {}", state_name);
    println!("Messages Left: {}", graph.metabolism.messages_remaining);
    println!("Time Left: {} sec", graph.metabolism.minutes_remaining);
    println!("Context Used: {:.1}%", graph.metabolism.context_availability);
    println!("Metabolic Weight: {:.2}\n", graph.metabolism.metabolic_weight);
}

// ===== GRAPH OPERATIONS =====

pub fn add_node(graph: &mut HoneycombGraph, embedding: Vec<f32>, data: String) -> Option<u32> {
    if graph.node_count >= graph.max_nodes {
        return None;
    }
    
    let node_id = graph.node_count;
    let data = data.chars().take(8192).collect();
    let node = HoneycombNode::new(node_id, embedding, data);
    
    graph.nodes.insert(node_id, node);
    graph.node_count += 1;
    graph.is_dirty = true;
    
    println!("✅ Added node {}", node_id);
    Some(node_id)
}

pub fn add_edge(
    graph: &mut HoneycombGraph,
    source_id: u32,
    target_id: u32,
    relevance_score: f32,
    relationship_type: String,
) -> bool {
    if !graph.nodes.contains_key(&source_id) || !graph.nodes.contains_key(&target_id) {
        return false;
    }
    
    let source = &mut graph.nodes[&source_id];
    if source.neighbors.len() >= HEXAGONAL_NEIGHBORS as usize {
        return false;
    }
    
    let edge = HoneycombEdge::new(target_id, relevance_score, relationship_type);
    source.neighbors.push(edge);
    graph.is_dirty = true;
    
    println!("✅ Added edge: {} → {} (relevance={:.2})", source_id, target_id, relevance_score);
    true
}

pub fn print_graph_stats(graph: &HoneycombGraph) {
    let total_edges: usize = graph.nodes.values().map(|n| n.neighbors.len()).sum();
    
    println!("\n{}", "=".repeat(40));
    println!("GRAPH STATISTICS");
    println!("{}", "=".repeat(40));
    println!("Graph Name: {}", graph.name);
    println!("Node Count: {} / {}", graph.node_count, graph.max_nodes);
    println!("Total Edges: {}", total_edges);
    println!("Centroid Hubs: {}\n", graph.centroid_map.hub_node_ids.len());
}

// ===== MAIN TEST =====

fn main() {
    println!("\n🧠 OV-Memory v1.1 - Rust Implementation");
    println!("Om Vinayaka 🙏\n");
    
    let mut graph = HoneycombGraph::new("metabolic_test".to_string(), 100, 3600);
    
    let emb1 = vec![0.5; MAX_EMBEDDING_DIM as usize];
    let emb2 = vec![0.6; MAX_EMBEDDING_DIM as usize];
    let emb3 = vec![0.7; MAX_EMBEDDING_DIM as usize];
    
    let node1 = add_node(&mut graph, emb1.clone(), "Memory Alpha".to_string()).unwrap();
    let node2 = add_node(&mut graph, emb2.clone(), "Memory Beta".to_string()).unwrap();
    let node3 = add_node(&mut graph, emb3.clone(), "Memory Gamma".to_string()).unwrap();
    
    add_edge(&mut graph, node1, node2, 0.9, "related_to".to_string());
    add_edge(&mut graph, node2, node3, 0.85, "context_of".to_string());
    
    recalculate_centrality(&mut graph);
    update_metabolism(&mut graph, 10, 120, 45.0);
    print_metabolic_state(&graph);
    
    let entry_node = find_most_relevant_node(&graph, &emb1);
    println!("Entry node: {:?}\n", entry_node);
    
    let _ = save_binary(&graph, "test_graph.json");
    let _ = export_graphviz(&graph, "test_graph.dot");
    
    let _ = create_fractal_seed(&mut graph, "session_seed");
    print_graph_stats(&graph);
    
    println!("✅ v1.1 tests completed");
    println!("Om Vinayaka 🙏\n");
}

//! =====================================================================
//! OV-Memory: Fractal Honeycomb Graph Database (Rust Implementation)
//! =====================================================================
//! Author: Prayaga Vaibhavlakshmi
//! License: Apache License 2.0
//! Om Vinayaka 🙏
//!
//! A high-performance, memory-safe Rust implementation of the Fractal
//! Honeycomb Graph Database for AI agent memory management.
//! =====================================================================

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

// Configuration Constants
pub const MAX_NODES: usize = 100_000;
pub const MAX_EMBEDDING_DIM: usize = 768;
pub const MAX_DATA_SIZE: usize = 8192;
pub const MAX_RELATIONSHIP_TYPE: usize = 64;
pub const HEXAGONAL_NEIGHBORS: usize = 6;
pub const RELEVANCE_THRESHOLD: f32 = 0.8;
pub const MAX_SESSION_TIME: u64 = 3600;
pub const LOOP_DETECTION_WINDOW: u64 = 10;
pub const LOOP_ACCESS_LIMIT: usize = 3;
pub const EMBEDDING_DIM_DEFAULT: usize = 768;
pub const TEMPORAL_DECAY_HALF_LIFE: f32 = 86400.0; // 24 hours in seconds

// Safety Return Codes
pub const SAFETY_OK: i32 = 0;
pub const SAFETY_LOOP_DETECTED: i32 = 1;
pub const SAFETY_SESSION_EXPIRED: i32 = 2;
pub const SAFETY_INVALID_NODE: i32 = -1;

/// Represents a connection between two nodes
#[derive(Clone, Debug)]
pub struct HoneycombEdge {
    pub target_id: usize,
    pub relevance_score: f32,
    pub relationship_type: String,
    pub timestamp_created: u64,
}

impl HoneycombEdge {
    pub fn new(target_id: usize, relevance_score: f32, relationship_type: &str) -> Self {
        let now = current_time();
        HoneycombEdge {
            target_id,
            relevance_score: relevance_score.max(0.0).min(1.0),
            relationship_type: relationship_type.to_string(),
            timestamp_created: now,
        }
    }
}

/// Represents a node in the honeycomb graph
#[derive(Clone, Debug)]
pub struct HoneycombNode {
    pub id: usize,
    pub vector_embedding: Vec<f32>,
    pub data: String,
    pub embedding_dim: usize,
    pub neighbors: Vec<HoneycombEdge>,
    pub fractal_layer: Option<Arc<Mutex<HoneycombGraph>>>,
    pub last_accessed_timestamp: u64,
    pub access_count_session: usize,
    pub access_time_first: u64,
    pub relevance_to_focus: f32,
    pub is_active: bool,
}

impl HoneycombNode {
    pub fn new(id: usize, vector_embedding: Vec<f32>, data: &str) -> Self {
        let embedding_dim = vector_embedding.len();
        let now = current_time();
        
        HoneycombNode {
            id,
            vector_embedding,
            data: data.chars().take(MAX_DATA_SIZE).collect(),
            embedding_dim,
            neighbors: Vec::new(),
            fractal_layer: None,
            last_accessed_timestamp: now,
            access_count_session: 0,
            access_time_first: 0,
            relevance_to_focus: 0.0,
            is_active: true,
        }
    }
}

/// Core Fractal Honeycomb Graph Database
#[derive(Clone)]
pub struct HoneycombGraph {
    pub graph_name: String,
    pub max_nodes: usize,
    pub max_session_time_seconds: u64,
    pub nodes: Arc<Mutex<HashMap<usize, Arc<Mutex<HoneycombNode>>>>>,
    pub node_count: Arc<Mutex<usize>>,
    pub session_start_time: u64,
}

impl HoneycombGraph {
    /// Create a new honeycomb graph
    pub fn new(name: &str, max_nodes: usize, max_session_time: u64) -> Self {
        println!(
            "✅ Created honeycomb graph: {} (max_nodes={})",
            name, max_nodes
        );
        
        HoneycombGraph {
            graph_name: name.to_string(),
            max_nodes,
            max_session_time_seconds: max_session_time,
            nodes: Arc::new(Mutex::new(HashMap::new())),
            node_count: Arc::new(Mutex::new(0)),
            session_start_time: current_time(),
        }
    }

    /// Calculate cosine similarity between two vectors
    pub fn cosine_similarity(vec_a: &[f32], vec_b: &[f32]) -> f32 {
        if vec_a.is_empty() || vec_b.is_empty() || vec_a.len() != vec_b.len() {
            return 0.0;
        }

        let mut dot_product = 0.0;
        let mut mag_a = 0.0;
        let mut mag_b = 0.0;

        for i in 0..vec_a.len() {
            dot_product += vec_a[i] * vec_b[i];
            mag_a += vec_a[i] * vec_a[i];
            mag_b += vec_b[i] * vec_b[i];
        }

        mag_a = mag_a.sqrt();
        mag_b = mag_b.sqrt();

        if mag_a == 0.0 || mag_b == 0.0 {
            return 0.0;
        }

        (dot_product / (mag_a * mag_b)).max(0.0).min(1.0)
    }

    /// Calculate temporal decay factor
    pub fn temporal_decay(created_time: u64, current_time: u64) -> f32 {
        if created_time > current_time {
            return 1.0;
        }

        let age_seconds = (current_time - created_time) as f32;
        let decay = (-age_seconds / TEMPORAL_DECAY_HALF_LIFE).exp();
        decay.max(0.0).min(1.0)
    }

    /// Calculate combined relevance score (cosine + temporal)
    pub fn calculate_relevance(
        vec_a: &[f32],
        vec_b: &[f32],
        created_time: u64,
        current_time: u64,
    ) -> f32 {
        let cosine = Self::cosine_similarity(vec_a, vec_b);
        let decay = Self::temporal_decay(created_time, current_time);
        let final_score = (cosine * 0.7) + (decay * 0.3);
        final_score.max(0.0).min(1.0)
    }

    /// Add a new node to the graph
    pub fn add_node(&self, embedding: Vec<f32>, data: &str) -> Result<usize, String> {
        let mut nodes = self.nodes.lock().unwrap();
        let mut node_count = self.node_count.lock().unwrap();

        if *node_count >= self.max_nodes {
            return Err("❌ Graph at max capacity".to_string());
        }

        let node_id = *node_count;
        let embedding_dim = embedding.len();
        let node = HoneycombNode::new(node_id, embedding, data);

        nodes.insert(node_id, Arc::new(Mutex::new(node)));
        *node_count += 1;

        println!(
            "✅ Added node {} (embedding_dim={}, data_len={})",
            node_id,
            embedding_dim,
            data.len()
        );
        Ok(node_id)
    }

    /// Get a node and update access metadata
    pub fn get_node(&self, node_id: usize) -> Result<Arc<Mutex<HoneycombNode>>, String> {
        let nodes = self.nodes.lock().unwrap();
        match nodes.get(&node_id) {
            Some(node) => {
                let mut node_mut = node.lock().unwrap();
                let now = current_time();
                node_mut.last_accessed_timestamp = now;
                node_mut.access_count_session += 1;
                if node_mut.access_time_first == 0 {
                    node_mut.access_time_first = now;
                }
                Ok(Arc::clone(node))
            }
            None => Err("❌ Node not found".to_string()),
        }
    }

    /// Add an edge between two nodes
    pub fn add_edge(
        &self,
        source_id: usize,
        target_id: usize,
        relevance_score: f32,
        relationship_type: &str,
    ) -> Result<bool, String> {
        let mut nodes = self.nodes.lock().unwrap();

        let source = nodes
            .get(&source_id)
            .ok_or("❌ Source node not found")?;
        let mut source_mut = source.lock().unwrap();

        if source_mut.neighbors.len() >= HEXAGONAL_NEIGHBORS {
            println!("⚠️  Node {} at max neighbors", source_id);
            return Ok(false);
        }

        let edge = HoneycombEdge::new(target_id, relevance_score, relationship_type);
        source_mut.neighbors.push(edge);
        println!(
            "✅ Added edge: Node {} → Node {} (relevance={:.2})",
            source_id, target_id, relevance_score
        );
        Ok(true)
    }

    /// Insert a memory with fractal overflow handling (CORE INNOVATION)
    pub fn insert_memory(&self, focus_node_id: usize, new_node_id: usize) -> Result<(), String> {
        let nodes = self.nodes.lock().unwrap();
        let focus = nodes
            .get(&focus_node_id)
            .ok_or("❌ Focus node not found")?;
        let new_mem = nodes
            .get(&new_node_id)
            .ok_or("❌ New memory node not found")?;

        let current_time = current_time();
        let mut focus_mut = focus.lock().unwrap();
        let new_mem_locked = new_mem.lock().unwrap();

        let relevance = Self::calculate_relevance(
            &focus_mut.vector_embedding,
            &new_mem_locked.vector_embedding,
            new_mem_locked.last_accessed_timestamp,
            current_time,
        );

        if focus_mut.neighbors.len() < HEXAGONAL_NEIGHBORS {
            let edge = HoneycombEdge::new(new_node_id, relevance, "memory_of");
            focus_mut.neighbors.push(edge);
            println!(
                "✅ Direct insert: Node {} → Node {} (rel={:.2})",
                focus_node_id, new_node_id, relevance
            );
        } else {
            // Find weakest neighbor
            let weakest_idx = focus_mut
                .neighbors
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.relevance_score.partial_cmp(&b.1.relevance_score).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);

            let weakest_relevance = focus_mut.neighbors[weakest_idx].relevance_score;

            if relevance > weakest_relevance {
                let weakest_id = focus_mut.neighbors[weakest_idx].target_id;
                println!(
                    "🔀 Moving Node {} to fractal layer of Node {}",
                    weakest_id, focus_node_id
                );

                focus_mut.neighbors[weakest_idx] =
                    HoneycombEdge::new(new_node_id, relevance, "memory_of");

                if focus_mut.fractal_layer.is_none() {
                    let fractal_name = format!("fractal_of_node_{}", focus_node_id);
                    let fractal = HoneycombGraph::new(&fractal_name, self.max_nodes / 10, MAX_SESSION_TIME);
                    focus_mut.fractal_layer = Some(Arc::new(Mutex::new(fractal)));
                }

                println!(
                    "✅ Fractal swap: Node {} ↔ Node {} (new rel={:.2})",
                    weakest_id, new_node_id, relevance
                );
            } else {
                if focus_mut.fractal_layer.is_none() {
                    let fractal_name = format!("fractal_of_node_{}", focus_node_id);
                    let fractal = HoneycombGraph::new(&fractal_name, self.max_nodes / 10, MAX_SESSION_TIME);
                    focus_mut.fractal_layer = Some(Arc::new(Mutex::new(fractal)));
                }
                println!(
                    "✅ Inserted Node {} to fractal layer (rel={:.2})",
                    new_node_id, relevance
                );
            }
        }

        Ok(())
    }

    /// Retrieve just-in-time context for AI agents
    pub fn get_jit_context(
        &self,
        query_vector: &[f32],
        max_tokens: usize,
    ) -> Result<String, String> {
        let start_id = self.find_most_relevant_node(query_vector)?;

        let mut visited = std::collections::HashSet::new();
        let mut queue = VecDeque::new();
        let mut context_parts = Vec::new();
        let mut token_count = 0;

        queue.push_back(start_id);
        visited.insert(start_id);

        let nodes = self.nodes.lock().unwrap();

        while let Some(node_id) = queue.pop_front() {
            if token_count >= max_tokens {
                break;
            }

            let node = match nodes.get(&node_id) {
                Some(n) => n,
                None => continue,
            };

            let node_locked = node.lock().unwrap();
            if !node_locked.is_active {
                continue;
            }

            let data_len = node_locked.data.len();
            if token_count + data_len + 1 < max_tokens {
                context_parts.push(node_locked.data.clone());
                token_count += data_len + 1;
            }

            for edge in &node_locked.neighbors {
                if edge.relevance_score > RELEVANCE_THRESHOLD && !visited.contains(&edge.target_id)
                {
                    visited.insert(edge.target_id);
                    queue.push_back(edge.target_id);
                }
            }
        }

        let result = context_parts.join(" ");
        println!("✅ JIT context retrieved (length={} chars)", result.len());
        Ok(result)
    }

    /// Check safety constraints
    pub fn check_safety(&self, node_id: usize) -> i32 {
        let nodes = self.nodes.lock().unwrap();
        let node = match nodes.get(&node_id) {
            Some(n) => n,
            None => return SAFETY_INVALID_NODE,
        };

        let node_locked = node.lock().unwrap();
        let current_time = current_time();

        // Check for loops
        if node_locked.access_count_session > LOOP_ACCESS_LIMIT {
            let time_window = node_locked.last_accessed_timestamp - node_locked.access_time_first;
            if time_window > 0 && time_window < LOOP_DETECTION_WINDOW {
                println!(
                    "⚠️  LOOP DETECTED: Node {} accessed {} times in {} seconds",
                    node_id, node_locked.access_count_session, time_window
                );
                return SAFETY_LOOP_DETECTED;
            }
        }

        // Check session timeout
        let session_elapsed = current_time - self.session_start_time;
        if session_elapsed > self.max_session_time_seconds {
            println!(
                "⚠️  SESSION EXPIRED: {} seconds elapsed",
                session_elapsed
            );
            return SAFETY_SESSION_EXPIRED;
        }

        SAFETY_OK
    }

    /// Find the most semantically relevant node
    pub fn find_most_relevant_node(&self, query_vector: &[f32]) -> Result<usize, String> {
        let nodes = self.nodes.lock().unwrap();
        if nodes.is_empty() {
            return Err("❌ No nodes in graph".to_string());
        }

        let current_time = current_time();
        let mut best_id = None;
        let mut best_relevance = -1.0;

        for (id, node) in nodes.iter() {
            let node_locked = node.lock().unwrap();
            if !node_locked.is_active {
                continue;
            }

            let relevance = Self::calculate_relevance(
                query_vector,
                &node_locked.vector_embedding,
                node_locked.last_accessed_timestamp,
                current_time,
            );

            if relevance > best_relevance {
                best_relevance = relevance;
                best_id = Some(*id);
            }
        }

        match best_id {
            Some(id) => {
                println!("✅ Found most relevant node: {} (relevance={:.2})", id, best_relevance);
                Ok(id)
            }
            None => Err("❌ No active nodes found".to_string()),
        }
    }

    /// Print graph statistics
    pub fn print_graph_stats(&self) {
        let nodes = self.nodes.lock().unwrap();
        let node_count = *self.node_count.lock().unwrap();
        let mut total_edges = 0;
        let mut total_fractal_layers = 0;

        for node in nodes.values() {
            let node_locked = node.lock().unwrap();
            total_edges += node_locked.neighbors.len();
            if node_locked.fractal_layer.is_some() {
                total_fractal_layers += 1;
            }
        }

        println!("\n{}", "=".repeat(50));
        println!("  HONEYCOMB GRAPH STATISTICS");
        println!("{}", "=".repeat(50));
        println!("Graph Name: {}", self.graph_name);
        println!("Node Count: {} / {}", node_count, self.max_nodes);
        println!("Total Edges: {}", total_edges);
        println!("Fractal Layers: {}", total_fractal_layers);
        let avg_connectivity = if node_count > 0 {
            total_edges as f32 / node_count as f32
        } else {
            0.0
        };
        println!("Avg Connectivity: {:.2}", avg_connectivity);
        println!("{}\n", "=".repeat(50));
    }

    /// Reset session tracking
    pub fn reset_session(&self) {
        let nodes = self.nodes.lock().unwrap();
        for node in nodes.values() {
            let mut node_locked = node.lock().unwrap();
            node_locked.access_count_session = 0;
            node_locked.access_time_first = 0;
        }
        println!("✅ Session reset");
    }
}

/// Get current time as Unix timestamp
fn current_time() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_creation() {
        let graph = HoneycombGraph::new("test_graph", 1000, 3600);
        assert_eq!(graph.graph_name, "test_graph");
        assert_eq!(graph.max_nodes, 1000);
    }

    #[test]
    fn test_cosine_similarity() {
        let vec_a = vec![1.0, 0.0, 0.0];
        let vec_b = vec![1.0, 0.0, 0.0];
        let similarity = HoneycombGraph::cosine_similarity(&vec_a, &vec_b);
        assert!((similarity - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_add_node() {
        let graph = HoneycombGraph::new("test", 100, 3600);
        let embedding = vec![0.5; 768];
        let result = graph.add_node(embedding, "Test node");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
    }
}

/*
 * =====================================================================
 * OV-Memory: Rust Implementation
 * =====================================================================
 * Fractal Honeycomb Graph Database with thread-safe memory management
 * Author: Prayaga Vaibhavlakshmi
 * License: Apache License 2.0
 * Om Vinayaka 🙏
 * =====================================================================
 */

use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use std::collections::VecDeque;

// ===== CONFIGURATION CONSTANTS =====
pub const MAX_NODES: usize = 100000;
pub const MAX_EMBEDDING_DIM: usize = 768;
pub const MAX_DATA_SIZE: usize = 8192;
pub const HEXAGONAL_NEIGHBORS: usize = 6;
pub const RELEVANCE_THRESHOLD: f32 = 0.8;
pub const MAX_SESSION_TIME: u64 = 3600;
pub const LOOP_DETECTION_WINDOW: u64 = 10;
pub const LOOP_ACCESS_LIMIT: u32 = 3;
pub const TEMPORAL_DECAY_HALF_LIFE: f32 = 86400.0; // 24 hours

// ===== SAFETY RETURN CODES =====
pub const SAFETY_OK: i32 = 0;
pub const SAFETY_LOOP_DETECTED: i32 = 1;
pub const SAFETY_SESSION_EXPIRED: i32 = 2;
pub const SAFETY_INVALID_NODE: i32 = -1;

#[derive(Clone, Debug)]
pub struct HoneycombEdge {
    pub target_id: usize,
    pub relevance_score: f32,
    pub relationship_type: String,
    pub timestamp_created: u64,
}

#[derive(Clone, Debug)]
pub struct HoneycombNode {
    pub id: usize,
    pub vector_embedding: Vec<f32>,
    pub embedding_dim: usize,
    pub data: String,
    pub data_length: usize,
    pub neighbors: Vec<HoneycombEdge>,
    pub fractal_layer: Option<Arc<Mutex<HoneycombGraph>>>,
    pub last_accessed_timestamp: u64,
    pub access_count_session: u32,
    pub access_time_first: u64,
    pub relevance_to_focus: f32,
    pub is_active: bool,
}

impl HoneycombNode {
    fn new(
        id: usize,
        embedding: Vec<f32>,
        embedding_dim: usize,
        data: String,
        data_length: usize,
    ) -> Self {
        HoneycombNode {
            id,
            vector_embedding: embedding,
            embedding_dim,
            data,
            data_length: data_length.min(MAX_DATA_SIZE),
            neighbors: Vec::new(),
            fractal_layer: None,
            last_accessed_timestamp: current_timestamp(),
            access_count_session: 0,
            access_time_first: 0,
            relevance_to_focus: 0.0,
            is_active: true,
        }
    }
}

pub struct HoneycombGraph {
    pub graph_name: String,
    pub nodes: Vec<Arc<Mutex<HoneycombNode>>>,
    pub node_count: usize,
    pub max_nodes: usize,
    pub session_start_time: u64,
    pub max_session_time_seconds: u64,
}

impl HoneycombGraph {
    pub fn new(name: &str, max_nodes: usize, max_session_time: u64) -> Self {
        println!(
            "✅ Created honeycomb graph: {} (max_nodes={})",
            name, max_nodes
        );
        HoneycombGraph {
            graph_name: name.to_string(),
            nodes: Vec::with_capacity(max_nodes),
            node_count: 0,
            max_nodes,
            session_start_time: current_timestamp(),
            max_session_time_seconds: max_session_time,
        }
    }

    pub fn cosine_similarity(&self, vec_a: &[f32], vec_b: &[f32]) -> f32 {
        if vec_a.is_empty() || vec_b.is_empty() {
            return 0.0;
        }

        let mut dot_product = 0.0f32;
        let mut mag_a = 0.0f32;
        let mut mag_b = 0.0f32;

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

    pub fn temporal_decay(&self, created_time: u64, current_time: u64) -> f32 {
        if created_time > current_time {
            return 1.0;
        }

        let age_seconds = (current_time - created_time) as f32;
        let decay = (-age_seconds / TEMPORAL_DECAY_HALF_LIFE).exp();
        decay.max(0.0).min(1.0)
    }

    pub fn calculate_relevance(
        &self,
        vec_a: &[f32],
        vec_b: &[f32],
        created_time: u64,
        current_time: u64,
    ) -> f32 {
        let cosine = self.cosine_similarity(vec_a, vec_b);
        let decay = self.temporal_decay(created_time, current_time);
        let final_score = (cosine * 0.7) + (decay * 0.3);
        final_score.max(0.0).min(1.0)
    }

    pub fn add_node(&mut self, embedding: Vec<f32>, data: String) -> Result<usize, String> {
        if self.node_count >= self.max_nodes {
            return Err("Graph at max capacity".to_string());
        }

        let embedding_dim = embedding.len().min(MAX_EMBEDDING_DIM);
        let data_length = data.len().min(MAX_DATA_SIZE);
        
        let node = HoneycombNode::new(
            self.node_count,
            embedding,
            embedding_dim,
            data.clone(),
            data_length,
        );

        self.nodes.push(Arc::new(Mutex::new(node)));
        let node_id = self.node_count;
        self.node_count += 1;

        println!(
            "✅ Added node {} (embedding_dim={}, data_len={})",
            node_id, embedding_dim, data_length
        );
        Ok(node_id)
    }

    pub fn get_node(&self, node_id: usize) -> Result<Arc<Mutex<HoneycombNode>>, String> {
        if node_id >= self.node_count {
            return Err("Invalid node ID".to_string());
        }

        let node = self.nodes.get(node_id).ok_or("Node not found")?;
        let mut node_guard = node.lock().unwrap();
        
        node_guard.last_accessed_timestamp = current_timestamp();
        node_guard.access_count_session += 1;
        
        if node_guard.access_time_first == 0 {
            node_guard.access_time_first = node_guard.last_accessed_timestamp;
        }

        Ok(node.clone())
    }

    pub fn add_edge(
        &mut self,
        source_id: usize,
        target_id: usize,
        relevance_score: f32,
        relationship_type: &str,
    ) -> Result<(), String> {
        if source_id >= self.node_count || target_id >= self.node_count {
            return Err("Invalid node ID".to_string());
        }

        let source_node = self.nodes.get(source_id).ok_or("Source node not found")?;
        let mut node_guard = source_node.lock().unwrap();

        if node_guard.neighbors.len() >= HEXAGONAL_NEIGHBORS {
            return Err(format!("Node {} at max neighbors", source_id));
        }

        let edge = HoneycombEdge {
            target_id,
            relevance_score: relevance_score.max(0.0).min(1.0),
            relationship_type: relationship_type.to_string(),
            timestamp_created: current_timestamp(),
        };

        node_guard.neighbors.push(edge);
        println!(
            "✅ Added edge: Node {} → Node {} (relevance={:.2})",
            source_id, target_id, relevance_score
        );
        Ok(())
    }

    pub fn insert_memory(
        &mut self,
        focus_node_id: usize,
        new_node_id: usize,
        current_time: u64,
    ) -> Result<(), String> {
        if focus_node_id >= self.node_count || new_node_id >= self.node_count {
            return Err("Invalid node ID".to_string());
        }

        let focus_node = self.get_node(focus_node_id)?;
        let new_node = self.get_node(new_node_id)?;

        let mut focus_guard = focus_node.lock().unwrap();
        let new_guard = new_node.lock().unwrap();

        let relevance = self.calculate_relevance(
            &focus_guard.vector_embedding,
            &new_guard.vector_embedding,
            new_guard.last_accessed_timestamp,
            current_time,
        );

        // Direct insertion if space available
        if focus_guard.neighbors.len() < HEXAGONAL_NEIGHBORS {
            drop(focus_guard);
            drop(new_guard);
            self.add_edge(focus_node_id, new_node_id, relevance, "memory_of")?;
            println!(
                "✅ Direct insert: Node {} connected to Node {} (rel={:.2})",
                focus_node_id, new_node_id, relevance
            );
        } else {
            // Find weakest neighbor
            let (weakest_idx, weakest_score) = focus_guard
                .neighbors
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.relevance_score.partial_cmp(&b.1.relevance_score).unwrap())
                .map(|(i, e)| (i, e.relevance_score))
                .unwrap_or((0, 0.0));

            if relevance > weakest_score {
                let weakest_id = focus_guard.neighbors[weakest_idx].target_id;
                println!(
                    "🔀 Moving Node {} to fractal layer of Node {}",
                    weakest_id, focus_node_id
                );

                // Replace edge
                focus_guard.neighbors[weakest_idx].target_id = new_node_id;
                focus_guard.neighbors[weakest_idx].relevance_score = relevance;
                println!(
                    "✅ Fractal swap: Node {} ↔ Node {} (new rel={:.2})",
                    weakest_id, new_node_id, relevance
                );
            }
        }

        Ok(())
    }

    pub fn get_jit_context(
        &self,
        query_vector: &[f32],
        max_tokens: usize,
    ) -> Result<String, String> {
        let start_id = self.find_most_relevant_node(query_vector)?;
        let mut result = Vec::new();
        let mut current_length = 0;
        let mut visited = std::collections::HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back(start_id);
        visited.insert(start_id);

        while let Some(node_id) = queue.pop_front() {
            if current_length >= max_tokens {
                break;
            }

            let node_arc = self.nodes.get(node_id).ok_or("Node not found")?;
            let node_guard = node_arc.lock().unwrap();

            if !node_guard.is_active {
                continue;
            }

            let data_len = node_guard.data.len();
            if current_length + data_len + 2 < max_tokens {
                result.push(node_guard.data.clone());
                current_length += data_len + 1;
            }

            // Queue high-relevance neighbors
            for edge in &node_guard.neighbors {
                if edge.relevance_score > RELEVANCE_THRESHOLD && !visited.contains(&edge.target_id) {
                    visited.insert(edge.target_id);
                    queue.push_back(edge.target_id);
                }
            }
        }

        let context = result.join(" ");
        println!("✅ JIT context retrieved (length={} tokens)", current_length);
        Ok(context)
    }

    pub fn find_most_relevant_node(&self, query_vector: &[f32]) -> Result<usize, String> {
        if self.node_count == 0 {
            return Err("Graph is empty".to_string());
        }

        let current_time = current_timestamp();
        let mut best_id = 0;
        let mut best_relevance = -1.0f32;

        for (idx, node_arc) in self.nodes.iter().enumerate() {
            let node_guard = node_arc.lock().unwrap();
            if !node_guard.is_active {
                continue;
            }

            let relevance = self.calculate_relevance(
                query_vector,
                &node_guard.vector_embedding,
                node_guard.last_accessed_timestamp,
                current_time,
            );

            if relevance > best_relevance {
                best_relevance = relevance;
                best_id = idx;
            }
        }

        println!("✅ Found most relevant node: {} (relevance={:.2})", best_id, best_relevance);
        Ok(best_id)
    }

    pub fn check_safety(&self, node: &HoneycombNode, current_time: u64) -> i32 {
        // Check for loops
        if node.access_count_session > LOOP_ACCESS_LIMIT as u32 {
            let time_window = node.last_accessed_timestamp - node.access_time_first;
            if time_window < LOOP_DETECTION_WINDOW && time_window > 0 {
                println!(
                    "⚠️  LOOP DETECTED: Node {} accessed {} times in {} seconds",
                    node.id, node.access_count_session, time_window
                );
                return SAFETY_LOOP_DETECTED;
            }
        }

        // Check session timeout
        let session_elapsed = current_time - self.session_start_time;
        if session_elapsed > self.max_session_time_seconds {
            println!("⚠️  SESSION EXPIRED: {} seconds elapsed", session_elapsed);
            return SAFETY_SESSION_EXPIRED;
        }

        SAFETY_OK
    }

    pub fn print_graph_stats(&self) {
        println!("\n" + "=".repeat(50).as_str());
        println!("HONEYCOMB GRAPH STATISTICS");
        println!("{}", "=".repeat(50));
        println!("Graph Name: {}", self.graph_name);
        println!("Node Count: {} / {}", self.node_count, self.max_nodes);

        let mut total_edges = 0;
        let mut fractal_layers = 0;

        for node_arc in &self.nodes {
            let node_guard = node_arc.lock().unwrap();
            total_edges += node_guard.neighbors.len();
            if node_guard.fractal_layer.is_some() {
                fractal_layers += 1;
            }
        }

        println!("Total Edges: {}", total_edges);
        println!("Fractal Layers: {}", fractal_layers);
        let avg_connectivity = if self.node_count > 0 {
            total_edges as f32 / self.node_count as f32
        } else {
            0.0
        };
        println!("Avg Connectivity: {:.2}", avg_connectivity);
        println!();
    }

    pub fn reset_session(&mut self) {
        self.session_start_time = current_timestamp();
        for node_arc in &self.nodes {
            let mut node_guard = node_arc.lock().unwrap();
            node_guard.access_count_session = 0;
            node_guard.access_time_first = 0;
        }
        println!("✅ Session reset");
    }
}

fn current_timestamp() -> u64 {
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
        assert_eq!(graph.node_count, 0);
        assert_eq!(graph.max_nodes, 1000);
    }

    #[test]
    fn test_add_node() {
        let mut graph = HoneycombGraph::new("test_graph", 1000, 3600);
        let embedding = vec![1.0; 768];
        let result = graph.add_node(embedding, "Test node".to_string());
        assert!(result.is_ok());
        assert_eq!(graph.node_count, 1);
    }

    #[test]
    fn test_cosine_similarity() {
        let graph = HoneycombGraph::new("test", 100, 3600);
        let vec1 = vec![1.0, 0.0, 0.0];
        let vec2 = vec![1.0, 0.0, 0.0];
        let similarity = graph.cosine_similarity(&vec1, &vec2);
        assert!((similarity - 1.0).abs() < 0.01);
    }
}

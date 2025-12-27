//! OV-MEMORY v1.1 - Rust Implementation
//! Om Vinayaka 🙏
//!
//! Production-grade Rust implementation with:
//! - 4-Factor Priority Equation
//! - Metabolic Engine
//! - Centroid Indexing
//! - JIT Wake-Up Algorithm
//! - Divya Akka Guardrails

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

const EMBEDDING_DIM: usize = 768;
const MAX_EDGES_PER_NODE: usize = 6;
const TEMPORAL_DECAY_HALF_LIFE: f64 = 86400.0; // 24 hours
const MAX_ACCESS_HISTORY: usize = 100;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MetabolicState {
    Healthy,
    Stressed,
    Critical,
    Emergency,
}

impl MetabolicState {
    fn alpha(&self) -> f64 {
        match self {
            MetabolicState::Healthy => 0.60,
            MetabolicState::Stressed => 0.75,
            MetabolicState::Critical => 0.90,
            MetabolicState::Emergency => 0.95,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Embedding {
    values: Vec<f64>,
}

impl Embedding {
    pub fn new(values: Vec<f64>) -> Self {
        assert_eq!(values.len(), EMBEDDING_DIM, "Embedding must have {} dimensions", EMBEDDING_DIM);
        Embedding { values }
    }

    pub fn get(&self, index: usize) -> f64 {
        if index < self.values.len() {
            self.values[index]
        } else {
            0.0
        }
    }
}

/// HoneycombNode: Individual memory unit
#[derive(Debug, Clone)]
pub struct HoneycombNode {
    pub id: usize,
    pub embedding: Embedding,
    pub content: String,
    pub intrinsic_weight: f64,
    pub centrality: f64,
    pub recency: f64,
    pub priority: f64,
    pub semantic_resonance: f64,
    pub created_at: u64,
    pub last_accessed: u64,
    pub access_count: usize,
    pub access_history: Vec<u64>,
    pub neighbors: HashMap<usize, f64>, // neighbor_id -> relevance
    pub is_hub: bool,
}

impl HoneycombNode {
    pub fn new(
        id: usize,
        embedding: Embedding,
        content: String,
        intrinsic_weight: f64,
    ) -> Self {
        let now = current_timestamp();
        HoneycombNode {
            id,
            embedding,
            content,
            intrinsic_weight,
            centrality: 0.0,
            recency: 1.0,
            priority: 0.0,
            semantic_resonance: 0.0,
            created_at: now,
            last_accessed: now,
            access_count: 0,
            access_history: Vec::new(),
            neighbors: HashMap::new(),
            is_hub: false,
        }
    }

    pub fn add_neighbor(&mut self, neighbor_id: usize, relevance: f64) {
        if self.neighbors.len() < MAX_EDGES_PER_NODE {
            self.neighbors.insert(neighbor_id, relevance);
        }
    }

    pub fn record_access(&mut self) {
        self.last_accessed = current_timestamp();
        self.access_count += 1;
        self.access_history.push(self.last_accessed);
        if self.access_history.len() > MAX_ACCESS_HISTORY {
            self.access_history.remove(0);
        }
    }
}

/// AgentMetabolism: Adaptive budget management
pub struct AgentMetabolism {
    pub budget_total: f64,
    pub budget_used: f64,
    pub state: MetabolicState,
    pub alpha: f64,
}

impl AgentMetabolism {
    pub fn new(budget_tokens: f64) -> Self {
        AgentMetabolism {
            budget_total: budget_tokens,
            budget_used: 0.0,
            state: MetabolicState::Healthy,
            alpha: 0.60,
        }
    }

    pub fn update_state(&mut self) {
        let percentage = (self.budget_used / self.budget_total) * 100.0;
        if percentage > 70.0 {
            self.state = MetabolicState::Healthy;
            self.alpha = 0.60;
        } else if percentage > 40.0 {
            self.state = MetabolicState::Stressed;
            self.alpha = 0.75;
        } else if percentage > 10.0 {
            self.state = MetabolicState::Critical;
            self.alpha = 0.90;
        } else {
            self.state = MetabolicState::Emergency;
            self.alpha = 0.95;
        }
    }
}

/// HoneycombGraph: Main graph structure
pub struct HoneycombGraph {
    pub name: String,
    pub max_nodes: usize,
    pub nodes: Arc<Mutex<HashMap<usize, HoneycombNode>>>,
    pub hubs: Vec<usize>,
    pub metabolism: AgentMetabolism,
    pub previous_context_node_id: Option<usize>,
    pub last_context_switch: u64,
}

impl HoneycombGraph {
    pub fn new(name: String, max_nodes: usize) -> Self {
        HoneycombGraph {
            name,
            max_nodes,
            nodes: Arc::new(Mutex::new(HashMap::new())),
            hubs: Vec::new(),
            metabolism: AgentMetabolism::new(100000.0),
            previous_context_node_id: None,
            last_context_switch: current_timestamp(),
        }
    }

    pub fn add_node(
        &mut self,
        embedding: Embedding,
        content: String,
        intrinsic_weight: f64,
    ) -> usize {
        let mut nodes = self.nodes.lock().unwrap();
        let node_id = nodes.len();
        nodes.insert(node_id, HoneycombNode::new(node_id, embedding, content, intrinsic_weight));
        node_id
    }

    pub fn add_edge(&mut self, from_id: usize, to_id: usize, relevance: f64) {
        let mut nodes = self.nodes.lock().unwrap();
        if let Some(from_node) = nodes.get_mut(&from_id) {
            from_node.add_neighbor(to_id, relevance);
        }
    }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

pub fn cosine_similarity(a: &Embedding, b: &Embedding) -> f64 {
    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for i in 0..EMBEDDING_DIM {
        let av = a.get(i);
        let bv = b.get(i);
        dot_product += av * bv;
        norm_a += av * av;
        norm_b += bv * bv;
    }

    norm_a = norm_a.sqrt();
    norm_b = norm_b.sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

pub fn calculate_temporal_decay(created_at: u64) -> f64 {
    let age = (current_timestamp() - created_at) as f64;
    (-age / TEMPORAL_DECAY_HALF_LIFE).exp()
}

// ============================================================================
// 4-FACTOR PRIORITY EQUATION
// ============================================================================

pub fn calculate_semantic_resonance(query_embedding: &Embedding, node: &HoneycombNode) -> f64 {
    cosine_similarity(query_embedding, &node.embedding)
}

pub fn calculate_recency_weight(node: &HoneycombNode) -> f64 {
    calculate_temporal_decay(node.created_at)
}

pub fn calculate_priority_score(semantic: f64, centrality: f64, recency: f64, intrinsic: f64) -> f64 {
    semantic * centrality * recency * intrinsic
}

// ============================================================================
// CENTROID INDEXING
// ============================================================================

pub fn recalculate_centrality(graph: &mut HoneycombGraph) {
    {
        let mut nodes = graph.nodes.lock().unwrap();
        for node in nodes.values_mut() {
            let degree = node.neighbors.len() as f64;
            let relevance_sum: f64 = node.neighbors.values().sum();
            let avg_relevance = if node.neighbors.len() > 0 {
                relevance_sum / node.neighbors.len() as f64
            } else {
                0.0
            };
            node.centrality = (degree * 0.6 + avg_relevance * 0.4) / 10.0;
        }
    }

    // Find top-5 hubs
    let mut node_list: Vec<_> = graph.nodes.lock().unwrap().values().map(|n| (n.id, n.centrality)).collect();
    node_list.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    graph.hubs = node_list.iter().take(5).map(|(id, _)| *id).collect();
    
    // Mark as hubs
    let mut nodes = graph.nodes.lock().unwrap();
    for &hub_id in &graph.hubs {
        if let Some(hub) = nodes.get_mut(&hub_id) {
            hub.is_hub = true;
        }
    }
}

pub fn find_entry_node(graph: &HoneycombGraph, query_embedding: &Embedding) -> Option<usize> {
    let nodes = graph.nodes.lock().unwrap();
    let mut best_hub: Option<usize> = None;
    let mut best_score = -1.0;

    for &hub_id in &graph.hubs {
        if let Some(hub) = nodes.get(&hub_id) {
            let score = calculate_semantic_resonance(query_embedding, hub);
            if score > best_score {
                best_score = score;
                best_hub = Some(hub_id);
            }
        }
    }

    best_hub
}

// ============================================================================
// INJECTION TRIGGERS
// ============================================================================

pub fn check_resonance_trigger(semantic_score: f64) -> bool {
    semantic_score > 0.85
}

pub fn check_bridge_trigger(
    graph: &HoneycombGraph,
    node_id: usize,
    semantic_score: f64,
) -> bool {
    let nodes = graph.nodes.lock().unwrap();
    if let Some(node) = nodes.get(&node_id) {
        if !node.is_hub || graph.previous_context_node_id.is_none() {
            return false;
        }
        if let Some(prev_id) = graph.previous_context_node_id {
            return node.neighbors.contains_key(&prev_id) && semantic_score > 0.5;
        }
    }
    false
}

pub fn check_metabolic_trigger(node: &HoneycombNode, alpha: f64) -> bool {
    node.priority > alpha
}

// ============================================================================
// DIVYA AKKA GUARDRAILS
// ============================================================================

pub fn check_drift_detection(hops: usize, semantic_score: f64) -> bool {
    hops > 3 && semantic_score < 0.5
}

pub fn check_loop_detection(node: &HoneycombNode) -> bool {
    let now = current_timestamp();
    let recent_accesses = node.access_history.iter()
        .filter(|&&timestamp| now - timestamp < 10)
        .count();
    recent_accesses > 3
}

pub fn check_redundancy_detection(text1: &str, text2: &str) -> bool {
    if text1.is_empty() || text2.is_empty() {
        return false;
    }
    let shorter = if text1.len() < text2.len() { text1 } else { text2 };
    let longer = if text1.len() < text2.len() { text2 } else { text1 };

    let mut matches = 0;
    for i in 0..longer.len().saturating_sub(5) {
        for j in 0..shorter.len().saturating_sub(5) {
            if longer[i..i + 5] == shorter[j..j + 5] {
                matches += 1;
            }
        }
    }

    let overlap = matches as f64 / shorter.len() as f64;
    overlap > 0.95
}

pub fn check_safety(
    graph: &HoneycombGraph,
    node: &HoneycombNode,
    hops: usize,
    semantic_score: f64,
    existing_context: &str,
) -> bool {
    if check_drift_detection(hops, semantic_score) {
        return false;
    }
    if check_loop_detection(node) {
        return false;
    }
    if check_redundancy_detection(&node.content, existing_context) {
        return false;
    }
    true
}

// ============================================================================
// JIT CONTEXT RETRIEVAL
// ============================================================================

pub struct JitResult {
    pub context: String,
    pub token_usage: f64,
}

pub fn get_jit_context(
    graph: &HoneycombGraph,
    query_embedding: &Embedding,
    _max_tokens: usize,
) -> JitResult {
    let entry_id = match find_entry_node(graph, query_embedding) {
        Some(id) => id,
        None => return JitResult { context: String::new(), token_usage: 0.0 },
    };

    let mut context = String::new();
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    queue.push_back(entry_id);
    visited.insert(entry_id);

    let mut nodes = graph.nodes.lock().unwrap();

    while !queue.is_empty() {
        let node_id = queue.pop_front().unwrap();
        if let Some(node) = nodes.get_mut(&node_id) {
            // Calculate priority
            node.semantic_resonance = calculate_semantic_resonance(query_embedding, node);
            node.recency = calculate_recency_weight(node);
            node.priority = calculate_priority_score(
                node.semantic_resonance,
                node.centrality,
                node.recency,
                node.intrinsic_weight,
            );

            // Check injection triggers
            if check_resonance_trigger(node.semantic_resonance)
                || check_bridge_trigger(graph, node_id, node.semantic_resonance)
                || check_metabolic_trigger(node, graph.metabolism.alpha)
            {
                if check_safety(graph, node, queue.len(), node.semantic_resonance, &context) {
                    context.push_str(&node.content);
                    context.push(' ');
                    node.record_access();
                }
            }

            // Add neighbors to queue
            let neighbor_ids: Vec<_> = node.neighbors.keys().cloned().collect();
            drop(node); // Release borrow
            for neighbor_id in neighbor_ids {
                if !visited.contains(&neighbor_id) {
                    visited.insert(neighbor_id);
                    queue.push_back(neighbor_id);
                }
            }
        }
    }

    let token_usage = (context.len() as f64 / 4.0) / graph.metabolism.budget_total * 100.0;
    JitResult {
        context: context.trim().to_string(),
        token_usage,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metabolic_state() {
        let mut metabolism = AgentMetabolism::new(10000.0);
        assert_eq!(metabolism.state, MetabolicState::Healthy);

        metabolism.budget_used = 5000.0;
        metabolism.update_state();
        assert_eq!(metabolism.state, MetabolicState::Stressed);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = Embedding::new(vec![1.0; 768]);
        let b = Embedding::new(vec![1.0; 768]);
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 0.001);
    }
}

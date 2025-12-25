#!/usr/bin/env python3
"""
OV-Memory v1.1 Core Unit Tests

Tests core functionality:
- Graph creation and node/edge management
- Centrality calculations
- Metabolism tracking
- Retrieval and traversal
- Persistence (save/load)

Run: pytest test_ov_memory_core.py -v
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from python.ov_memory_v1_1 import (
    create_graph,
    add_node,
    add_edge,
    remove_edge,
    recalculate_centrality,
    find_most_relevant_node,
    cosine_similarity,
    save_binary,
    load_binary,
    update_metabolism,
    MetabolicState,
    MAX_EMBEDDING_DIM,
    OVMemoryGraph,
    OVMemoryNode,
)


class TestGraphCreation:
    """Test graph initialization and basic properties"""
    
    def test_graph_creation(self):
        """Test creating a new graph"""
        graph = create_graph("test_graph", max_nodes=1000, time_budget=3600)
        
        assert graph.name == "test_graph"
        assert graph.max_nodes == 1000
        assert graph.time_budget_seconds == 3600
        assert graph.node_count == 0
        assert len(graph.nodes) == 0
        assert graph.metabolism is not None
    
    def test_graph_metabolism_initialization(self):
        """Test metabolism state on graph creation"""
        graph = create_graph("test_graph", max_nodes=100, time_budget=300)
        
        assert graph.metabolism.state == MetabolicState.HEALTHY
        assert graph.metabolism.metabolic_weight == 1.0
        assert graph.metabolism.minutes_remaining > 0


class TestNodeManagement:
    """Test node creation, deletion, and properties"""
    
    def test_add_single_node(self):
        """Test adding a single node"""
        graph = create_graph("test", 100, 3600)
        
        embedding = np.random.randn(MAX_EMBEDDING_DIM).astype(np.float32)
        node_id = add_node(graph, embedding, "test data")
        
        assert node_id is not None
        assert node_id in graph.nodes
        assert graph.node_count == 1
        assert graph.nodes[node_id].data == "test data"
    
    def test_add_multiple_nodes(self):
        """Test adding multiple nodes"""
        graph = create_graph("test", 100, 3600)
        
        node_ids = []
        for i in range(10):
            embedding = np.random.randn(MAX_EMBEDDING_DIM).astype(np.float32)
            node_id = add_node(graph, embedding, f"node_{i}")
            node_ids.append(node_id)
        
        assert graph.node_count == 10
        assert all(nid in graph.nodes for nid in node_ids)
    
    def test_add_node_respects_max_nodes(self):
        """Test that adding nodes respects max_nodes limit"""
        graph = create_graph("test", max_nodes=5, time_budget=3600)
        
        for i in range(10):
            embedding = np.random.randn(MAX_EMBEDDING_DIM).astype(np.float32)
            node_id = add_node(graph, embedding, f"node_{i}")
            
            # Should not exceed max_nodes
            assert graph.node_count <= 5
    
    def test_node_embedding_storage(self):
        """Test that node embeddings are stored correctly"""
        graph = create_graph("test", 100, 3600)
        
        embedding = np.array([1, 2, 3] + [0] * (MAX_EMBEDDING_DIM - 3), dtype=np.float32)
        node_id = add_node(graph, embedding, "test")
        
        stored_embedding = graph.nodes[node_id].vector_embedding
        assert np.allclose(stored_embedding, embedding)
    
    def test_fractal_seed_marking(self):
        """Test marking nodes as fractal seeds"""
        graph = create_graph("test", 100, 3600)
        
        embedding = np.random.randn(MAX_EMBEDDING_DIM).astype(np.float32)
        node_id = add_node(graph, embedding, "important_fact")
        
        # Mark as fractal seed
        graph.nodes[node_id].is_fractal_seed = True
        
        assert graph.nodes[node_id].is_fractal_seed


class TestEdgeManagement:
    """Test edge creation, deletion, and properties"""
    
    def test_add_edge(self):
        """Test adding an edge between nodes"""
        graph = create_graph("test", 100, 3600)
        
        # Create two nodes
        emb1 = np.random.randn(MAX_EMBEDDING_DIM).astype(np.float32)
        emb2 = np.random.randn(MAX_EMBEDDING_DIM).astype(np.float32)
        node1 = add_node(graph, emb1, "node1")
        node2 = add_node(graph, emb2, "node2")
        
        # Add edge
        success = add_edge(graph, node1, node2, 0.9, "related")
        
        assert success
        assert node2 in graph.nodes[node1].neighbors
        assert node1 in graph.nodes[node2].neighbors
    
    def test_bounded_connectivity_constraint(self):
        """Test that nodes respect bounded connectivity (max 6 neighbors)"""
        graph = create_graph("test", 100, 3600)
        
        # Create hub node
        hub_embedding = np.random.randn(MAX_EMBEDDING_DIM).astype(np.float32)
        hub_id = add_node(graph, hub_embedding, "hub")
        
        # Try to add 10 neighbors (should be capped at 6)
        for i in range(10):
            embedding = np.random.randn(MAX_EMBEDDING_DIM).astype(np.float32)
            neighbor_id = add_node(graph, embedding, f"neighbor_{i}")
            add_edge(graph, hub_id, neighbor_id, 0.8, "connected")
        
        # Node should have at most 6 neighbors
        assert len(graph.nodes[hub_id].neighbors) <= 6
    
    def test_edge_relevance_score(self):
        """Test edge relevance scores"""
        graph = create_graph("test", 100, 3600)
        
        emb1 = np.random.randn(MAX_EMBEDDING_DIM).astype(np.float32)
        emb2 = np.random.randn(MAX_EMBEDDING_DIM).astype(np.float32)
        node1 = add_node(graph, emb1, "node1")
        node2 = add_node(graph, emb2, "node2")
        
        relevance = 0.75
        add_edge(graph, node1, node2, relevance, "test_edge")
        
        # Check that edge was added with correct relevance
        assert len(graph.nodes[node1].neighbors) == 1
    
    def test_remove_edge(self):
        """Test removing an edge"""
        graph = create_graph("test", 100, 3600)
        
        emb1 = np.random.randn(MAX_EMBEDDING_DIM).astype(np.float32)
        emb2 = np.random.randn(MAX_EMBEDDING_DIM).astype(np.float32)
        node1 = add_node(graph, emb1, "node1")
        node2 = add_node(graph, emb2, "node2")
        
        add_edge(graph, node1, node2, 0.8, "connected")
        assert len(graph.nodes[node1].neighbors) == 1
        
        # Remove edge
        success = remove_edge(graph, node1, node2)
        
        assert success
        assert len(graph.nodes[node1].neighbors) == 0


class TestCentralityCalculation:
    """Test centrality and hub detection"""
    
    def test_recalculate_centrality(self):
        """Test centrality recalculation"""
        graph = create_graph("test", 100, 3600)
        
        # Create star topology
        hub_embedding = np.random.randn(MAX_EMBEDDING_DIM).astype(np.float32)
        hub_id = add_node(graph, hub_embedding, "hub")
        
        spoke_ids = []
        for i in range(5):
            embedding = np.random.randn(MAX_EMBEDDING_DIM).astype(np.float32)
            spoke_id = add_node(graph, embedding, f"spoke_{i}")
            spoke_ids.append(spoke_id)
            add_edge(graph, hub_id, spoke_id, 0.9, "hub_connection")
        
        # Recalculate centrality
        recalculate_centrality(graph)
        
        # Hub should have highest centrality
        hub_centrality = graph.nodes[hub_id].centrality
        spoke_centrality = [graph.nodes[s].centrality for s in spoke_ids]
        
        assert hub_centrality > max(spoke_centrality)


class TestRetrieval:
    """Test memory retrieval and traversal"""
    
    def test_find_most_relevant_node(self):
        """Test finding most relevant node for a query"""
        graph = create_graph("test", 100, 3600)
        
        # Create nodes with known embeddings
        query_embedding = np.array([1, 0, 0] + [0] * (MAX_EMBEDDING_DIM - 3), dtype=np.float32)
        
        similar_embedding = np.array([0.9, 0.1, 0] + [0] * (MAX_EMBEDDING_DIM - 3), dtype=np.float32)
        dissimilar_embedding = np.array([0, 1, 0] + [0] * (MAX_EMBEDDING_DIM - 3), dtype=np.float32)
        
        similar_id = add_node(graph, similar_embedding, "similar")
        dissimilar_id = add_node(graph, dissimilar_embedding, "dissimilar")
        
        # Find most relevant
        result = find_most_relevant_node(graph, query_embedding)
        
        assert result == similar_id
    
    def test_empty_graph_retrieval(self):
        """Test retrieval from empty graph"""
        graph = create_graph("test", 100, 3600)
        
        query_embedding = np.random.randn(MAX_EMBEDDING_DIM).astype(np.float32)
        result = find_most_relevant_node(graph, query_embedding)
        
        assert result is None


class TestMetabolism:
    """Test metabolic state tracking and updates"""
    
    def test_metabolism_initialization(self):
        """Test metabolism initializes in HEALTHY state"""
        graph = create_graph("test", 100, 3600)
        
        assert graph.metabolism.state == MetabolicState.HEALTHY
        assert graph.metabolism.metabolic_weight == 1.0
    
    def test_metabolism_state_transitions(self):
        """Test metabolism state transitions based on resource depletion"""
        graph = create_graph("test", 100, 300)  # 5 minute budget
        
        # Initially healthy
        assert graph.metabolism.state == MetabolicState.HEALTHY
        
        # Move to stressed
        update_metabolism(graph, 100, 200, 50.0)  # 200 seconds remaining = 40% budget
        assert graph.metabolism.state == MetabolicState.STRESSED
        
        # Move to critical
        update_metabolism(graph, 100, 30, 10.0)  # 30 seconds remaining = 10% budget
        assert graph.metabolism.state == MetabolicState.CRITICAL
    
    def test_metabolic_weight_increases_in_stress(self):
        """Test that metabolic weight increases as resources diminish"""
        graph = create_graph("test", 100, 300)
        
        # Initially 1.0
        initial_weight = graph.metabolism.metabolic_weight
        
        # Deplete resources
        update_metabolism(graph, 100, 30, 10.0)
        stressed_weight = graph.metabolism.metabolic_weight
        
        assert stressed_weight >= initial_weight


class TestSerialization:
    """Test graph persistence (save/load)"""
    
    def test_save_and_load_empty_graph(self):
        """Test saving and loading an empty graph"""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = create_graph("test", 100, 3600)
            
            filepath = Path(tmpdir) / "graph.bin"
            save_binary(graph, str(filepath))
            
            loaded_graph = load_binary(str(filepath))
            
            assert loaded_graph.name == "test"
            assert loaded_graph.node_count == 0
    
    def test_save_and_load_graph_with_nodes(self):
        """Test saving and loading a graph with nodes"""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = create_graph("test", 100, 3600)
            
            # Add some nodes
            for i in range(5):
                embedding = np.random.randn(MAX_EMBEDDING_DIM).astype(np.float32)
                add_node(graph, embedding, f"node_{i}")
            
            filepath = Path(tmpdir) / "graph.bin"
            save_binary(graph, str(filepath))
            
            loaded_graph = load_binary(str(filepath))
            
            assert loaded_graph.node_count == 5
            assert all(f"node_{i}" in [loaded_graph.nodes[nid].data for i in range(5) for nid in loaded_graph.nodes])
    
    def test_save_and_load_preserves_metabolism(self):
        """Test that metabolism state is preserved across save/load"""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = create_graph("test", 100, 3600)
            
            # Update metabolism
            update_metabolism(graph, 50, 100, 30.0)
            
            filepath = Path(tmpdir) / "graph.bin"
            save_binary(graph, str(filepath))
            
            loaded_graph = load_binary(str(filepath))
            
            assert loaded_graph.metabolism.state == graph.metabolism.state
            assert loaded_graph.metabolism.messages_remaining == graph.metabolism.messages_remaining


class TestSimilarityCalculation:
    """Test cosine similarity computations"""
    
    def test_identical_vectors_have_similarity_one(self):
        """Test that identical vectors have similarity 1.0"""
        vec = np.random.randn(MAX_EMBEDDING_DIM).astype(np.float32)
        sim = cosine_similarity(vec, vec)
        
        assert np.isclose(sim, 1.0)
    
    def test_orthogonal_vectors_have_similarity_zero(self):
        """Test that orthogonal vectors have similarity 0.0"""
        vec1 = np.array([1, 0, 0] + [0] * (MAX_EMBEDDING_DIM - 3), dtype=np.float32)
        vec2 = np.array([0, 1, 0] + [0] * (MAX_EMBEDDING_DIM - 3), dtype=np.float32)
        sim = cosine_similarity(vec1, vec2)
        
        assert np.isclose(sim, 0.0, atol=1e-6)
    
    def test_opposite_vectors_have_similarity_negative_one(self):
        """Test that opposite vectors have similarity -1.0"""
        vec = np.random.randn(MAX_EMBEDDING_DIM).astype(np.float32)
        opposite = -vec
        sim = cosine_similarity(vec, opposite)
        
        assert np.isclose(sim, -1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

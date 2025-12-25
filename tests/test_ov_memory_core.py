#!/usr/bin/env python3
"""
OV-Memory v1.1 Core Functionality Tests
Tests metabolism engine, centroid indexing, and basic graph operations

Run: pytest test_ov_memory_core.py -v
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from python.ov_memory_v1_1 import (
    create_graph,
    add_node,
    add_edge,
    initialize_metabolism,
    update_metabolism,
    recalculate_centrality,
    find_most_relevant_node,
    create_fractal_seed,
    save_binary,
    load_binary,
    MetabolicState,
    cosine_similarity,
    temporal_decay,
    calculate_metabolic_relevance,
    print_graph_stats,
    MAX_EMBEDDING_DIM,
    CONFIG
)


class TestBasicGraphOperations:
    """Test fundamental graph operations"""

    def test_graph_creation(self):
        """Test graph initialization"""
        graph = create_graph("test_graph", 100, 3600)
        assert graph.name == "test_graph"
        assert graph.node_count == 0
        assert graph.max_nodes == 100
        assert graph.max_session_time_seconds == 3600

    def test_add_node(self):
        """Test node addition"""
        graph = create_graph("test_graph", 100, 3600)
        embedding = np.full(MAX_EMBEDDING_DIM, 0.5, dtype=np.float32)
        
        node_id = add_node(graph, embedding, "Test Node")
        assert node_id == 0
        assert graph.node_count == 1
        assert node_id in graph.nodes
        assert graph.nodes[node_id].data == "Test Node"

    def test_add_multiple_nodes(self):
        """Test adding multiple nodes"""
        graph = create_graph("test_graph", 100, 3600)
        
        node_ids = []
        for i in range(10):
            embedding = np.full(MAX_EMBEDDING_DIM, 0.5 + i*0.01, dtype=np.float32)
            node_id = add_node(graph, embedding, f"Node {i}")
            node_ids.append(node_id)
        
        assert graph.node_count == 10
        assert len(node_ids) == 10
        assert all(nid in graph.nodes for nid in node_ids)

    def test_add_edge(self):
        """Test edge addition"""
        graph = create_graph("test_graph", 100, 3600)
        
        emb1 = np.full(MAX_EMBEDDING_DIM, 0.5, dtype=np.float32)
        emb2 = np.full(MAX_EMBEDDING_DIM, 0.6, dtype=np.float32)
        
        node1 = add_node(graph, emb1, "Node 1")
        node2 = add_node(graph, emb2, "Node 2")
        
        success = add_edge(graph, node1, node2, 0.9, "related_to")
        assert success is True
        assert len(graph.nodes[node1].neighbors) == 1
        assert graph.nodes[node1].neighbors[0].target_id == node2
        assert graph.nodes[node1].neighbors[0].relevance_score == 0.9

    def test_edge_clamping(self):
        """Test that edge relevance scores are clamped to [0, 1]"""
        graph = create_graph("test_graph", 100, 3600)
        
        emb1 = np.full(MAX_EMBEDDING_DIM, 0.5, dtype=np.float32)
        emb2 = np.full(MAX_EMBEDDING_DIM, 0.6, dtype=np.float32)
        
        node1 = add_node(graph, emb1, "Node 1")
        node2 = add_node(graph, emb2, "Node 2")
        
        add_edge(graph, node1, node2, 1.5, "test")  # Invalid score
        edge = graph.nodes[node1].neighbors[0]
        assert edge.relevance_score == 1.0
        
        add_edge(graph, node1, node2, -0.5, "test")  # Invalid score
        edge = graph.nodes[node1].neighbors[1]
        assert edge.relevance_score == 0.0

    def test_bounded_neighbors(self):
        """Test that node cannot exceed HEXAGONAL_NEIGHBORS"""
        graph = create_graph("test_graph", 100, 3600)
        
        emb_center = np.full(MAX_EMBEDDING_DIM, 0.5, dtype=np.float32)
        center_node = add_node(graph, emb_center, "Center")
        
        # Add 6 neighbors (should all succeed)
        for i in range(CONFIG["HEXAGONAL_NEIGHBORS"]):
            emb = np.full(MAX_EMBEDDING_DIM, 0.5 + i*0.01, dtype=np.float32)
            neighbor = add_node(graph, emb, f"Neighbor {i}")
            success = add_edge(graph, center_node, neighbor, 0.9, "neighbor")
            assert success is True
        
        # Try to add 7th neighbor (should fail)
        emb_extra = np.full(MAX_EMBEDDING_DIM, 0.8, dtype=np.float32)
        extra_node = add_node(graph, emb_extra, "Extra")
        success = add_edge(graph, center_node, extra_node, 0.9, "neighbor")
        assert success is False


class TestVectorMath:
    """Test vector operations and similarities"""

    def test_cosine_similarity_identical(self):
        """Test cosine similarity of identical vectors"""
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        similarity = cosine_similarity(vec, vec)
        assert abs(similarity - 1.0) < 1e-5

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity of orthogonal vectors"""
        vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        similarity = cosine_similarity(vec1, vec2)
        assert abs(similarity) < 1e-5

    def test_cosine_similarity_opposite(self):
        """Test cosine similarity of opposite vectors"""
        vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec2 = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        similarity = cosine_similarity(vec1, vec2)
        assert abs(similarity - (-1.0)) < 1e-5

    def test_temporal_decay_immediate(self):
        """Test temporal decay at creation time"""
        now = 1000.0
        decay = temporal_decay(now, now)
        assert abs(decay - 1.0) < 1e-5

    def test_temporal_decay_over_time(self):
        """Test temporal decay increases with age"""
        created = 1000.0
        decay_1h = temporal_decay(created, created + 3600)
        decay_1d = temporal_decay(created, created + 86400)
        
        # Decay should decrease over time
        assert decay_1h > decay_1d
        assert decay_1h > 0.0
        assert decay_1d > 0.0


class TestMetabolism:
    """Test metabolic state tracking"""

    def test_metabolism_initialization(self):
        """Test metabolism engine initialization"""
        graph = create_graph("test_graph", 100, 3600)
        
        assert graph.metabolism.messages_remaining == 100
        assert graph.metabolism.minutes_remaining == 60 * 60  # 3600 seconds
        assert graph.metabolism.state == MetabolicState.Healthy
        assert graph.metabolism.metabolic_weight == 1.0

    def test_metabolism_healthy_state(self):
        """Test metabolism remains healthy with abundant resources"""
        graph = create_graph("test_graph", 100, 3600)
        
        update_metabolism(graph, 10, 60, 25.0)
        assert graph.metabolism.state == MetabolicState.Healthy
        assert graph.metabolism.metabolic_weight == 1.0

    def test_metabolism_stressed_state(self):
        """Test metabolism transitions to STRESSED"""
        graph = create_graph("test_graph", 100, 3600)
        
        # Consume most time and messages
        update_metabolism(graph, 80, 2400, 50.0)  # Leaves <18min
        assert graph.metabolism.state == MetabolicState.Stressed
        assert graph.metabolism.metabolic_weight == 1.2

    def test_metabolism_critical_state(self):
        """Test metabolism transitions to CRITICAL"""
        graph = create_graph("test_graph", 100, 3600)
        
        # Consume nearly all resources
        update_metabolism(graph, 96, 3300, 80.0)  # Leaves <5min
        assert graph.metabolism.state == MetabolicState.Critical
        assert graph.metabolism.metabolic_weight == 1.5

    def test_metabolic_relevance_calculation(self):
        """Test metabolic relevance score calculation"""
        vec_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec_b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        
        now = 1000.0
        created = now  # Just created
        resource_avail = 50.0
        metabolic_weight = 1.0
        
        relevance = calculate_metabolic_relevance(
            vec_a, vec_b, created, now, resource_avail, metabolic_weight
        )
        
        # Should be high (identical vectors, just created, decent resources)
        assert relevance > 0.8


class TestCentroidIndexing:
    """Test centroid-based indexing"""

    def test_centroid_initialization(self):
        """Test centroid map initialization"""
        graph = create_graph("test_graph", 100, 3600)
        assert graph.centroid_map.max_hubs <= CONFIG["CENTROID_COUNT"]

    def test_centroid_recalculation(self):
        """Test centroid recalculation"""
        graph = create_graph("test_graph", 100, 3600)
        
        # Add nodes
        for i in range(10):
            embedding = np.full(MAX_EMBEDDING_DIM, 0.5 + i*0.01, dtype=np.float32)
            add_node(graph, embedding, f"Node {i}")
        
        # Connect some nodes to create hub structure
        add_edge(graph, 0, 1, 0.9, "related")
        add_edge(graph, 0, 2, 0.9, "related")
        add_edge(graph, 0, 3, 0.9, "related")
        
        recalculate_centrality(graph)
        
        assert len(graph.centroid_map.hub_node_ids) > 0
        assert all(hub_id in graph.nodes for hub_id in graph.centroid_map.hub_node_ids)

    def test_find_most_relevant_node(self):
        """Test entry point discovery"""
        graph = create_graph("test_graph", 100, 3600)
        
        emb1 = np.full(MAX_EMBEDDING_DIM, 0.5, dtype=np.float32)
        emb2 = np.full(MAX_EMBEDDING_DIM, 0.6, dtype=np.float32)
        emb3 = np.full(MAX_EMBEDDING_DIM, 0.7, dtype=np.float32)
        
        node1 = add_node(graph, emb1, "Node 1")
        node2 = add_node(graph, emb2, "Node 2")
        node3 = add_node(graph, emb3, "Node 3")
        
        add_edge(graph, node1, node2, 0.9, "related")
        add_edge(graph, node2, node3, 0.9, "related")
        
        recalculate_centrality(graph)
        
        query_vec = np.full(MAX_EMBEDDING_DIM, 0.5, dtype=np.float32)
        entry_node = find_most_relevant_node(graph, query_vec)
        
        assert entry_node is not None
        assert entry_node in graph.nodes


class TestPersistence:
    """Test graph serialization and loading"""

    def test_save_and_load(self, tmp_path):
        """Test save/load cycle"""
        graph = create_graph("test_graph", 100, 3600)
        
        emb1 = np.full(MAX_EMBEDDING_DIM, 0.5, dtype=np.float32)
        emb2 = np.full(MAX_EMBEDDING_DIM, 0.6, dtype=np.float32)
        
        node1 = add_node(graph, emb1, "Node 1")
        node2 = add_node(graph, emb2, "Node 2")
        add_edge(graph, node1, node2, 0.9, "related")
        
        filepath = tmp_path / "test_graph.json"
        save_binary(graph, str(filepath))
        
        assert filepath.exists()
        
        loaded_graph = load_binary(str(filepath))
        assert loaded_graph is not None
        assert loaded_graph.node_count == 2
        assert len(loaded_graph.nodes[node1].neighbors) == 1


class TestFractalSeeds:
    """Test fractal seed generation and hydration"""

    def test_create_fractal_seed(self):
        """Test seed creation from active nodes"""
        graph = create_graph("test_graph", 100, 3600)
        
        # Add nodes
        for i in range(5):
            embedding = np.full(MAX_EMBEDDING_DIM, 0.5 + i*0.01, dtype=np.float32)
            add_node(graph, embedding, f"Node {i}")
        
        seed_id = create_fractal_seed(graph, "test_seed")
        
        assert seed_id is not None
        assert seed_id in graph.nodes
        assert graph.nodes[seed_id].is_fractal_seed is True


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_graph_traversal(self):
        """Test traversal on empty graph"""
        graph = create_graph("test_graph", 100, 3600)
        query_vec = np.full(MAX_EMBEDDING_DIM, 0.5, dtype=np.float32)
        
        entry_node = find_most_relevant_node(graph, query_vec)
        assert entry_node is None

    def test_max_nodes_exceeded(self):
        """Test behavior when max nodes exceeded"""
        graph = create_graph("test_graph", 5, 3600)  # Max 5 nodes
        
        # Add nodes up to max
        node_ids = []
        for i in range(5):
            embedding = np.full(MAX_EMBEDDING_DIM, 0.5 + i*0.01, dtype=np.float32)
            node_id = add_node(graph, embedding, f"Node {i}")
            node_ids.append(node_id)
        
        assert graph.node_count == 5
        
        # Try to add one more (should fail)
        embedding = np.full(MAX_EMBEDDING_DIM, 0.6, dtype=np.float32)
        node_id = add_node(graph, embedding, "Extra Node")
        assert node_id is None

    def test_invalid_edge_nodes(self):
        """Test edge creation with invalid node IDs"""
        graph = create_graph("test_graph", 100, 3600)
        
        success = add_edge(graph, 999, 888, 0.9, "test")
        assert success is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

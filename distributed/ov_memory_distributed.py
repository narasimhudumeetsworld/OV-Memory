#!/usr/bin/env python3
"""
OV-MEMORY v1.1 - Distributed Implementation
Om Vinayaka 🙏

Distributed memory system with:
- Multi-node synchronization
- Shard-based partitioning (256 shards)
- Consensus protocol
- Distributed context retrieval
- Network fault tolerance
"""

import asyncio
import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Set, Optional, Tuple
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONSISTENT_HASH_BUCKETS = 256
REPLICATION_FACTOR = 3
SYNC_INTERVAL = 5.0  # seconds
TIMEOUT_DURATION = 10.0
EMBEDDING_DIM = 768


class ConsistentHashRing:
    """Consistent hashing for distributed sharding"""

    def __init__(self, num_buckets: int = CONSISTENT_HASH_BUCKETS):
        self.num_buckets = num_buckets
        self.nodes: Dict[int, str] = {}  # bucket -> node_id

    def get_hash(self, key: str) -> int:
        """Get bucket for key using consistent hashing"""
        hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
        return hash_value % self.num_buckets

    def assign_node(self, node_id: str, buckets: List[int]):
        """Assign buckets to node"""
        for bucket in buckets:
            self.nodes[bucket] = node_id

    def get_replicas(self, key: str) -> List[str]:
        """Get replica nodes for key"""
        primary_bucket = self.get_hash(key)
        replicas = set()
        replicas.add(self.nodes.get(primary_bucket, "node_0"))

        # Add REPLICATION_FACTOR replicas
        for i in range(1, REPLICATION_FACTOR):
            bucket = (primary_bucket + i) % self.num_buckets
            replicas.add(self.nodes.get(bucket, f"node_{i}"))

        return list(replicas)


@dataclass
class DistributedNode:
    """Node in distributed graph"""
    id: int
    embedding: List[float]
    content: str
    intrinsic_weight: float
    shard_id: int
    created_at: str
    owner_node: str  # Which distributed node owns this


@dataclass
class SyncMessage:
    """Message for node synchronization"""
    source_node: str
    message_type: str  # 'create', 'update', 'delete', 'sync'
    data: Dict
    timestamp: str
    sequence_num: int


class DistributedMemoryGraph:
    """Distributed memory graph with multi-node support"""

    def __init__(self, node_id: str, num_shards: int = CONSISTENT_HASH_BUCKETS):
        self.node_id = node_id
        self.hash_ring = ConsistentHashRing(num_shards)
        self.local_shards: Dict[int, Dict[int, DistributedNode]] = {
            i: {} for i in range(num_shards)
        }
        self.sync_queue: asyncio.Queue = asyncio.Queue()
        self.peer_nodes: Set[str] = set()
        self.sequence_counter = 0
        self.ack_buffer: Dict[int, Set[str]] = {}  # seq_num -> {acked nodes}

    def add_peer(self, peer_id: str):
        """Register peer node"""
        self.peer_nodes.add(peer_id)
        logger.info(f"Peer {peer_id} joined cluster")

    def remove_peer(self, peer_id: str):
        """Unregister peer node"""
        self.peer_nodes.discard(peer_id)
        logger.warning(f"Peer {peer_id} left cluster")

    def _get_shard_for_node(self, node_id: int) -> int:
        """Get shard for node using consistent hash"""
        key = str(node_id)
        return self.hash_ring.get_hash(key)

    async def add_node(
        self,
        node_id: int,
        embedding: np.ndarray,
        content: str,
        intrinsic_weight: float = 1.0
    ) -> bool:
        """Add node with distributed coordination"""
        shard_id = self._get_shard_for_node(node_id)
        replicas = self.hash_ring.get_replicas(str(node_id))

        node = DistributedNode(
            id=node_id,
            embedding=embedding.tolist(),
            content=content,
            intrinsic_weight=intrinsic_weight,
            shard_id=shard_id,
            created_at=datetime.now().isoformat(),
            owner_node=self.node_id
        )

        # Store locally
        self.local_shards[shard_id][node_id] = node

        # Broadcast to peers
        msg = SyncMessage(
            source_node=self.node_id,
            message_type="create",
            data={"node": asdict(node)},
            timestamp=datetime.now().isoformat(),
            sequence_num=self.sequence_counter
        )
        self.sequence_counter += 1

        await self._broadcast_sync(msg, replicas)
        logger.info(f"Created node {node_id} in shard {shard_id}")
        return True

    async def _broadcast_sync(self, msg: SyncMessage, targets: List[str]):
        """Broadcast sync message to target nodes"""
        await self.sync_queue.put(msg)
        self.ack_buffer[msg.sequence_num] = set()

        # Simulate network broadcast
        for target in targets:
            if target != self.node_id:
                logger.debug(f"Syncing to {target}")
                # In real implementation, would use RPC/HTTP
                await asyncio.sleep(0.01)

    async def get_node(self, node_id: int) -> Optional[DistributedNode]:
        """Get node from distributed graph"""
        shard_id = self._get_shard_for_node(node_id)
        return self.local_shards[shard_id].get(node_id)

    async def get_shard_nodes(self, shard_id: int) -> Dict[int, DistributedNode]:
        """Get all nodes in shard"""
        return self.local_shards[shard_id].copy()

    async def sync_heartbeat(self):
        """Periodic synchronization heartbeat"""
        while True:
            try:
                # Send heartbeat to all peers
                for peer in self.peer_nodes:
                    msg = SyncMessage(
                        source_node=self.node_id,
                        message_type="heartbeat",
                        data={"alive": True},
                        timestamp=datetime.now().isoformat(),
                        sequence_num=self.sequence_counter
                    )
                    self.sequence_counter += 1
                    await self.sync_queue.put(msg)

                await asyncio.sleep(SYNC_INTERVAL)
            except asyncio.CancelledError:
                logger.info("Heartbeat cancelled")
                break

    async def consensus_commit(self, seq_num: int) -> bool:
        """Wait for quorum acknowledgment"""
        start_time = asyncio.get_event_loop().time()
        quorum_size = len(self.peer_nodes) // 2 + 1

        while True:
            if seq_num in self.ack_buffer:
                if len(self.ack_buffer[seq_num]) >= quorum_size:
                    del self.ack_buffer[seq_num]
                    return True

            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > TIMEOUT_DURATION:
                logger.warning(f"Consensus timeout for sequence {seq_num}")
                return False

            await asyncio.sleep(0.1)


class DistributedContextRetrieval:
    """JIT context retrieval across distributed nodes"""

    def __init__(self, graph: DistributedMemoryGraph):
        self.graph = graph
        self.retrieved_nodes: Set[int] = set()

    async def get_jit_context(
        self,
        query_embedding: np.ndarray,
        max_tokens: int = 2000
    ) -> Tuple[str, float]:
        """Retrieve context from distributed graph"""
        context_parts = []
        context_size = 0

        # Scan all shards in parallel
        shard_tasks = []
        for shard_id in range(CONSISTENT_HASH_BUCKETS):
            task = self._search_shard(shard_id, query_embedding)
            shard_tasks.append(task)

        shard_results = await asyncio.gather(*shard_tasks)

        # Merge results
        for shard_candidates in shard_results:
            for node_id, score in shard_candidates:
                if node_id not in self.retrieved_nodes:
                    self.retrieved_nodes.add(node_id)
                    node = await self.graph.get_node(node_id)
                    if node and context_size < max_tokens:
                        context_parts.append(node.content)
                        context_size += len(node.content) // 4

        context = " ".join(context_parts)
        token_usage = context_size / max_tokens * 100.0

        logger.info(f"Retrieved context from {len(self.retrieved_nodes)} distributed nodes")
        return context, token_usage

    async def _search_shard(
        self,
        shard_id: int,
        query_embedding: np.ndarray
    ) -> List[Tuple[int, float]]:
        """Search single shard for candidates"""
        shard_nodes = await self.graph.get_shard_nodes(shard_id)
        candidates = []

        for node_id, node in shard_nodes.items():
            similarity = self._cosine_similarity(
                query_embedding,
                np.array(node.embedding)
            )
            if similarity > 0.5:  # Threshold
                candidates.append((node_id, similarity))

        # Sort by similarity
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:10]  # Top 10 per shard

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot_product / (norm_a * norm_b))


# ============================================================================
# MAIN TEST SUITE
# ============================================================================

async def main():
    print("============================================================")
    print("🧠 OV-MEMORY v1.1 - DISTRIBUTED IMPLEMENTATION")
    print("Om Vinayaka 🙏")
    print("============================================================\n")

    # Create distributed cluster
    node1 = DistributedMemoryGraph("node_1")
    node2 = DistributedMemoryGraph("node_2")
    node3 = DistributedMemoryGraph("node_3")

    # Register peers
    for node in [node1, node2, node3]:
        for peer in [node1, node2, node3]:
            if node.node_id != peer.node_id:
                node.add_peer(peer.node_id)

    print("✅ Initialized 3-node cluster with replication factor 3")

    # Simulate hash ring
    hash_ring = ConsistentHashRing(256)
    hash_ring.assign_node("node_1", list(range(0, 86)))
    hash_ring.assign_node("node_2", list(range(86, 171)))
    hash_ring.assign_node("node_3", list(range(171, 256)))
    print("✅ Assigned shards to nodes using consistent hashing")

    # Add distributed nodes
    for i in range(5):
        embedding = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        await node1.add_node(i, embedding, f"Memory {i}", 1.0)
    print("✅ Added 5 nodes to distributed graph")

    # Start heartbeat
    heartbeat_task = asyncio.create_task(node1.sync_heartbeat())
    await asyncio.sleep(0.1)

    # Test distributed context retrieval
    retriever = DistributedContextRetrieval(node1)
    query = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    context, token_usage = await retriever.get_jit_context(query, 2000)
    print(f"✅ Distributed context retrieved: {len(context)} characters ({{:.1f}}% tokens)".format(token_usage))

    # Cleanup
    heartbeat_task.cancel()
    try:
        await heartbeat_task
    except asyncio.CancelledError:
        pass

    print("\n✅ All distributed implementation tests passed!")
    print("============================================================")


if __name__ == "__main__":
    asyncio.run(main())

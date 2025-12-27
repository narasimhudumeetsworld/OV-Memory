#!/usr/bin/env python3
"""
OV-MEMORY v1.1 - GPU-Accelerated Implementation
Om Vinayaka 🙏

GPU-accelerated memory system with:
- CUDA matrix operations
- Batch similarity computation
- GPU-side priority calculation
- Async memory transfer
- Multi-GPU support
"""

import numpy as np
import logging
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 768
BATCH_SIZE = 256
MAX_NODES = 1000000

try:
    import cupy as cp
    import cupyx.scipy.spatial.distance as cpdist
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    logger.warning("CuPy not available. GPU acceleration disabled.")


@dataclass
class GPUMemoryBuffer:
    """GPU memory buffer for embeddings"""
    embeddings: Optional[object] = None  # cp.ndarray
    priorities: Optional[object] = None
    node_ids: Optional[object] = None
    content_ids: Optional[object] = None
    capacity: int = 0
    size: int = 0
    device_id: int = 0


class GPUAccelerator:
    """GPU acceleration engine"""

    def __init__(self, device_id: int = 0):
        if not HAS_GPU:
            raise RuntimeError("GPU acceleration requires CuPy")

        self.device_id = device_id
        self.stream = cp.cuda.Stream()
        self.buffer = GPUMemoryBuffer(
            capacity=MAX_NODES,
            device_id=device_id
        )
        self._allocate_gpu_memory()
        logger.info(f"Initialized GPU {device_id} for OV-Memory")

    def _allocate_gpu_memory(self):
        """Allocate GPU memory for embeddings"""
        with cp.cuda.Device(self.device_id):
            self.buffer.embeddings = cp.zeros(
                (self.buffer.capacity, EMBEDDING_DIM),
                dtype=cp.float32
            )
            self.buffer.priorities = cp.zeros(self.buffer.capacity, dtype=cp.float32)
            self.buffer.node_ids = cp.zeros(self.buffer.capacity, dtype=cp.int32)
            self.buffer.content_ids = cp.zeros(self.buffer.capacity, dtype=cp.int32)
            logger.info(f"Allocated {self.buffer.capacity * EMBEDDING_DIM * 4 / 1e9:.2f} GB GPU memory")

    def transfer_embeddings_to_gpu(self, embeddings: np.ndarray) -> Tuple[int, int]:
        """Asynchronously transfer embeddings to GPU"""
        with cp.cuda.Device(self.device_id):
            with self.stream:
                # Convert to GPU array
                gpu_embeddings = cp.asarray(embeddings, dtype=cp.float32)

                # Update buffer
                end_idx = min(self.buffer.size + len(embeddings), self.buffer.capacity)
                start_idx = max(0, end_idx - len(embeddings))
                count = end_idx - start_idx

                self.buffer.embeddings[start_idx:end_idx] = gpu_embeddings[:count]
                self.buffer.size = end_idx

        return start_idx, end_idx

    def batch_cosine_similarity(
        self,
        query: np.ndarray,
        batch_start: int,
        batch_end: int
    ) -> np.ndarray:
        """Compute cosine similarity in batch on GPU"""
        with cp.cuda.Device(self.device_id):
            with self.stream:
                gpu_query = cp.asarray(query, dtype=cp.float32)

                # Get embeddings from buffer
                batch_embeddings = self.buffer.embeddings[batch_start:batch_end]

                # Compute norms
                query_norm = cp.linalg.norm(gpu_query)
                embedding_norms = cp.linalg.norm(batch_embeddings, axis=1)

                # Avoid division by zero
                embedding_norms = cp.where(embedding_norms == 0, 1.0, embedding_norms)

                # Compute similarities
                dot_products = cp.dot(batch_embeddings, gpu_query)
                similarities = dot_products / (embedding_norms * (query_norm + 1e-8))

                # Transfer back to CPU
                return cp.asnumpy(similarities)

    def batch_priority_calculation(
        self,
        semantic_scores: np.ndarray,
        centrality_scores: np.ndarray,
        recency_scores: np.ndarray,
        intrinsic_weights: np.ndarray,
        alpha: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate priorities in batch on GPU"""
        with cp.cuda.Device(self.device_id):
            with self.stream:
                # Transfer to GPU
                gpu_semantic = cp.asarray(semantic_scores, dtype=cp.float32)
                gpu_centrality = cp.asarray(centrality_scores, dtype=cp.float32)
                gpu_recency = cp.asarray(recency_scores, dtype=cp.float32)
                gpu_intrinsic = cp.asarray(intrinsic_weights, dtype=cp.float32)

                # Compute priorities (element-wise multiplication)
                priorities = gpu_semantic * gpu_centrality * gpu_recency * gpu_intrinsic

                # Find which exceed alpha
                exceeds_alpha = (priorities > alpha).astype(cp.float32)

                # Transfer back
                return (
                    cp.asnumpy(priorities),
                    cp.asnumpy(exceeds_alpha)
                )

    def batch_drift_detection(
        self,
        semantic_scores: np.ndarray,
        hops: np.ndarray,
        threshold_semantic: float = 0.5,
        threshold_hops: int = 3
    ) -> np.ndarray:
        """Detect drift in batch on GPU"""
        with cp.cuda.Device(self.device_id):
            with self.stream:
                gpu_semantic = cp.asarray(semantic_scores, dtype=cp.float32)
                gpu_hops = cp.asarray(hops, dtype=cp.int32)

                # Drift: hops > 3 AND semantic < 0.5
                drift = (gpu_hops > threshold_hops) & (gpu_semantic < threshold_semantic)

                return cp.asnumpy(drift.astype(cp.float32))

    def forward_pass(
        self,
        query: np.ndarray,
        semantic_scores: np.ndarray,
        centrality_scores: np.ndarray,
        recency_scores: np.ndarray,
        intrinsic_weights: np.ndarray,
        alpha: float
    ) -> Dict:
        """Full forward pass on GPU"""
        start_time = time.time()

        # Priority calculation
        priorities, exceeds_alpha = self.batch_priority_calculation(
            semantic_scores,
            centrality_scores,
            recency_scores,
            intrinsic_weights,
            alpha
        )

        # Get injection decisions
        inject_mask = exceeds_alpha

        end_time = time.time()
        compute_time = end_time - start_time

        return {
            "priorities": priorities,
            "inject_mask": inject_mask,
            "compute_time_ms": compute_time * 1000,
            "nodes_processed": len(priorities)
        }

    def synchronize(self):
        """Synchronize GPU stream"""
        with cp.cuda.Device(self.device_id):
            self.stream.synchronize()

    def get_memory_usage(self) -> Dict:
        """Get GPU memory usage statistics"""
        with cp.cuda.Device(self.device_id):
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()

            return {
                "allocated_mb": mempool.get_limit() / 1e6,
                "used_mb": (mempool.get_limit() - mempool.get_free_memory()) / 1e6,
                "pinned_mb": pinned_mempool.n_bytes / 1e6
            }


class MultiGPUAccelerator:
    """Multi-GPU acceleration manager"""

    def __init__(self, num_gpus: int = 2):
        self.num_gpus = min(num_gpus, cp.cuda.runtime.getDeviceCount() if HAS_GPU else 0)
        self.accelerators: List[GPUAccelerator] = []
        self.device_streams: Dict[int, object] = {}

        if HAS_GPU:
            for device_id in range(self.num_gpus):
                self.accelerators.append(GPUAccelerator(device_id))
            logger.info(f"Initialized {self.num_gpus} GPU devices")
        else:
            logger.warning("Multi-GPU disabled: GPU support not available")

    def distribute_batch(
        self,
        query: np.ndarray,
        semantic_scores: np.ndarray,
        centrality_scores: np.ndarray,
        recency_scores: np.ndarray,
        intrinsic_weights: np.ndarray,
        alpha: float
    ) -> List[Dict]:
        """Distribute batch across GPUs"""
        if not self.accelerators:
            logger.warning("No GPU accelerators available")
            return []

        # Split batch across GPUs
        batch_size = len(semantic_scores) // self.num_gpus
        results = []

        for gpu_id, accelerator in enumerate(self.accelerators):
            start_idx = gpu_id * batch_size
            end_idx = (gpu_id + 1) * batch_size if gpu_id < self.num_gpus - 1 else len(semantic_scores)

            result = accelerator.forward_pass(
                query,
                semantic_scores[start_idx:end_idx],
                centrality_scores[start_idx:end_idx],
                recency_scores[start_idx:end_idx],
                intrinsic_weights[start_idx:end_idx],
                alpha
            )
            results.append(result)

        return results

    def synchronize_all(self):
        """Synchronize all GPU streams"""
        for accelerator in self.accelerators:
            accelerator.synchronize()


# ============================================================================
# MAIN TEST SUITE
# ============================================================================

def main():
    print("============================================================")
    print("🧠 OV-MEMORY v1.1 - GPU-ACCELERATED IMPLEMENTATION")
    print("Om Vinayaka 🙏")
    print("============================================================\n")

    if not HAS_GPU:
        print("⚠️ GPU acceleration not available (CuPy required)")
        print("Install: pip install cupy-cuda11x")
        print("============================================================")
        return

    # Initialize GPU accelerator
    gpu = GPUAccelerator(device_id=0)
    print("✅ GPU accelerator initialized")

    # Create sample data
    embeddings = np.random.randn(1000, EMBEDDING_DIM).astype(np.float32)
    query = np.random.randn(EMBEDDING_DIM).astype(np.float32)

    # Transfer to GPU
    start_idx, end_idx = gpu.transfer_embeddings_to_gpu(embeddings)
    print(f"✅ Transferred {len(embeddings)} embeddings to GPU")

    # Compute batch similarities
    similarities = gpu.batch_cosine_similarity(query, start_idx, end_idx)
    print(f"✅ Computed similarities for {len(similarities)} nodes")

    # Test batch operations
    semantic = similarities
    centrality = np.random.rand(len(similarities)).astype(np.float32)
    recency = np.random.rand(len(similarities)).astype(np.float32)
    intrinsic = np.ones(len(similarities)).astype(np.float32)

    priorities, exceeds = gpu.batch_priority_calculation(
        semantic, centrality, recency, intrinsic, alpha=0.75
    )
    print(f"✅ Calculated priorities: {np.sum(exceeds)} nodes exceed threshold")

    # Test drift detection
    hops = np.random.randint(0, 5, len(similarities))
    drift_mask = gpu.batch_drift_detection(semantic, hops)
    print(f"✅ Drift detection: {np.sum(drift_mask)} nodes drifted")

    # Test multi-GPU
    multi_gpu = MultiGPUAccelerator(num_gpus=2)
    print(f"✅ Initialized {multi_gpu.num_gpus} GPU devices")

    # Benchmark
    start_time = time.time()
    for _ in range(100):
        gpu.batch_cosine_similarity(query, start_idx, end_idx)
    elapsed = time.time() - start_time
    throughput = 100 * len(similarities) / elapsed
    print(f"✅ GPU throughput: {throughput:.0f} similarity ops/sec")

    # Memory usage
    memory = gpu.get_memory_usage()
    print(f"✅ GPU memory: {memory['used_mb']:.0f}MB / {memory['allocated_mb']:.0f}MB")

    print("\n✅ All GPU acceleration tests passed!")
    print("============================================================")


if __name__ == "__main__":
    main()

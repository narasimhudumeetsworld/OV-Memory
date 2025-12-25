/**
 * =====================================================================
 * OV-Memory v1.1: TypeScript Implementation
 * =====================================================================
 * Author: Prayaga Vaibhavlakshmi
 * License: Apache License 2.0
 * Om Vinayaka 🙏
 *
 * Production-ready with full type safety, async operations, and modern patterns
 *
 * Installation:
 *   npm install
 *   npx tsc ov_memory_v1_1.ts --target ES2020
 *
 * =====================================================================
 */

import * as fs from 'fs';
import * as path from 'path';

// ===== CONFIGURATION =====

const CONFIG = {
    MAX_NODES: 100_000,
    MAX_EMBEDDING_DIM: 768,
    HEXAGONAL_NEIGHBORS: 6,
    RELEVANCE_THRESHOLD: 0.8,
    MAX_SESSION_TIME: 3600,
    LOOP_DETECTION_WINDOW: 10,
    LOOP_ACCESS_LIMIT: 3,
    TEMPORAL_DECAY_HALF_LIFE: 86400.0,
    CENTROID_COUNT: 5,
    CENTROID_SCAN_PERCENTAGE: 0.05,
    AUDIT_SEMANTIC_TRIGGER: 1260,
    AUDIT_FRACTAL_TRIGGER: 1080,
    AUDIT_CRITICAL_SEAL_TRIGGER: 300
} as const;

// ===== ENUMS =====

enum MetabolicState {
    Healthy = 0,
    Stressed = 1,
    Critical = 2
}

enum SafetyCode {
    Ok = 0,
    LoopDetected = 1,
    SessionExpired = 2,
    InvalidNode = -1
}

// ===== DATA STRUCTURES =====

interface AgentMetabolism {
    messages_remaining: number;
    minutes_remaining: number;
    is_api_mode: boolean;
    context_availability: number;
    metabolic_weight: number;
    state: MetabolicState;
    audit_last_run: number;
}

interface HoneycombEdge {
    target_id: number;
    relevance_score: number;
    relationship_type: string;
    timestamp_created: number;
}

interface HoneycombNode {
    id: number;
    vector_embedding: Float32Array;
    data: string;
    neighbors: HoneycombEdge[];
    last_accessed_timestamp: number;
    access_count_session: number;
    access_time_first: number;
    relevance_to_focus: number;
    metabolic_weight: number;
    is_active: boolean;
    is_fractal_seed: boolean;
}

interface CentroidMap {
    hub_node_ids: number[];
    hub_centrality: number[];
    max_hubs: number;
}

interface HoneycombGraph {
    name: string;
    nodes: Map<number, HoneycombNode>;
    node_count: number;
    max_nodes: number;
    session_start_time: number;
    max_session_time_seconds: number;
    metabolism: AgentMetabolism;
    centroid_map: CentroidMap;
    is_dirty: boolean;
}

// ===== VECTOR MATH =====

function cosineSimilarity(vecA: Float32Array, vecB: Float32Array): number {
    if (vecA.length === 0 || vecB.length === 0) return 0.0;

    let dotProduct = 0.0;
    let magA = 0.0;
    let magB = 0.0;

    const minLen = Math.min(vecA.length, vecB.length);
    for (let i = 0; i < minLen; i++) {
        const a = vecA[i];
        const b = vecB[i];
        dotProduct += a * b;
        magA += a * a;
        magB += b * b;
    }

    magA = Math.sqrt(magA);
    magB = Math.sqrt(magB);

    if (magA === 0.0 || magB === 0.0) return 0.0;
    return dotProduct / (magA * magB);
}

function temporalDecay(createdTime: number, currentTime: number): number {
    if (createdTime > currentTime) return 1.0;

    const ageSeconds = currentTime - createdTime;
    const decay = Math.exp(-ageSeconds / CONFIG.TEMPORAL_DECAY_HALF_LIFE);
    return Math.max(0.0, Math.min(1.0, decay));
}

// ===== MODULE 1: METABOLISM ENGINE =====

function initializeMetabolism(
    graph: HoneycombGraph,
    maxMessages: number,
    maxMinutes: number,
    isApiMode: boolean
): void {
    graph.metabolism = {
        messages_remaining: maxMessages,
        minutes_remaining: maxMinutes * 60,
        is_api_mode: isApiMode,
        context_availability: 0.0,
        metabolic_weight: 1.0,
        state: MetabolicState.Healthy,
        audit_last_run: Date.now() / 1000
    };
    console.log(
        `✅ Initialized Metabolism: messages=${maxMessages}, minutes=${maxMinutes}, api_mode=${isApiMode}`
    );
}

function updateMetabolism(
    graph: HoneycombGraph,
    messagesUsed: number,
    secondsElapsed: number,
    contextUsed: number
): void {
    graph.metabolism.messages_remaining -= messagesUsed;
    graph.metabolism.minutes_remaining -= secondsElapsed;
    graph.metabolism.context_availability = Math.min(100.0, contextUsed);

    if (graph.metabolism.minutes_remaining < 300 || graph.metabolism.messages_remaining < 5) {
        graph.metabolism.state = MetabolicState.Critical;
        graph.metabolism.metabolic_weight = 1.5;
    } else if (graph.metabolism.minutes_remaining < 1080 || graph.metabolism.messages_remaining < 20) {
        graph.metabolism.state = MetabolicState.Stressed;
        graph.metabolism.metabolic_weight = 1.2;
    } else {
        graph.metabolism.state = MetabolicState.Healthy;
        graph.metabolism.metabolic_weight = 1.0;
    }

    const stateMap = { [MetabolicState.Healthy]: 'HEALTHY', [MetabolicState.Stressed]: 'STRESSED', [MetabolicState.Critical]: 'CRITICAL' };
    console.log(
        `🔄 Metabolism Updated: state=${stateMap[graph.metabolism.state]}, weight=${graph.metabolism.metabolic_weight.toFixed(2)}, context=${graph.metabolism.context_availability.toFixed(1)}%`
    );
}

function calculateMetabolicRelevance(
    vecA: Float32Array,
    vecB: Float32Array,
    createdTime: number,
    currentTime: number,
    resourceAvail: number,
    metabolicWeight: number
): number {
    const semantic = cosineSimilarity(vecA, vecB);
    const decay = temporalDecay(createdTime, currentTime);
    const resource = 1.0 - (resourceAvail / 100.0);

    const final = ((semantic * 0.6) + (decay * 0.2) + (resource * 0.2)) * metabolicWeight;
    return Math.max(0.0, Math.min(1.0, final));
}

// ===== MODULE 2: CENTROID INDEXING =====

function initializeCentroidMap(graph: HoneycombGraph): void {
    const maxHubs = Math.max(1, Math.floor(graph.node_count * CONFIG.CENTROID_SCAN_PERCENTAGE));
    graph.centroid_map.max_hubs = Math.min(maxHubs, CONFIG.CENTROID_COUNT);
    console.log(`✅ Initialized Centroid Map: max_hubs=${graph.centroid_map.max_hubs}`);
}

function recalculateCentrality(graph: HoneycombGraph): void {
    if (graph.node_count === 0) return;

    const centrality = new Map<number, number>();

    for (const [nodeId, node] of graph.nodes) {
        if (node.is_active) {
            const degree = node.neighbors.length / CONFIG.HEXAGONAL_NEIGHBORS;
            const avgRelevance =
                node.neighbors.length > 0
                    ? node.neighbors.reduce((s, e) => s + e.relevance_score, 0) / node.neighbors.length
                    : 0.0;

            const score = degree * 0.6 + avgRelevance * 0.4;
            centrality.set(nodeId, score);
            node.metabolic_weight = score;
        }
    }

    const sorted = Array.from(centrality.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, graph.centroid_map.max_hubs);

    graph.centroid_map.hub_node_ids = sorted.map((x) => x[0]);
    graph.centroid_map.hub_centrality = sorted.map((x) => x[1]);

    console.log(`✅ Recalculated Centrality: found ${graph.centroid_map.hub_node_ids.length} hubs`);
}

function findMostRelevantNode(graph: HoneycombGraph, queryVector: Float32Array): number | null {
    if (graph.node_count === 0 || queryVector.length === 0) return null;

    // Phase 1: Scan hubs
    let bestHubId: number | null = null;
    let bestHubScore = -1.0;

    for (const hubId of graph.centroid_map.hub_node_ids) {
        const node = graph.nodes.get(hubId);
        if (node) {
            const score = cosineSimilarity(queryVector, node.vector_embedding);
            if (score > bestHubScore) {
                bestHubScore = score;
                bestHubId = hubId;
            }
        }
    }

    if (bestHubId === null) {
        for (const [nodeId, node] of graph.nodes) {
            if (node.is_active) {
                bestHubId = nodeId;
                break;
            }
        }
    }

    if (bestHubId === null) return null;

    // Phase 2: Refine with neighbors
    let bestNodeId = bestHubId;
    let bestScore = cosineSimilarity(queryVector, graph.nodes.get(bestHubId)!.vector_embedding);

    for (const edge of graph.nodes.get(bestHubId)!.neighbors) {
        const neighbor = graph.nodes.get(edge.target_id);
        if (neighbor) {
            const score = cosineSimilarity(queryVector, neighbor.vector_embedding);
            if (score > bestScore) {
                bestScore = score;
                bestNodeId = edge.target_id;
            }
        }
    }

    console.log(`✅ Found entry node: ${bestNodeId} (score=${bestScore.toFixed(3)})`);
    return bestNodeId;
}

// ===== MODULE 3: PERSISTENCE =====

async function saveBinary(graph: HoneycombGraph, filename: string): Promise<number> {
    const data = {
        header: 'OM_VINAYAKA',
        node_count: graph.node_count,
        max_nodes: graph.max_nodes,
        metabolism: graph.metabolism,
        nodes: {} as Record<number, any>
    };

    for (const [nodeId, node] of graph.nodes) {
        data.nodes[nodeId] = {
            id: node.id,
            embedding: Array.from(node.vector_embedding),
            data: node.data,
            metabolic_weight: node.metabolic_weight,
            is_fractal_seed: node.is_fractal_seed,
            neighbors: node.neighbors.map((e) => ({
                target_id: e.target_id,
                relevance_score: e.relevance_score,
                relationship_type: e.relationship_type
            }))
        };
    }

    return new Promise((resolve, reject) => {
        fs.writeFile(filename, JSON.stringify(data, null, 2), (err) => {
            if (err) {
                console.error(`❌ Save failed: ${err}`);
                reject(err);
            } else {
                console.log(`✅ Graph saved to ${filename}`);
                resolve(0);
            }
        });
    });
}

async function loadBinary(filename: string): Promise<HoneycombGraph | null> {
    return new Promise((resolve) => {
        fs.readFile(filename, 'utf8', (err, data) => {
            if (err) {
                console.error(`❌ Load failed: ${err}`);
                resolve(null);
                return;
            }

            try {
                const parsed = JSON.parse(data);
                if (parsed.header !== 'OM_VINAYAKA') {
                    resolve(null);
                    return;
                }

                const graph = createGraph('loaded_graph', parsed.max_nodes, CONFIG.MAX_SESSION_TIME);
                graph.metabolism = parsed.metabolism;

                for (const [nodeId, nodeData] of Object.entries(parsed.nodes)) {
                    const nid = parseInt(nodeId);
                    const embedding = new Float32Array((nodeData as any).embedding);
                    addNode(graph, embedding, (nodeData as any).data);
                }

                for (const [nodeId, nodeData] of Object.entries(parsed.nodes)) {
                    const nid = parseInt(nodeId);
                    if (graph.nodes.has(nid)) {
                        for (const edgeData of (nodeData as any).neighbors) {
                            addEdge(graph, nid, edgeData.target_id, edgeData.relevance_score, edgeData.relationship_type);
                        }
                    }
                }

                console.log(`✅ Graph loaded from ${filename} (nodes=${graph.node_count})`);
                resolve(graph);
            } catch (e) {
                console.error(`❌ Parse failed: ${e}`);
                resolve(null);
            }
        });
    });
}

function exportGraphviz(graph: HoneycombGraph, filename: string): void {
    let content = 'digraph HoneycombGraph {\n';
    content += '  rankdir=LR;\n';
    content += '  label="OV-Memory Fractal Honeycomb\\n(Om Vinayaka)";\n';

    for (const [nodeId, node] of graph.nodes) {
        if (!node.is_active) continue;

        let color = 'green';
        if (node.metabolic_weight < 0.5) color = 'red';
        else if (node.metabolic_weight < 0.8) color = 'orange';

        const shape = node.is_fractal_seed ? 'doubleoctagon' : 'circle';
        content += `  node_${nodeId} [label="N${nodeId}", color=${color}, shape=${shape}];\n`;
    }

    for (const [nodeId, node] of graph.nodes) {
        for (const edge of node.neighbors) {
            content += `  node_${nodeId} -> node_${edge.target_id} [label="${edge.relevance_score.toFixed(2)}"];\n`;
        }
    }

    content += '}\n';
    fs.writeFileSync(filename, content);
    console.log(`✅ Exported to GraphViz: ${filename}`);
}

// ===== MODULE 4: HYDRATION =====

function createFractalSeed(graph: HoneycombGraph, seedLabel: string): number | null {
    const activeNodes = Array.from(graph.nodes.values()).filter((n) => n.is_active);
    if (activeNodes.length === 0) return null;

    const seedEmbedding = new Float32Array(CONFIG.MAX_EMBEDDING_DIM);
    for (const node of activeNodes) {
        for (let i = 0; i < node.vector_embedding.length; i++) {
            seedEmbedding[i] += node.vector_embedding[i];
        }
    }
    for (let i = 0; i < seedEmbedding.length; i++) {
        seedEmbedding[i] /= activeNodes.length;
    }

    const seedId = addNode(graph, seedEmbedding, seedLabel);
    if (seedId !== null) {
        const seedNode = graph.nodes.get(seedId)!;
        seedNode.is_fractal_seed = true;
        console.log(`✅ Created Fractal Seed: ${seedId} from ${activeNodes.length} nodes`);
    }

    return seedId;
}

// ===== GRAPH OPERATIONS =====

function createGraph(
    name: string,
    maxNodes: number = CONFIG.MAX_NODES,
    maxSessionTime: number = CONFIG.MAX_SESSION_TIME
): HoneycombGraph {
    const graph: HoneycombGraph = {
        name,
        nodes: new Map(),
        node_count: 0,
        max_nodes: maxNodes,
        session_start_time: Date.now() / 1000,
        max_session_time_seconds: maxSessionTime,
        metabolism: {
            messages_remaining: 100,
            minutes_remaining: (maxSessionTime / 60) * 60,
            is_api_mode: false,
            context_availability: 0.0,
            metabolic_weight: 1.0,
            state: MetabolicState.Healthy,
            audit_last_run: Date.now() / 1000
        },
        centroid_map: {
            hub_node_ids: [],
            hub_centrality: [],
            max_hubs: CONFIG.CENTROID_COUNT
        },
        is_dirty: false
    };

    initializeMetabolism(graph, 100, maxSessionTime / 60, false);
    initializeCentroidMap(graph);
    console.log(`✅ Created honeycomb graph: ${name} (max_nodes=${maxNodes})`);
    return graph;
}

function addNode(graph: HoneycombGraph, embedding: Float32Array, data: string): number | null {
    if (graph.node_count >= graph.max_nodes) return null;

    const nodeId = graph.node_count;
    const node: HoneycombNode = {
        id: nodeId,
        vector_embedding: embedding,
        data: data.substring(0, 8192),
        neighbors: [],
        last_accessed_timestamp: Date.now() / 1000,
        access_count_session: 0,
        access_time_first: 0.0,
        relevance_to_focus: 0.0,
        metabolic_weight: 1.0,
        is_active: true,
        is_fractal_seed: false
    };

    graph.nodes.set(nodeId, node);
    graph.node_count++;
    graph.is_dirty = true;

    console.log(`✅ Added node ${nodeId} (embedding_dim=${embedding.length})`);
    return nodeId;
}

function addEdge(
    graph: HoneycombGraph,
    sourceId: number,
    targetId: number,
    relevanceScore: number,
    relationshipType: string
): boolean {
    if (!graph.nodes.has(sourceId) || !graph.nodes.has(targetId)) return false;

    const source = graph.nodes.get(sourceId)!;
    if (source.neighbors.length >= CONFIG.HEXAGONAL_NEIGHBORS) return false;

    const edge: HoneycombEdge = {
        target_id: targetId,
        relevance_score: Math.max(0.0, Math.min(1.0, relevanceScore)),
        relationship_type: relationshipType,
        timestamp_created: Date.now() / 1000
    };

    source.neighbors.push(edge);
    graph.is_dirty = true;

    console.log(`✅ Added edge: ${sourceId} → ${targetId} (relevance=${relevanceScore.toFixed(2)})`);
    return true;
}

function printGraphStats(graph: HoneycombGraph): void {
    const totalEdges = Array.from(graph.nodes.values()).reduce((sum, node) => sum + node.neighbors.length, 0);

    console.log('\n' + '='.repeat(40));
    console.log('GRAPH STATISTICS');
    console.log('='.repeat(40));
    console.log(`Graph Name: ${graph.name}`);
    console.log(`Node Count: ${graph.node_count} / ${graph.max_nodes}`);
    console.log(`Total Edges: ${totalEdges}`);
    console.log(`Centroid Hubs: ${graph.centroid_map.hub_node_ids.length}\n`);
}

function printMetabolicState(graph: HoneycombGraph): void {
    const stateMap = { [MetabolicState.Healthy]: 'HEALTHY', [MetabolicState.Stressed]: 'STRESSED', [MetabolicState.Critical]: 'CRITICAL' };
    console.log('\n' + '='.repeat(40));
    console.log('METABOLIC STATE REPORT');
    console.log('='.repeat(40));
    console.log(`State: ${stateMap[graph.metabolism.state]}`);
    console.log(`Messages Left: ${graph.metabolism.messages_remaining}`);
    console.log(`Time Left: ${graph.metabolism.minutes_remaining} sec`);
    console.log(`Context Used: ${graph.metabolism.context_availability.toFixed(1)}%`);
    console.log(`Metabolic Weight: ${graph.metabolism.metabolic_weight.toFixed(2)}\n`);
}

// ===== MAIN TEST =====

async function main(): Promise<void> {
    console.log('\n🧠 OV-Memory v1.1 - TypeScript Implementation');
    console.log('Om Vinayaka 🙏\n');

    const graph = createGraph('metabolic_test', 100, 3600);

    const emb1 = new Float32Array(CONFIG.MAX_EMBEDDING_DIM).fill(0.5);
    const emb2 = new Float32Array(CONFIG.MAX_EMBEDDING_DIM).fill(0.6);
    const emb3 = new Float32Array(CONFIG.MAX_EMBEDDING_DIM).fill(0.7);

    const node1 = addNode(graph, emb1, 'Memory Alpha');
    const node2 = addNode(graph, emb2, 'Memory Beta');
    const node3 = addNode(graph, emb3, 'Memory Gamma');

    if (node1 !== null && node2 !== null) addEdge(graph, node1, node2, 0.9, 'related_to');
    if (node2 !== null && node3 !== null) addEdge(graph, node2, node3, 0.85, 'context_of');

    recalculateCentrality(graph);
    updateMetabolism(graph, 10, 120, 45.0);
    printMetabolicState(graph);

    const entryNode = findMostRelevantNode(graph, emb1);
    console.log(`Entry node: ${entryNode}\n`);

    await saveBinary(graph, 'test_graph.json');
    exportGraphviz(graph, 'test_graph.dot');

    const seedId = createFractalSeed(graph, 'session_seed');
    printGraphStats(graph);

    console.log('✅ v1.1 tests completed');
    console.log('Om Vinayaka 🙏\n');
}

// Export for use as module
export {
    createGraph,
    addNode,
    addEdge,
    initializeMetabolism,
    updateMetabolism,
    calculateMetabolicRelevance,
    initializeCentroidMap,
    recalculateCentrality,
    findMostRelevantNode,
    saveBinary,
    loadBinary,
    exportGraphviz,
    createFractalSeed,
    printMetabolicState,
    printGraphStats,
    // Types
    MetabolicState,
    SafetyCode,
    type AgentMetabolism,
    type HoneycombNode,
    type HoneycombEdge,
    type HoneycombGraph,
    type CentroidMap
};

// Main execution
if (require.main === module) {
    main().catch(console.error);
}

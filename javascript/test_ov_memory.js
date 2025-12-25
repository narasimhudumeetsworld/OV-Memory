/**
 * OV-Memory: JavaScript Test Suite
 * Om Vinayaka 🙏
 */

// Load the implementation
const code = require('fs').readFileSync('./ov_memory.js', 'utf8');
eval(code);

console.log('\n🧠 OV-Memory: JavaScript Tests');
console.log('Om Vinayaka 🙏\n');

// Test helper
function assert(condition, message) {
    if (!condition) {
        console.error(`❌ FAILED: ${message}`);
        process.exit(1);
    }
}

// Create graph
const graph = honeycombCreateGraph('test_memory', 1000, 3600);
assert(graph !== null, 'Graph created');
assert(graph.name === 'test_memory', 'Graph name matches');
console.log('✅ Graph creation test passed');

// Create embeddings
const emb1 = new Float32Array(768).fill(0.5);
const emb2 = new Float32Array(768).fill(0.6);
const emb3 = new Float32Array(768).fill(0.7);

// Add nodes
const node1 = honeycombAddNode(graph, emb1, 'First memory unit');
const node2 = honeycombAddNode(graph, emb2, 'Second memory unit');
const node3 = honeycombAddNode(graph, emb3, 'Third memory unit');

assert(node1 === 0, 'Node 1 ID correct');
assert(node2 === 1, 'Node 2 ID correct');
assert(node3 === 2, 'Node 3 ID correct');
console.log('✅ Node addition test passed');

// Add edges
const edge1 = honeycombAddEdge(graph, node1, node2, 0.9, 'related_to');
const edge2 = honeycombAddEdge(graph, node2, node3, 0.85, 'context_of');

assert(edge1 === true, 'Edge 1 added');
assert(edge2 === true, 'Edge 2 added');
console.log('✅ Edge addition test passed');

// Insert memory
honeycombInsertMemory(graph, node1, node2);
console.log('✅ Memory insertion test passed');

// Get node
const retrievedNode = honeycombGetNode(graph, node1);
assert(retrievedNode !== null, 'Node retrieved');
assert(retrievedNode.id === node1, 'Retrieved node ID correct');
console.log('✅ Node retrieval test passed');

// Check safety
const safetyStatus = honeycombCheckSafety(retrievedNode);
assert(safetyStatus === 0, 'Safety status OK');
console.log('✅ Safety check test passed');

// Vector similarity
const sim = cosineSimilarity(emb1, emb1);
assert(sim > 0.99, 'Cosine similarity correct (self-similarity = 1.0)');
console.log('✅ Vector similarity test passed');

// Temporal decay
const decay1 = temporalDecay(Date.now() / 1000, Date.now() / 1000);
assert(decay1 > 0.99, 'Temporal decay correct (current time = 1.0)');
console.log('✅ Temporal decay test passed');

// Print stats
honeycombPrintGraphStats(graph);

console.log('✅ All JavaScript tests passed!');
console.log('Om Vinayaka 🙏\n');

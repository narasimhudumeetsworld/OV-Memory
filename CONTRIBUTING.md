# 🙤 Contributing to OV-Memory

**Om Vinayaka 🙏**

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing.

---

## 👥 Code of Conduct

- Be respectful and inclusive
- No discrimination, harassment, or hate speech
- Welcome all skill levels
- Constructive feedback only
- Report issues to: [maintainers]

---

## 🌟 Ways to Contribute

### 1. **Report Bugs**
- Create an issue with detailed reproduction steps
- Include environment info (OS, language version, etc.)
- Add relevant code examples
- Label as `bug`

### 2. **Suggest Features**
- Describe the feature and use case
- Explain why it's needed
- Provide examples if possible
- Label as `enhancement`

### 3. **Fix Bugs**
- Pick an issue labeled `bug`
- Comment to avoid duplicate work
- Create a PR with your fix
- Include tests

### 4. **Add Features**
- Pick an issue labeled `enhancement`
- Discuss approach in comments
- Implement with tests
- Update documentation

### 5. **Improve Documentation**
- Fix typos and unclear sections
- Add examples and clarifications
- Translate to other languages
- Add to FAQ or troubleshooting

### 6. **Write Tests**
- Add edge case tests
- Improve coverage
- Test performance
- Test cross-platform compatibility

### 7. **Optimize Performance**
- Profile and benchmark
- Identify bottlenecks
- Propose and test optimizations
- Document improvements

---

## 🚀 Getting Started

### Step 1: Fork & Clone
```bash
git clone https://github.com/YOUR_USERNAME/OV-Memory.git
cd OV-Memory
git remote add upstream https://github.com/narasimhudumeetsworld/OV-Memory.git
```

### Step 2: Create Feature Branch
```bash
git checkout -b feature/your-feature-name
# or for bugs:
git checkout -b fix/issue-description
```

### Step 3: Set Up Development Environment

#### **For All Languages**
```bash
# Install Python dependencies
cd python && pip install -r requirements.txt && cd ..

# Install Node dependencies
cd javascript && npm install && cd ..

# Install C tools
cd c && make clean && cd ..
```

#### **For Specific Languages**

**Python**:
```bash
cd python
pip install -r requirements.txt
python -m pytest ov_memory_test.py -v
```

**JavaScript**:
```bash
cd javascript
npm install
node test_ov_memory.js
```

**C**:
```bash
cd c
make build
make test
```

### Step 4: Make Your Changes
- Follow code style (see below)
- Write clear commit messages
- Add tests for new features
- Update documentation

### Step 5: Test Your Changes
```bash
# Run tests for your language
cd [language] && [test command]

# Run all tests
make test-all
```

### Step 6: Push & Create PR
```bash
git push origin feature/your-feature-name
```

Then create a PR on GitHub with:
- Clear title
- Description of changes
- Reference to related issues (#123)
- Screenshots/benchmarks if applicable

---

## 💫 Code Style Guide

### **Python**
```python
# Follow PEP 8
# Use type hints
def add_node(graph: HoneycombGraph, embedding: np.ndarray, data: str) -> int:
    """Add a node to the graph."""
    pass

# Docstrings for all public functions
# Use 4-space indentation
# Max line length: 100 characters
```

### **JavaScript**
```javascript
// Use const/let, not var
const maxNodes = 100000;

// JSDoc comments
/**
 * Add a node to the graph
 * @param {HoneycombGraph} graph - The graph
 * @param {Float32Array} embedding - Vector embedding
 * @returns {number} Node ID
 */
function honeycombAddNode(graph, embedding, data) {
    // Implementation
}

// 2-space indentation
// Use arrow functions where appropriate
// No semicolons (Prettier formatted)
```

### **C**
```c
// Use lowercase with underscores for functions
int honeycomb_add_node(HoneycombGraph *graph, const float *embedding,
                      uint32_t dim, const char *data);

// Clear type definitions
typedef struct HoneycombNode { ... } HoneycombNode;

// Comments for complex logic
// Use snake_case for variables
const uint32_t max_nodes = 100000;

// 4-space indentation
```

### **General**
- Write self-documenting code
- Use meaningful variable names
- Add comments for WHY, not WHAT
- Keep functions small and focused
- Follow DRY (Don't Repeat Yourself)
- No hard-coded magic numbers (use constants)

---

## 🙈 Testing Requirements

### All Contributions Must Include:

1. **Unit Tests**
   - Test main functionality
   - Test edge cases
   - Test error handling

2. **Integration Tests** (if applicable)
   - Test with multiple components
   - Test with real embeddings

3. **Performance Tests** (for performance changes)
   - Benchmark before/after
   - Show results in PR

4. **Documentation Tests**
   - Verify code examples work
   - Test in actual environment

### Running Tests

```bash
# Python
cd python && python -m pytest ov_memory_test.py -v --cov

# JavaScript
cd javascript && npm test

# C
cd c && make test

# All
make test-all
```

---

## 📝 Commit Message Guidelines

### Format
```
<type>: <subject>

<body>

<footer>
```

### Types
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `style:` - Code style (formatting)
- `refactor:` - Code refactoring
- `perf:` - Performance improvement
- `test:` - Test additions/changes
- `chore:` - Build, deps, etc.

### Examples
```
feat: Add temporal decay for memory relevance

Implement exponential decay based on memory age.
Memory older than 24 hours gets scaled down.

Closes #123
```

```
fix: Prevent infinite loop in fractal insertion

Check recursion depth before creating new layer.

Fixes #456
```

---

## 🔍 Code Review Process

### Expectations
1. Maintainers review within 48 hours
2. Response to feedback within 7 days
3. Tests must pass
4. No merge without 2+ approvals
5. Coverage should not decrease

### Review Criteria
- Code quality and style
- Test coverage (>80%)
- Documentation updates
- Performance impact
- Security considerations
- Backward compatibility

---

## 🐗 Documentation

### Update These When Applicable

1. **README.md** - Features, quick start
2. **ARCHITECTURE.md** - System design
3. **API.md** - API changes
4. **PERFORMANCE.md** - Benchmark changes
5. **Code Comments** - Docstrings, inline comments

### Documentation Format
```markdown
## Feature Name

### Description
What does it do?

### Usage
```code example
```

### Performance
BigO complexity, benchmarks

### References
Links to related docs
```

---

## 🌟 Git Workflow

### Keep Fork in Sync
```bash
git fetch upstream
git rebase upstream/main
git push -f origin main
```

### Before Creating PR
```bash
# Update branch
git fetch upstream
git rebase upstream/main

# Run all tests
make test-all

# Check code style
make lint

# Commit with good messages
git add .
git commit -m "feat: clear description"
```

### PR Review Loop
```bash
# Implement feedback
git add .
git commit -m "Address review feedback"
git push origin feature/your-feature

# Maintainer will re-review
```

---

## 👘 Performance Expectations

### Benchmarks
```
Add Node:        < 100 µs
Add Edge:        < 50 µs
Get Node:        < 10 µs
JIT Context:     < 5 ms
Safety Check:    < 1 µs
```

### Memory Limits
```
Per Node:        ~1.2 KB (768-dim embedding)
Per Graph:       ~120 MB (100K nodes)
Stack Usage:     < 100 KB
```

---

## 📊 Version Numbers

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New backward-compatible features
- **PATCH**: Bug fixes

Example: `1.0.0`

---

## 👈 Pull Request Template

```markdown
## Description
Clear description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Tests added/updated
- [ ] All tests passing
- [ ] No coverage decrease

## Documentation
- [ ] README updated
- [ ] API docs updated
- [ ] Code comments added

## Related Issues
Closes #123

## Checklist
- [ ] Code follows style guide
- [ ] No new warnings
- [ ] Tests pass locally
- [ ] Documentation updated
```

---

## 🍐 Development Timeline

### Typical Flow
1. **Issue Discussion**: 1-2 days
2. **Development**: 3-7 days
3. **Testing**: 1-2 days
4. **Review**: 1-3 days
5. **Merge**: Immediate
6. **Release**: Next sprint

---

## 🙏 Acknowledgments

All contributors are recognized in:
- [Contributors List](https://github.com/narasimhudumeetsworld/OV-Memory/contributors)
- Release notes
- Annual contributor report

---

## 🐧 Questions?

- **Issues**: Create a discussion issue
- **Slack**: [Join our community](#)
- **Email**: Contact maintainers
- **Docs**: Check [ARCHITECTURE.md](ARCHITECTURE.md)

---

**Thank you for contributing to OV-Memory!**

**Om Vinayaka 🙏**

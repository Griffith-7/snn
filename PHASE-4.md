# PHASE 4: Release — IN PROGRESS

## Goal
Prepare exact-snn for public release: paper, PyPI, documentation.

## Prerequisites
- Phase 2 complete (multi-spike) ✅
- Phase 3 complete (real benchmarks) ✅

## What Needs to Happen

### Step 1: Code Quality (Week 1-2)
- [ ] Type hints on all public functions
- [ ] Docstrings on all classes and methods
- [ ] Clean API surface (remove internal helpers from public exports)
- [ ] Remove dead code, unused imports
- [ ] Run pytest, fix any failures

### Step 2: Documentation (Week 2-4)
- [ ] Full README with install, quick start, API reference
- [ ] API documentation (what each function does, parameters, returns)
- [ ] Examples: simple training, multi-spike, convolutional, event-driven
- [ ] Migration guide: from snnTorch/Norse to exact-snn

### Step 3: Paper Draft (Week 4-8)
- [ ] Title: "Exact Gradient Training for Spiking Neural Networks via Implicit Function Theorem"
- [ ] Abstract, Introduction, Method, Experiments, Conclusion
- [ ] Figures: gradient accuracy, training curves, architecture diagrams
- [ ] Comparison tables: exact vs surrogate on real benchmarks
- [ ] Limitations section (honest)
- [ ] Submit to arXiv or conference (NeurIPS, ICLR, ICML)

### Step 4: PyPI Release (Week 8-10)
- [ ] Clean pyproject.toml
- [ ] Version bump to 1.1.0 (multi-spike)
- [ ] `pip install exact-snn` works
- [ ] Release notes
- [ ] GitHub release with changelog

### Step 5: Community (Week 10-12)
- [ ] Write blog post / Twitter thread
- [ ] Respond to issues on GitHub
- [ ] Add examples for common use cases
- [ ] Consider: Colab notebook for quick demo

## Success Criteria
- [ ] Paper submitted (arXiv or conference)
- [ ] PyPI package installable
- [ ] README complete with examples
- [ ] At least 10 GitHub stars (organic interest)
- [ ] Zero known critical bugs

# ROADMAP

## Vision
Build a production-ready SNN training library with exact gradients that works on general spiking networks, not just TTFS single-spike.

## Phases

### Phase 1: Foundation ✅ COMPLETE
- Exact IFT gradients for TTFS single-spike
- Event-driven engine (2.89x faster)
- Saltation matrices (in extended)
- Conv/Recurrent/BatchNorm (in extended)
- Library packaged as `exact-snn` v1.0.0
- Pushed to GitHub

### Phase 2: General Multi-Spike (Months 1-3)
- Move saltation from extended to core
- Multi-spike as default training mode
- General event-driven engine for multi-spike
- Multiple resets per neuron handled correctly
- Verified gradients on multi-spike networks
- Target: train general SNNs with exact gradients

### Phase 3: Real Benchmarks ✅ COMPLETE
- Vectorized GPU forward/backward (eliminated Python loops)
- Fixed autograd memory leak (torch.no_grad() wrapper)
- Conv SNN learns CIFAR-10: loss 7.7 → 2.6 in 1 epoch (5K subset)
- FC-only TTFS confirmed unable to learn CIFAR-10 (10% = random)
- Convolutional architecture: Conv(3→16)→Pool→Conv(16→32)→Pool→FC(2048→64→10)
- Memory stable at 13MB (was 2GB/step leak before fix)
- Honest reporting: documented failures AND successes

### Phase 4: Release ✅ COMPLETE
- Code quality: type hints on all 82 public entities, 39 docstrings added
- Documentation: README with examples, API reference, architecture table
- Paper draft: paper/paper.tex (LaTeX, arXiv-ready)
- PyPI: exact-snn v1.1.0 built (sdist + wheel)
- Clean API: removed dead imports, dead code, private leaks from __all__

## Rules
1. Every claim must be verified with code + experiments
2. Failures reported as prominently as wins
3. No overstating results
4. Each phase completed before moving to next
5. MEMORY.md updated after every significant session


# SVG Vision v8 — Architecture Summary

## Core Philosophy: Pipeline Logic (Non-Hardcoded)

### 1. Composable Stages (Not Monolithic)
```python
Pipeline = LoadStage | AnalysisStage | RouterStage | PreprocessStage | ...
```
- Each stage is independent, swappable, testable
- Stages compose via `|` operator (fluent API)
- New stage = new class, no edit to core logic

### 2. Strategy Registry (Not Switch/Case)
```python
RouterStage.TEMPLATES = {
    "pixel_art": {"quantize_method": "exact", "use_lab": False, ...},
    "vector_approx": {"quantize_method": "kmeans_rich", "use_lab": True, ...},
    # Add new strategy = add entry, no code change elsewhere
}
```

### 3. Adaptive Router (Continuous, Not Discrete)
- Scoring: `pa_score = quant_score * 3.0 + outline_bonus + entropy_penalty + edge_bonus`
- Confidence = max_score / total_scores (continuous 0-1)
- No hardcoded thresholds — analysis drives routing

### 4. Fallback Chain (Resilient)
```
User forced strategy → Auto-detect → Hybrid fallback (confidence < 0.35)
```

### 5. Config-Driven Execution
- Strategy template = config dict, not code branch
- All parameters injectable via PipelineContext.user_params
- No magic numbers in execution path

### 6. Field-Based Computation (Tensor-Driven)
- Multi-scale analysis = weighted voting (wave interference)
- CIELab color space = continuous perceptual field
- Connected component labeling = continuous region fields
- No discrete entity spawn/kill

### 7. SOA + Pre-allocated Buffers
- PipelineContext.ensure_buf_rgb/lab = ghost state reuse
- Quantizer global instance = buffer reuse across calls
- Vectorized numpy ops, minimal Python branching

## Test Results (Pixel Art House)
| Test | Strategy | Confidence | Colors | Status |
|------|----------|------------|--------|--------|
| Auto | pixel_art | 43.1% | 20 | ✓ |
| Forced PixelArt | pixel_art | 100% | 20 | ✓ |
| Forced Vector | vector_approx | 100% | 23 | ✓ |
| Forced Photo | photo_simplified | 100% | 7 | ✓ |
| Custom Pipeline | photo_simplified | — | 7 | ✓ |

## File Size (Path-based SVG)
- Auto: ~8.7 KB
- PixelArt: ~8.7 KB
- Vector: ~301 KB (path fragmentation)
- Photo: ~82 KB

## API Compatibility
- Legacy `image_to_grid()` → wraps Pipeline internally
- New `Pipeline.auto_from_image()` → composable, extensible
- Custom `Pipeline() | Stage1 | Stage2 | ...` → full control

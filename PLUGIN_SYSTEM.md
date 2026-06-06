
# SVG Vision v8.1 — Plugin System Documentation

## Overview

Plugin system memungkinkan ekstensi pipeline tanpa edit core code.
Plugins adalah fungsi Python yang diregister via decorator dan dieksekusi
pada hook points tertentu dalam pipeline lifecycle.

## Hook Points

| Hook | When | Use Case |
|------|------|----------|
| `pre_pipeline` | Before any stage runs | Setup, logging |
| `pre_load` / `post_load` | Around LoadStage | Custom image loading |
| `pre_analysis` / `post_analysis` | Around AnalysisStage | Custom metrics |
| `pre_router` / `post_router` | Around RouterStage | Override routing |
| `pre_quantize` / `post_quantize` | Around QuantizeStage | Dithering, palette mods |
| `pre_output` / `post_output` | Around OutputStage | SVG injection |
| `post_pipeline` | After all stages | Cleanup, stats |
| `fallback` | When stage fails | Recovery transforms |

## Plugin Priority

Lower = earlier execution (0-100). Default = 50.

```
Priority 0-20:   Pre-processing (setup, validation)
Priority 21-50:  Core transforms (dithering, enhancement)
Priority 51-80:  Post-processing (watermark, metadata)
Priority 81-100: Cleanup (stats, logging)
```

## Creating a Plugin

### Method 1: Decorator (Recommended)

```python
from vision import plugin, PipelineContext

@plugin("post_quantize", priority=30, 
        condition=lambda ctx: ctx.analysis["edge_density"] > 0.1)
def my_enhancer(ctx: PipelineContext):
    # Mutate ctx.grid, ctx.palette, or ctx.metadata
    ctx.metadata["enhanced"] = True
```

### Method 2: PluginSpec (Programmatic)

```python
from vision import PluginRegistry, PluginSpec

reg = PluginRegistry()
reg.register(PluginSpec(
    name="my_plugin",
    hook="post_quantize",
    priority=25,
    condition=lambda ctx: ctx.strategy_name == "pixel_art",
    transform=lambda ctx: ctx.grid.fill(0),  # example
))
```

### Method 3: External File (Auto-Discovery)

Create file in `plugins/` directory:

```python
# plugins/my_effect.py
from vision import plugin, PipelineContext

@plugin("post_quantize", priority=10)
def my_effect(ctx: PipelineContext):
    # Your transform here
    pass
```

Pipeline auto-discovers:
```python
pipe = PluginAwarePipeline.auto_with_plugins(
    "image.png", 
    plugin_dir="./plugins"
)
```

## Plugin Condition

Plugin hanya berjalan jika condition return True:

```python
@plugin("post_quantize", 
        condition=lambda ctx: ctx.user_params.get("enable_effect") == True)
```

## Plugin Context Access

Plugin menerima `PipelineContext` yang berisi:

| Attribute | Description |
|-----------|-------------|
| `ctx.image` | PIL Image |
| `ctx.arr` | NumPy array |
| `ctx.analysis` | Analysis dict |
| `ctx.strategy_name` | Selected strategy |
| `ctx.strategy_config` | Strategy config dict |
| `ctx.grid` | Quantized grid (post-quantize) |
| `ctx.palette` | Color palette |
| `ctx.color_map` | Index → hex mapping |
| `ctx.metadata` | Shared metadata bag |
| `ctx.user_params` | User-provided parameters |

## Built-in Plugins

| Plugin | Hook | Condition | Function |
|--------|------|-----------|----------|
| `dithering` | post_quantize | `dithering=True` | Floyd-Steinberg dithering |
| `outline_enhance` | post_quantize | `outline_enhance=True` | Thicken outlines |
| `metadata_inject` | pre_output | Always | Inject gen metadata |
| `analysis_validator` | post_analysis | Always | Validate analysis |

## Example: Complete Plugin

```python
# plugins/pixel_perfect.py
from vision import plugin, PipelineContext
import numpy as np

@plugin("post_quantize", priority=15,
        condition=lambda ctx: ctx.strategy_name == "pixel_art",
        name="pixel_perfect")
def ensure_pixel_perfect(ctx: PipelineContext):
    # Ensure no anti-aliasing artifacts
    grid = ctx.grid
    palette = ctx.palette

    # Snap colors to nearest palette entry
    # (Already done by quantizer, but this is example)

    ctx.metadata["pixel_perfect"] = True
```

## Error Handling

Default: plugin error → skip plugin, continue pipeline.

Abort mode:
```python
pipe.run_on("img.png", plugin_abort_on_error=True)
```

## Performance

- Plugin registry: singleton, O(1) lookup
- Condition check: O(1) per plugin
- Execution: inline, no subprocess overhead
- Discovery: one-time at pipeline creation

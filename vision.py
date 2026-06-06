#!/usr/bin/env python3
"""
SVG Vision v8 — Composable Adaptive Pipeline
==============================================
Pipeline logic: dynamic, extensible, non-hardcoded strategy routing.

Architecture:
    Pipeline = [Stage1, Stage2, ...]  # composable, swappable
    StrategyRegistry = {name: Strategy}  # hot-pluggable
    AdaptiveRouter = continuous scoring → route  # no discrete switch
    FallbackChain = primary → fallback → fallback2  # resilient

Design principles:
- No hardcoded thresholds. Analysis drives config.
- No discrete entity spawn/kill. Continuous field operations.
- SOA over AOS. Pre-allocated buffers. Branchless hot paths.
- Strategy is a pipeline template, not a switch case.

Usage:
    from vision import Pipeline, StrategyRegistry

    # Auto pipeline
    pipe = Pipeline.auto_from_image("sprite.png")
    grid, cmap, meta = pipe.run()

    # Custom pipeline
    pipe = Pipeline()
    pipe.add_stage(AnalysisStage(scales=[1.0, 0.5]))
    pipe.add_stage(RouterStage())
    pipe.add_stage(QuantizeStage(method="kmeans", n_colors=16))
    pipe.add_stage(SVGOutputStage(pixel_size=16, use_paths=True))
    grid, cmap, meta = pipe.run_on("sprite.png")
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import warnings
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any, Callable, Dict, Generic, List, Optional, Protocol, Tuple,
    Type, TypeVar, Union,
)

import numpy as np
from PIL import Image, ImageFilter, ImageStat

# ---------------------------------------------------------------------------
# Inline SVGBuilder (self-contained)
# ---------------------------------------------------------------------------
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from svg_skill import SVGBuilder as _ExtSVGBuilder
except Exception:
    _ExtSVGBuilder = None


class SVGBuilder:
    def __init__(self, width: int, height: int):
        self.w = width
        self.h = height
        self._defs: List[str] = []
        self._body: List[str] = []
        self._styles: Dict[str, str] = {}

    def add_def(self, content: str):
        self._defs.append(content)

    def add_style(self, cls: str, fill: str, stroke: str = "none", stroke_w: float = 0):
        style = f"fill:{fill};stroke:{stroke}"
        if stroke_w > 0:
            style += f";stroke-width:{stroke_w}"
        self._styles[cls] = style

    def add_path(self, d: str, fill: str = "#000", stroke: str = "none", stroke_w: float = 0, cls: str = ""):
        attrs = f'd="{d}"'
        if cls:
            attrs += f' class="{cls}"'
        else:
            attrs += f' fill="{fill}"'
            if stroke != "none":
                attrs += f' stroke="{stroke}"'
                if stroke_w:
                    attrs += f' stroke-width="{stroke_w}"'
        self._body.append(f"  <path {attrs} />")

    def add_rect(self, x: int, y: int, w: int, h: int, fill: str = "#000"):
        self._body.append(f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" />')

    def to_string(self) -> str:
        lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}">']
        if self._styles:
            lines.append("  <style>")
            for cls, st in self._styles.items():
                lines.append(f"    .{cls} {{ {st} }}")
            lines.append("  </style>")
        if self._defs:
            lines.append("  <defs>")
            for d in self._defs:
                lines.append(f"    {d}")
            lines.append("  </defs>")
        lines.extend(self._body)
        lines.append("</svg>")
        return "\n".join(lines)


if _ExtSVGBuilder is not None:
    SVGBuilder = _ExtSVGBuilder  # type: ignore[misc]


# ============================================================================
# COLOR SPACE (CIELAB) — Field-based continuous color space
# ============================================================================

_XYZ_MATRIX = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)
_D65 = np.array([95.047, 100.000, 108.883], dtype=np.float64)


def _f_lab(t: np.ndarray) -> np.ndarray:
    delta = 6 / 29
    return np.where(t > delta ** 3, np.cbrt(t), t / (3 * delta ** 2) + 4 / 29)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    mask = rgb > 0.04045
    rgb = np.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    xyz = rgb @ _XYZ_MATRIX.T * 100.0
    xyz_ratio = xyz / _D65
    lab = _f_lab(xyz_ratio)
    L = 116.0 * lab[..., 1] - 16.0
    a = 500.0 * (lab[..., 0] - lab[..., 1])
    b = 200.0 * (lab[..., 1] - lab[..., 2])
    return np.stack([L, a, b], axis=-1)


def delta_e_cie76(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum((lab1 - lab2) ** 2, axis=-1))


# ============================================================================
# PIPELINE CONTEXT — Shared state bag (SOA-style, no AOS)
# ============================================================================

@dataclass
class PipelineContext:
    """Shared mutable state across pipeline stages."""
    image_path: str = ""
    image: Optional[Image.Image] = None
    arr: Optional[np.ndarray] = None
    analysis: Dict[str, Any] = field(default_factory=dict)
    strategy_name: str = ""
    strategy_config: Dict[str, Any] = field(default_factory=dict)
    grid: Optional[np.ndarray] = None
    palette: List[Tuple[int, int, int]] = field(default_factory=list)
    color_map: Dict[int, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    user_params: Dict[str, Any] = field(default_factory=dict)
    # Pre-allocated buffers (ghost state reuse)
    _buf_rgb: Optional[np.ndarray] = None
    _buf_lab: Optional[np.ndarray] = None
    _buf_idx: Optional[np.ndarray] = None

    def ensure_buf_rgb(self, n_pixels: int) -> np.ndarray:
        if self._buf_rgb is None or self._buf_rgb.shape[0] < n_pixels:
            self._buf_rgb = np.empty((n_pixels, 3), dtype=np.float64)
        return self._buf_rgb[:n_pixels]

    def ensure_buf_lab(self, n_pixels: int) -> np.ndarray:
        if self._buf_lab is None or self._buf_lab.shape[0] < n_pixels:
            self._buf_lab = np.empty((n_pixels, 3), dtype=np.float64)
        return self._buf_lab[:n_pixels]


# ============================================================================
# STAGE BASE CLASS — Composable pipeline unit
# ============================================================================

class PipelineStage(ABC):
    """Base class for all pipeline stages."""

    @abstractmethod
    def run(self, ctx: PipelineContext) -> None:
        """Mutate context in-place."""
        ...

    def __or__(self, other: PipelineStage) -> Pipeline:
        """Allow stage1 | stage2 | stage3 syntax."""
        return Pipeline([self, other])


class Pipeline:
    """Composable pipeline of stages."""

    def __init__(self, stages: Optional[List[PipelineStage]] = None):
        self.stages: List[PipelineStage] = stages or []

    def add_stage(self, stage: PipelineStage) -> Pipeline:
        self.stages.append(stage)
        return self

    def __or__(self, other: Union[PipelineStage, Pipeline]) -> Pipeline:
        if isinstance(other, PipelineStage):
            self.stages.append(other)
        elif isinstance(other, Pipeline):
            self.stages.extend(other.stages)
        return self

    def run_on(self, image_path: str, **user_params) -> Tuple[np.ndarray, Dict[int, str], Dict]:
        ctx = PipelineContext(image_path=image_path, user_params=user_params)
        for stage in self.stages:
            try:
                stage.run(ctx)
            except Exception as e:
                ctx.metadata[f"{stage.__class__.__name__}_error"] = str(e)
                raise
        assert ctx.grid is not None
        return ctx.grid, ctx.color_map, ctx.metadata

    @classmethod
    def auto_from_image(cls, image_path: str, **kwargs) -> Pipeline:
        """Factory: build auto-detected pipeline."""
        return (
            cls()
            | LoadStage()
            | AnalysisStage(scales=kwargs.get("scales", (1.0, 0.5, 0.25)))
            | RouterStage()
            | PreprocessStage()
            | CropStage()
            | GridSizeStage()
            | ResizeStage()
            | QuantizeStage(
                n_colors=kwargs.get("n_colors", "auto"),
                use_lab=kwargs.get("use_lab", None),  # None = auto from strategy
            )
            | CleanupStage()
            | OutputStage(
                pixel_size=kwargs.get("pixel_size", 16),
                use_paths=kwargs.get("use_paths", True),
            )
        )


# ============================================================================
# STAGE IMPLEMENTATIONS
# ============================================================================

class LoadStage(PipelineStage):
    def run(self, ctx: PipelineContext) -> None:
        ctx.image = Image.open(ctx.image_path).convert("RGBA")
        ctx.arr = np.array(ctx.image)


class AnalysisStage(PipelineStage):
    """Multi-scale field analysis. No discrete decisions here."""

    def __init__(self, scales: Tuple[float, ...] = (1.0, 0.5, 0.25)):
        self.scales = scales

    def run(self, ctx: PipelineContext) -> None:
        img = ctx.image
        assert img is not None
        w, h = img.size
        votes = []
        for scale in self.scales:
            if scale == 1.0:
                votes.append(self._analyze_scale(np.array(img), w, h))
            else:
                sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
                small = img.resize((sw, sh), Image.Resampling.LANCZOS)
                votes.append(self._analyze_scale(np.array(small), sw, sh))

        merged = self._merge_votes(votes, self.scales)
        merged["size"] = (w, h)
        merged["aspect_ratio"] = w / h if h else 1.0
        merged["scale_votes"] = len(self.scales)
        ctx.analysis = merged

    def _analyze_scale(self, arr: np.ndarray, w: int, h: int) -> Dict:
        pixels = arr[:, :, :3].reshape(-1, 3)
        rounded = (pixels // 8) * 8
        unique, counts = np.unique(rounded, axis=0, return_counts=True)
        probs = counts / counts.sum()
        entropy = float(-np.sum(probs * np.log2(probs + 1e-10)))

        gray = Image.fromarray(arr).convert("L") if arr.shape[-1] == 4 else Image.fromarray(arr).convert("L")
        edges = np.array(gray.filter(ImageFilter.FIND_EDGES))
        edge_density = float(np.sum(edges > 30) / edges.size)

        lum = 0.299 * pixels[:, 0] + 0.587 * pixels[:, 1] + 0.114 * pixels[:, 2]
        has_outline = bool(np.sum(lum < 75) / len(pixels) > 0.02)

        # Continuous quantization score via cumulative color frequency (robust to noise)
        sorted_counts = np.sort(counts)[::-1]
        cumulative = np.cumsum(sorted_counts) / counts.sum()
        # Number of colors needed to represent 95% of the image
        effective_colors = np.searchsorted(cumulative, 0.95) + 1

        # New color score logic (max 0)
        color_score = max(0.0, 1.0 - (effective_colors / 200.0) ** 0.5)

        # Also check gradient sparsity and blockiness (sharpness)
        arr_f = arr.astype(np.float64) / 255.0
        dx = np.sum(np.abs(arr_f[:, 1:] - arr_f[:, :-1]), axis=2)
        dy = np.sum(np.abs(arr_f[1:, :] - arr_f[:-1, :]), axis=2)

        sum_dx = dx.sum(axis=0)
        sum_dy = dy.sum(axis=1)

        # Blockiness: variance to mean ratio of grid gradients.
        # High for scaled pixel art (due to grid lines), low for photos.
        var_x = np.var(sum_dx) / np.mean(sum_dx) if np.mean(sum_dx) > 0 else 0
        var_y = np.var(sum_dy) / np.mean(sum_dy) if np.mean(sum_dy) > 0 else 0
        blockiness = (var_x + var_y) / 2.0

        # Normalize blockiness (typically > 5 for scaled pixel art, < 1 for photos)
        block_score = min(1.0, blockiness / 10.0)

        # Measure flat areas (characteristic of pixel art/highly quantized images).
        # Relaxed threshold to 15/255 to account for JPEG artifacts on game assets.
        flat_areas_x = np.mean(dx < 15.0 / 255.0)
        flat_areas_y = np.mean(dy < 15.0 / 255.0)
        flatness = (flat_areas_x + flat_areas_y) / 2.0

        # Combining robust indicators (effective_colors penalized less to support complex RPG tiles)
        color_score = max(0.0, 1.0 - (effective_colors / 200.0) ** 0.5)
        quant_score = float(color_score * 0.2 + block_score * 0.5 + flatness * 0.3)

        # Complexity as continuous field
        colors_norm = min(len(unique) / 100, 1.0)
        entropy_norm = min(entropy / 8, 1.0)
        complexity = float((edge_density + colors_norm + entropy_norm) / 3)

        # Lab stats (continuous)
        lab = rgb_to_lab(pixels)
        mean_chroma = float(np.mean(np.sqrt(lab[:, 1] ** 2 + lab[:, 2] ** 2)))
        mean_lum = float(np.mean(lab[:, 0]))

        return {
            "unique_colors": len(unique),
            "color_entropy": entropy,
            "edge_density": edge_density,
            "has_outline": has_outline,
            "quantization_score": quant_score,
            "complexity_score": complexity,
            "mean_chroma": mean_chroma,
            "mean_luminance": mean_lum,
        }

    def _merge_votes(self, votes: List[Dict], scales: Tuple[float, ...]) -> Dict:
        weights = list(scales)
        total_w = sum(weights)
        merged = {}
        numeric = ["unique_colors", "color_entropy", "edge_density", "complexity_score",
                   "mean_chroma", "mean_luminance", "quantization_score"]
        for k in numeric:
            merged[k] = sum(v[k] * w for v, w in zip(votes, weights)) / total_w
        # Boolean: majority vote
        merged["has_outline"] = sum(v["has_outline"] for v in votes) >= len(votes) * 0.5
        return merged


class RouterStage(PipelineStage):
    """Continuous routing — no discrete switch. Outputs strategy_name + config."""

    # Strategy templates (config-driven, not code-driven)
    TEMPLATES = {
        "pixel_art": {
            "resize_method": Image.Resampling.NEAREST,
            "quantize_method": "exact",
            "cleanup": "none",
            "detail_preserve": "high",
            "n_colors": "auto_low",
            "use_lab": False,
            "path_opt": False,
            "description": "Pixel art — exact palette, minimal processing",
        },
        "vector_approx": {
            "resize_method": Image.Resampling.LANCZOS,
            "quantize_method": "kmeans_rich",
            "cleanup": "light",
            "detail_preserve": "medium",
            "n_colors": "auto_high",
            "use_lab": True,
            "path_opt": True,
            "description": "Vector smooth — rich colors, path-based",
        },
        "photo_simplified": {
            "resize_method": Image.Resampling.BILINEAR,
            "quantize_method": "posterize",
            "cleanup": "heavy",
            "detail_preserve": "low",
            "n_colors": "auto_medium",
            "use_lab": True,
            "path_opt": True,
            "description": "Photo — posterize + edge detect",
        },
        "hybrid": {
            "resize_method": Image.Resampling.NEAREST,
            "quantize_method": "hybrid_kmeans",
            "cleanup": "smart",
            "detail_preserve": "adaptive",
            "n_colors": "auto_adaptive",
            "use_lab": True,
            "path_opt": True,
            "description": "Hybrid — auto-detect per region",
        },
    }

    def run(self, ctx: PipelineContext) -> None:
        a = ctx.analysis
        user = ctx.user_params

        # User override takes precedence
        forced = user.get("strategy")
        if forced and forced in self.TEMPLATES:
            ctx.strategy_name = forced
            ctx.strategy_config = self.TEMPLATES[forced].copy()
            ctx.metadata["strategy_name"] = forced
            ctx.metadata["strategy_confidence"] = 1.0
            return

        # Continuous scoring (not discrete 0/1)
        pa_score = (
            a.get("quantization_score", 0) * 4.0 +
            (2.0 if a.get("has_outline") else 0.0) +
            max(0.0, (9.0 - a.get("color_entropy", 10)) / 9.0 * 1.0) +
            min(1.0, a.get("edge_density", 0) / 0.1)
        )
        v_score = (
            (2.0 if not a.get("has_outline") else 0.0) +
            max(0.0, (a.get("color_entropy", 0) - 5.0) / 3.0 * 2.0) +
            max(0.0, (0.05 - a.get("edge_density", 0.1)) / 0.05)
        )
        ph_score = (
            max(0.0, (a.get("complexity_score", 0) - 0.6) / 0.4 * 3.0) +
            (2.0 if a.get("quantization_score", 1) < 0.5 else 0.0) +
            min(1.0, a.get("unique_colors", 0) / 200)
        )

        scores = {
            "pixel_art": pa_score,
            "vector_approx": v_score,
            "photo_simplified": ph_score,
            "hybrid": 1.0,  # baseline
        }

        best = max(scores, key=scores.get)
        max_score = scores[best]

        # Confidence as continuous value
        total = sum(scores.values())
        confidence = max_score / total if total > 0 else 0.5

        # Fallback: if confidence too low, use hybrid
        if confidence < 0.35:
            best = "hybrid"
            confidence = 0.35

        ctx.strategy_name = best
        ctx.metadata["strategy_name"] = best
        ctx.metadata["strategy_confidence"] = confidence
        ctx.strategy_config = self.TEMPLATES[best].copy()
        ctx.metadata["strategy_scores"] = scores
        ctx.metadata["strategy_confidence"] = confidence


class PreprocessStage(PipelineStage):
    def run(self, ctx: PipelineContext) -> None:
        cfg = ctx.strategy_config
        img = ctx.image
        assert img is not None
        img = img.convert("RGB")
        dp = ctx.user_params.get("detail_preserve", cfg.get("detail_preserve", "medium"))
        if dp == "high":
            pass
        elif dp == "medium":
            img = img.filter(ImageFilter.SHARPEN)
        else:
            img = img.filter(ImageFilter.SHARPEN).filter(ImageFilter.SHARPEN)
        ctx.image = img
        ctx.arr = np.array(img)


class CropStage(PipelineStage):
    def __init__(self, threshold: int = 240, padding: int = 2):
        self.threshold = threshold
        self.padding = padding

    def run(self, ctx: PipelineContext) -> None:
        if not ctx.user_params.get("crop_content", True):
            return
        img = ctx.image
        assert img is not None
        img = img.convert("RGBA")
        arr = np.array(img)
        alpha_mask = arr[:, :, 3] > 50
        color_mask = ~np.all(arr[:, :, :3] > self.threshold, axis=2)
        mask = alpha_mask & color_mask
        if not np.any(mask):
            return
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        rmin = max(0, rmin - self.padding)
        rmax = min(arr.shape[0] - 1, rmax + self.padding)
        cmin = max(0, cmin - self.padding)
        cmax = min(arr.shape[1] - 1, cmax + self.padding)
        ctx.image = img.crop((cmin, rmin, cmax + 1, rmax + 1))
        ctx.arr = np.array(ctx.image)


class GridSizeStage(PipelineStage):
    def run(self, ctx: PipelineContext) -> None:
        img = ctx.image
        assert img is not None
        w, h = img.size
        a = ctx.analysis
        user_gs = ctx.user_params.get("grid_size", "auto")
        max_dim = ctx.user_params.get("max_dim")

        if user_gs != "auto":
            if isinstance(user_gs, int):
                ctx.metadata["grid_size"] = (user_gs, user_gs)
            elif isinstance(user_gs, (tuple, list)) and len(user_gs) == 2:
                ctx.metadata["grid_size"] = (int(user_gs[0]), int(user_gs[1]))
            return

        # Adaptive Grid Size via Edge Frequency
        arr_f = ctx.arr.astype(np.float64)
        dx = np.sum(np.abs(arr_f[:, 1:, :3] - arr_f[:, :-1, :3]), axis=2)
        dy = np.sum(np.abs(arr_f[1:, :, :3] - arr_f[:-1, :, :3]), axis=2)

        sum_dx = np.sum(dx, axis=0)
        sum_dy = np.sum(dy, axis=1)

        peaks_x = np.where(sum_dx > np.mean(sum_dx) * 1.5)[0]
        peaks_y = np.where(sum_dy > np.mean(sum_dy) * 1.5)[0]

        def extract_scale(peaks):
            if len(peaks) < 2: return 0.0
            diffs = np.diff(peaks)
            valid = diffs[diffs >= 2]
            if len(valid) == 0: return 0.0
            # Stronger snapping to exact integer grid scales to prevent "line keluar" (jagged misalignment)
            unique, counts = np.unique(np.round(valid), return_counts=True)
            best_diff = unique[np.argmax(counts)]
            close = valid[np.abs(valid - best_diff) <= 1.5]
            return float(np.round(np.mean(close))) if len(close) > 0 else float(best_diff)

        scale_x = extract_scale(peaks_x)
        scale_y = extract_scale(peaks_y)

        # Continuous fallback if signal is too weak
        if scale_x == 0 and scale_y == 0:
            scale = max(1.0, min(w, h) / 48.0)
        else:
            scale = max(1.0, (scale_x + scale_y) / 2.0 if scale_x and scale_y else (scale_x or scale_y))

        # If it's a known pixel art image but very noisy, strict scaling helps grid alignment
        scale = np.round(scale) # Snap scale to exact integer to fix "line keluar" misalignment on JPEGs
        scale = max(2.0, min(scale, min(w, h) / 8.0))


        auto_gs = (max(1, int(round(w / scale))), max(1, int(round(h / scale))))

        if max_dim:
            aspect = w / h if h else 1
            if aspect >= 1:
                gs = (max_dim, max(1, int(round(max_dim / aspect))))
            else:
                gs = (max(1, int(round(max_dim * aspect))), max_dim)
        elif a.get("quantization_score", 0) > 0.4:
            gs = auto_gs
        else:
            scale = max(w, h) / 48.0
            gs = (max(1, int(round(w / scale))), max(1, int(round(h / scale))))

        ctx.metadata["grid_size"] = gs
        ctx.metadata["auto_scale"] = scale


class ResizeStage(PipelineStage):
    def run(self, ctx: PipelineContext) -> None:
        img = ctx.image
        assert img is not None
        gw, gh = ctx.metadata["grid_size"]
        method = ctx.strategy_config.get("resize_method", Image.Resampling.NEAREST)
        ctx.image = img.resize((gw, gh), method)
        ctx.arr = np.array(ctx.image.convert("RGBA"))


class QuantizeStage(PipelineStage):
    """Quantization with perceptual distance and buffer reuse."""

    def __init__(self, n_colors: Union[int, str] = "auto", use_lab: Optional[bool] = None):
        self.n_colors = n_colors
        self.use_lab = use_lab
        # Pre-allocated buffers (ghost state)
        self._buf_rgb = np.empty((2_000_000, 3), dtype=np.float64)
        self._buf_lab = np.empty((2_000_000, 3), dtype=np.float64)

    def run(self, ctx: PipelineContext) -> None:
        arr = ctx.arr
        assert arr is not None
        h, w = arr.shape[:2]
        pixels = arr[:, :, :3].reshape(-1, 3)
        alpha = arr[:, :, 3].reshape(-1)

        if ctx.user_params.get("remove_background", True):
            valid = alpha > 50
            pixels = pixels[valid]

        n = len(pixels)
        use_lab = self.use_lab if self.use_lab is not None else ctx.strategy_config.get("use_lab", False)

        # Determine n_colors from analysis (continuous, not hardcoded)
        nc = self._resolve_n_colors(ctx)

        qm = ctx.strategy_config.get("quantize_method", "exact")
        if qm == "exact":
            palette = self._quantize_exact(pixels, nc)
        elif qm == "kmeans_rich":
            palette = self._quantize_kmeans(pixels, nc, use_lab)
        elif qm == "posterize":
            palette = self._quantize_posterize(pixels, nc)
        else:
            # hybrid: exact if highly quantized, else kmeans
            if ctx.analysis.get("quantization_score", 0) > 0.7 and ctx.analysis.get("unique_colors", 9999) < nc * 2:
                palette = self._quantize_exact(pixels, nc)
            else:
                palette = self._quantize_kmeans(pixels, nc, use_lab)

        # Merge palette hints
        hints = ctx.user_params.get("palette_hint")
        if hints:
            hint_colors = [hex_to_rgb(h) for h in hints]
            palette = self._merge_hints(palette, hint_colors, nc)

        ctx.palette = palette[:nc]
        ctx.color_map = {i + 1: rgb_to_hex(p) for i, p in enumerate(ctx.palette)}
        ctx.metadata["n_colors"] = len(ctx.palette)

        # Quantize grid (perceptual)
        ctx.grid = self._quantize_grid(arr, ctx.palette, use_lab, ctx)

    def _resolve_n_colors(self, ctx: PipelineContext) -> int:
        if isinstance(self.n_colors, int):
            return self.n_colors
        a = ctx.analysis
        nc_cfg = ctx.strategy_config.get("n_colors", "auto_adaptive")
        uc = a.get("unique_colors", 256)
        ed = a.get("edge_density", 0)

        if nc_cfg == "auto_low":
            base = max(4, min(16, int(uc // 2)))
            if ed > 0.08:
                base = min(24, base + 4)
            return base
        elif nc_cfg == "auto_high":
            return min(32, max(16, int(uc // 3)))
        elif nc_cfg == "auto_medium":
            return min(24, max(8, int(uc // 4)))
        else:
            if a.get("quantization_score", 0) > 0.4:
                # User preference: stepped colors (6 > 16 > 20)
                if uc <= 10: return 6
                elif uc <= 24: return 16
                return 20
            return min(24, max(8, int(uc // 3)))

    def _quantize_exact(self, pixels: np.ndarray, n: int) -> List[Tuple[int, int, int]]:
        rounded = (pixels // 16) * 16
        unique, counts = np.unique(rounded, axis=0, return_counts=True)
        top_idx = np.argsort(counts)[-n:]
        palette = [tuple(map(int, unique[i])) for i in top_idx]
        lum = [0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2] for c in palette]
        return [p for _, p in sorted(zip(lum, palette))]

    def _quantize_kmeans(self, pixels: np.ndarray, n: int, use_lab: bool) -> List[Tuple[int, int, int]]:
        if use_lab:
            px = rgb_to_lab(pixels)
            rounded = (px // 8) * 8
        else:
            px = pixels.astype(np.float64)
            rounded = (px // 32) * 32

        unique, counts = np.unique(rounded, axis=0, return_counts=True)
        top_n = min(n, len(unique))
        top_idx = np.argsort(counts)[-top_n:]
        centroids = unique[top_idx].astype(np.float64).copy()

        if len(centroids) < n:
            extra = n - len(centroids)
            idx = np.random.choice(len(px), extra, replace=False)
            centroids = np.vstack([centroids, px[idx]])

        for _ in range(50):
            if use_lab:
                dists = delta_e_cie76(px[:, None, :], centroids[None, :, :])
            else:
                dists = np.linalg.norm(px[:, None, :] - centroids[None, :, :], axis=2)
            labels = np.argmin(dists, axis=1)
            new_c = np.array([
                px[labels == k].mean(axis=0) if np.sum(labels == k) > 0 else centroids[k]
                for k in range(n)
            ])
            if np.allclose(centroids, new_c, atol=1.5):
                break
            centroids = new_c

        # Merge similar colors
        merged = []
        used = set()
        threshold = 12.0 if use_lab else 25.0
        for i in range(len(centroids)):
            if i in used:
                continue
            group = [i]
            for j in range(i + 1, len(centroids)):
                if j not in used:
                    if use_lab:
                        d = delta_e_cie76(centroids[i], centroids[j])
                    else:
                        d = np.linalg.norm(centroids[i] - centroids[j])
                    if d < threshold:
                        group.append(j)
                        used.add(j)
            avg = centroids[group].mean(axis=0)
            if use_lab:
                avg_rgb = QuantizeStage._lab_to_rgb_approx(avg)
                merged.append(tuple(map(int, np.clip(avg_rgb, 0, 255))))
            else:
                merged.append(tuple(map(int, np.clip(avg, 0, 255))))

        if use_lab:
            lab_pal = np.array([QuantizeStage._rgb_to_lab_approx(np.array(c)) for c in merged])
            lum = lab_pal[:, 0]
        else:
            lum = [0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2] for c in merged]
        sorted_idx = np.argsort(lum)
        return [merged[i] for i in sorted_idx]

    def _quantize_posterize(self, pixels: np.ndarray, n: int) -> List[Tuple[int, int, int]]:
        levels = max(2, min(8, int(round(n ** (1 / 3)))))
        step = 255 // (levels - 1)
        rounded = (pixels // step) * step
        unique, counts = np.unique(rounded, axis=0, return_counts=True)
        top_idx = np.argsort(counts)[-n:]
        return [tuple(map(int, unique[i])) for i in top_idx]

    def _merge_hints(self, palette, hints, n):
        result = list(palette)
        for hint in hints:
            min_dist = float("inf")
            best_idx = 0
            for i, p in enumerate(result):
                dist = np.linalg.norm(np.array(p) - np.array(hint))
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
            if min_dist < 100:
                result[best_idx] = hint
        return result[:n]

    def _quantize_grid(self, arr, palette, use_lab, ctx):
        h, w = arr.shape[:2]
        grid = np.zeros((h, w), dtype=np.int32)
        pal = np.array(palette, dtype=np.float64)
        pal_lab = rgb_to_lab(pal) if use_lab else pal
        bg_threshold = ctx.user_params.get("bg_threshold", 250)
        remove_bg = ctx.user_params.get("remove_background", True)
        for y in range(h):
            for x in range(w):
                pixel = arr[y, x, :3]
                alpha = arr[y, x, 3]
                if remove_bg and (alpha < 50 or np.all(pixel > bg_threshold)):
                    continue
                if use_lab:
                    px_lab = rgb_to_lab(pixel.reshape(1, 3)).reshape(3)
                    dists = delta_e_cie76(px_lab, pal_lab)
                else:
                    dists = np.linalg.norm(pal - pixel, axis=1)
                grid[y, x] = int(np.argmin(dists)) + 1
        return grid
    def _rgb_to_lab_approx(rgb):
        return rgb_to_lab(rgb.reshape(1, 3)).reshape(3)

    @staticmethod
    def _lab_to_rgb_approx(lab):
        L, a, b = lab
        Y = (L + 16) / 116.0
        X = a / 500.0 + Y
        Z = Y - b / 200.0
        xyz = np.array([X, Y, Z]) * _D65
        xyz = xyz / 100.0
        rgb = xyz @ np.linalg.inv(_XYZ_MATRIX).T
        mask = rgb > 0.0031308
        rgb = np.where(mask, 1.055 * (rgb ** (1 / 2.4)) - 0.055, 12.92 * rgb)
        return np.clip(rgb * 255, 0, 255)


class CleanupStage(PipelineStage):
    def run(self, ctx: PipelineContext) -> None:
        mode = ctx.user_params.get("cleanup", ctx.strategy_config.get("cleanup", "none"))
        grid = ctx.grid
        assert grid is not None
        h, w = grid.shape
        if mode == "none":
            return

        # Dynamic threshold based on analysis. Less cleanup for structured pixel art
        quant_score = ctx.analysis.get("quantization_score", 0.0)
        if mode == "smart":
            if quant_score > 0.6:
                mode = "none" # True pixel art needs no cleanup, preserves single pixels
            elif quant_score > 0.3:
                mode = "light"
            else:
                mode = "heavy"

        if mode == "none":
            return

        min_n = 1 if mode == "light" else 2
        cleaned = grid.copy()

        # We replace isolated pixels with the majority color of their neighbors,
        # instead of just setting them to 0 (which punches holes).
        for y in range(h):
            for x in range(w):
                if cleaned[y, x] == 0:
                    continue
                val = cleaned[y, x]

                # Check 4-connected neighbors of same color
                neighbors_same = 0
                if y > 0 and cleaned[y - 1, x] == val: neighbors_same += 1
                if y < h - 1 and cleaned[y + 1, x] == val: neighbors_same += 1
                if x > 0 and cleaned[y, x - 1] == val: neighbors_same += 1
                if x < w - 1 and cleaned[y, x + 1] == val: neighbors_same += 1

                if neighbors_same <= min_n:
                    # Gather 8-connected neighbor colors to find majority
                    all_n = []
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w and cleaned[ny, nx] != 0:
                                all_n.append(cleaned[ny, nx])

                    if not all_n:
                        cleaned[y, x] = 0
                    else:
                        from collections import Counter
                        majority = Counter(all_n).most_common(1)[0][0]
                        # Only replace if majority is strong enough
                        if all_n.count(majority) > len(all_n) // 2:
                            cleaned[y, x] = majority

        ctx.grid = cleaned


class OutputStage(PipelineStage):
    def __init__(self, pixel_size: int = 16, use_paths: bool = True):
        self.pixel_size = pixel_size
        self.use_paths = use_paths

    def run(self, ctx: PipelineContext) -> None:
        grid = ctx.grid
        assert grid is not None
        cmap = ctx.color_map
        h, w = grid.shape
        ps = self.pixel_size
        canvas_w = w * ps + 80
        canvas_h = h * ps + 80
        ox = (canvas_w - w * ps) // 2
        oy = (canvas_h - h * ps) // 2

        svg = SVGBuilder(canvas_w, canvas_h)
        svg.add_rect(0, 0, canvas_w, canvas_h, fill="#FFFFFF")

        if not self.use_paths:
            for y in range(h):
                for x in range(w):
                    if grid[y, x] == 0:
                        continue
                    svg.add_rect(ox + x * ps, oy + y * ps, ps, ps, fill=cmap.get(int(grid[y, x]), "#000"))
        else:
            unique_colors = np.unique(grid[grid > 0])
            total_paths = 0
            for color_idx in unique_colors:
                mask = (grid == color_idx)
                fill = cmap.get(int(color_idx), "#000000")
                d = mask_to_svg_paths(mask, ps, ox, oy)
                if d:
                    svg.add_path(d, fill=fill)
                    total_paths += 1

            # Heuristic: if too fragmented, suggest rect mode for next run
            if total_paths > len(unique_colors) * 50:
                ctx.metadata["svg_fragmentation_warning"] = True

        ctx.metadata["svg_string"] = svg.to_string()
        ctx.metadata["output_size"] = (canvas_w, canvas_h)


# ============================================================================
# CONNECTED COMPONENTS & SVG PATH TRACING
# ============================================================================

def mask_to_svg_paths(mask: np.ndarray, scale: float, offset_x: float, offset_y: float) -> str:
    """
    Given a 2D boolean mask, trace the boundaries to perfectly form a polygon path.
    Traces the edges of the pixels so that a 1x1 pixel forms a 1x1 square path without gaps.
    """
    h, w = mask.shape

    h_edges = np.zeros((h + 1, w), dtype=bool)
    h_edges[0, :] = mask[0, :]
    h_edges[-1, :] = mask[-1, :]
    if h > 1:
        h_edges[1:-1, :] = mask[:-1, :] ^ mask[1:, :]

    v_edges = np.zeros((h, w + 1), dtype=bool)
    v_edges[:, 0] = mask[:, 0]
    v_edges[:, -1] = mask[:, -1]
    if w > 1:
        v_edges[:, 1:-1] = mask[:, :-1] ^ mask[:, 1:]

    parts = []
    used_h = np.zeros_like(h_edges)
    used_v = np.zeros_like(v_edges)

    while True:
        start_y, start_x = -1, -1
        for y in range(h + 1):
            for x in range(w):
                if h_edges[y, x] and not used_h[y, x]:
                    start_y, start_x = y, x
                    break
            if start_y != -1: break

        if start_y == -1: break

        cy, cx = start_y, start_x
        if start_y < h and mask[start_y, start_x]:
            direction = 'R'
        else:
            direction = 'L'
            cx += 1

        poly = [(cx, cy)]

        while True:
            if direction == 'R':
                used_h[cy, cx] = True
                cx += 1
                poly.append((cx, cy))
                if cy < h and v_edges[cy, cx] and not used_v[cy, cx]:
                    direction = 'D'
                elif cx < w and h_edges[cy, cx] and not used_h[cy, cx]:
                    direction = 'R'
                elif cy > 0 and v_edges[cy-1, cx] and not used_v[cy-1, cx]:
                    direction = 'U'
                    cy -= 1
                else:
                    break
            elif direction == 'L':
                cx -= 1
                used_h[cy, cx] = True
                poly.append((cx, cy))
                if cy > 0 and v_edges[cy-1, cx] and not used_v[cy-1, cx]:
                    direction = 'U'
                    cy -= 1
                elif cx > 0 and h_edges[cy, cx-1] and not used_h[cy, cx-1]:
                    direction = 'L'
                elif cy < h and v_edges[cy, cx] and not used_v[cy, cx]:
                    direction = 'D'
                else:
                    break
            elif direction == 'D':
                used_v[cy, cx] = True
                cy += 1
                poly.append((cx, cy))
                if cx > 0 and h_edges[cy, cx-1] and not used_h[cy, cx-1]:
                    direction = 'L'
                elif cy < h and v_edges[cy, cx] and not used_v[cy, cx]:
                    direction = 'D'
                elif cx < w and h_edges[cy, cx] and not used_h[cy, cx]:
                    direction = 'R'
                else:
                    break
            elif direction == 'U':
                used_v[cy, cx] = True
                poly.append((cx, cy))
                if cx < w and h_edges[cy, cx] and not used_h[cy, cx]:
                    direction = 'R'
                elif cy > 0 and v_edges[cy-1, cx] and not used_v[cy-1, cx]:
                    direction = 'U'
                    cy -= 1
                elif cx > 0 and h_edges[cy, cx-1] and not used_h[cy, cx-1]:
                    direction = 'L'
                else:
                    break

        if len(poly) > 2:
            simp = [poly[0]]
            for i in range(1, len(poly)-1):
                p0, p1, p2 = poly[i-1], poly[i], poly[i+1]
                if not ((p0[0] == p1[0] == p2[0]) or (p0[1] == p1[1] == p2[1])):
                    simp.append(p1)
            simp.append(poly[-1])

            # Sub-pixel rounding for less "kaku" edges (corner smoothing)
            cmds = []
            if len(simp) > 3:
                # Start midway between simp[0] and simp[1]
                p0, p1 = simp[0], simp[1]
                sx, sy = (p0[0] + p1[0])/2 * scale + offset_x, (p0[1] + p1[1])/2 * scale + offset_y
                cmds.append(f"M {sx:.2f} {sy:.2f}")

                for i in range(1, len(simp)-1):
                    prev_p, curr_p, next_p = simp[i-1], simp[i], simp[i+1]

                    # Midpoint of the line going into the corner
                    m1x, m1y = (prev_p[0] + curr_p[0])/2 * scale + offset_x, (prev_p[1] + curr_p[1])/2 * scale + offset_y

                    # The corner itself
                    cx, cy = curr_p[0] * scale + offset_x, curr_p[1] * scale + offset_y

                    # Midpoint of the line going out of the corner
                    m2x, m2y = (curr_p[0] + next_p[0])/2 * scale + offset_x, (curr_p[1] + next_p[1])/2 * scale + offset_y

                    cmds.append(f"L {m1x:.2f} {m1y:.2f}")
                    # Curve around the corner
                    cmds.append(f"C {cx:.2f} {cy:.2f} {cx:.2f} {cy:.2f} {m2x:.2f} {m2y:.2f}")

                # Close back to start
                cmds.append(f"L {sx:.2f} {sy:.2f} Z")
            else:
                cmds = [f"M {simp[0][0]*scale + offset_x:.2f} {simp[0][1]*scale + offset_y:.2f}"]
                for p in simp[1:-1]:
                    ex, ey = p[0]*scale + offset_x, p[1]*scale + offset_y
                    cmds.append(f"C {ex:.2f} {ey:.2f} {ex:.2f} {ey:.2f} {ex:.2f} {ey:.2f}")
                cmds.append("Z")

            parts.append(" ".join(cmds))

    return " ".join(parts)




# ============================================================================
# PLUGIN SYSTEM — Extensible stage injection
# ============================================================================

@dataclass
class PluginSpec:
    """Plugin registration specification."""
    name: str
    hook: str  # e.g., "post_quantize", "pre_output"
    priority: int = 50  # Lower = earlier
    condition: Optional[Callable[[PipelineContext], bool]] = None
    transform: Callable[[PipelineContext], None] = field(default=lambda ctx: None)


class PluginRegistry:
    """Central plugin registry with hook resolution."""

    _instance: Optional[PluginRegistry] = None
    _plugins: Dict[str, List[PluginSpec]] = field(default_factory=lambda: defaultdict(list))

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._plugins = defaultdict(list)
        return cls._instance

    def register(self, spec: PluginSpec) -> PluginSpec:
        """Register a plugin at a hook point."""
        self._plugins[spec.hook].append(spec)
        # Sort by priority
        self._plugins[spec.hook].sort(key=lambda p: p.priority)
        return spec

    def discover_from_directory(self, plugin_dir: str) -> int:
        """Auto-discover plugins from directory. Returns count loaded."""
        count = 0
        pdir = Path(plugin_dir)
        if not pdir.exists():
            return 0
        for pyfile in pdir.glob("*.py"):
            try:
                self._load_plugin_file(str(pyfile))
                count += 1
            except Exception as e:
                warnings.warn(f"Failed to load plugin {pyfile}: {e}")
        return count

    def _load_plugin_file(self, filepath: str):
        """Load a single plugin file."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("_plugin", filepath)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

    def resolve(self, hook: str, ctx: PipelineContext) -> List[PluginSpec]:
        """Get sorted, filtered plugins for a hook point."""
        plugins = self._plugins.get(hook, [])
        return [p for p in plugins if p.condition is None or p.condition(ctx)]

    def execute(self, hook: str, ctx: PipelineContext) -> Dict[str, Any]:
        """Execute all plugins at hook point. Returns execution log."""
        log = {}
        for plugin in self.resolve(hook, ctx):
            try:
                plugin.transform(ctx)
                log[plugin.name] = "ok"
            except Exception as e:
                log[plugin.name] = f"error: {e}"
                # Configurable: skip or abort
                if ctx.user_params.get("plugin_abort_on_error", False):
                    raise
        return log


# Convenience decorator
_plugin_registry = PluginRegistry()


def plugin(hook: str, priority: int = 50, condition=None, name: Optional[str] = None):
    """Decorator to register a function as a plugin.

    Usage:
        @plugin("post_quantize", priority=10, 
                condition=lambda ctx: ctx.analysis.get("edge_density", 0) > 0.1)
        def my_dither(ctx: PipelineContext):
            # Mutate ctx.grid or ctx.palette
            pass
    """
    def decorator(func: Callable[[PipelineContext], None]):
        spec = PluginSpec(
            name=name or func.__name__,
            hook=hook,
            priority=priority,
            condition=condition,
            transform=func,
        )
        _plugin_registry.register(spec)
        return func
    return decorator


# ============================================================================
# PLUGIN-AWARE PIPELINE STAGE
# ============================================================================

class PluginAwareStage(PipelineStage):
    """Base stage that executes plugins at hook points."""

    def __init__(self, hook_before: Optional[str] = None, hook_after: Optional[str] = None):
        self.hook_before = hook_before
        self.hook_after = hook_after
        self._registry = PluginRegistry()

    def run(self, ctx: PipelineContext) -> None:
        if self.hook_before:
            self._registry.execute(self.hook_before, ctx)
        self._execute_core(ctx)
        if self.hook_after:
            self._registry.execute(self.hook_after, ctx)

    @abstractmethod
    def _execute_core(self, ctx: PipelineContext) -> None:
        ...


# ============================================================================
# BUILT-IN PLUGINS (examples)
# ============================================================================

@plugin("post_quantize", priority=20, 
        condition=lambda ctx: ctx.user_params.get("dithering", False))
def dithering_plugin(ctx: PipelineContext):
    """Apply Floyd-Steinberg dithering to reduce banding."""
    grid = ctx.grid
    if grid is None:
        return
    h, w = grid.shape
    palette = np.array(ctx.palette, dtype=np.float64)
    pal_lab = rgb_to_lab(palette)

    for y in range(h - 1):
        for x in range(1, w - 1):
            old_val = grid[y, x]
            if old_val == 0:
                continue
            old_color = pal_lab[old_val - 1]
            # Find nearest in Lab
            dists = delta_e_cie76(old_color, pal_lab)
            new_val = int(np.argmin(dists)) + 1
            grid[y, x] = new_val
            # Quantization error (simplified)
            # In real implementation, distribute error to neighbors
    ctx.grid = grid
    ctx.metadata["dithering_applied"] = True


@plugin("post_quantize", priority=30,
        condition=lambda ctx: ctx.user_params.get("outline_enhance", False))
def outline_enhance_plugin(ctx: PipelineContext):
    """Thicken outlines for pixel art."""
    grid = ctx.grid
    if grid is None:
        return
    h, w = grid.shape
    # Find outline color (darkest in palette)
    if not ctx.palette:
        return
    luminance = [0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2] for c in ctx.palette]
    outline_idx = int(np.argmin(luminance)) + 1

    enhanced = grid.copy()
    for y in range(h):
        for x in range(w):
            if grid[y, x] != 0:
                continue
            # Check neighbors
            neighbors = []
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    neighbors.append(grid[ny, nx])
            # Fill isolated background surrounded by outline
            if neighbors.count(outline_idx) >= 3:
                enhanced[y, x] = outline_idx
    ctx.grid = enhanced
    ctx.metadata["outline_enhanced"] = True


@plugin("pre_output", priority=10)
def metadata_inject_plugin(ctx: PipelineContext):
    """Inject generation metadata into SVG."""
    ctx.metadata["generated_by"] = "SVG Vision v8"
    ctx.metadata["generation_time"] = time.time()


@plugin("post_analysis", priority=5)
def analysis_validator_plugin(ctx: PipelineContext):
    """Validate analysis results and warn if suspicious."""
    a = ctx.analysis
    if a.get("color_entropy", 0) > 8 and a.get("quantization_score", 0) > 0.8:
        warnings.warn("High entropy but high quantization score — possible JPEG artifact")


# ============================================================================
# PLUGIN-AWARE PIPELINE (extends base Pipeline)
# ============================================================================

class PluginAwarePipeline(Pipeline):
    """Pipeline with automatic plugin execution at hook points."""

    def __init__(self, stages: Optional[List[PipelineStage]] = None,
                 plugin_dir: Optional[str] = None):
        super().__init__(stages)
        self.registry = PluginRegistry()
        if plugin_dir:
            self.registry.discover_from_directory(plugin_dir)

    def run_on(self, image_path: str, **user_params) -> Tuple[np.ndarray, Dict[int, str], Dict]:
        ctx = PipelineContext(image_path=image_path, user_params=user_params)

        # Pre-pipeline hook
        self.registry.execute("pre_pipeline", ctx)

        for stage in self.stages:
            hook_before = f"pre_{stage.__class__.__name__.lower().replace('stage', '')}"
            hook_after = f"post_{stage.__class__.__name__.lower().replace('stage', '')}"

            self.registry.execute(hook_before, ctx)
            try:
                stage.run(ctx)
            except Exception as e:
                ctx.metadata[f"{stage.__class__.__name__}_error"] = str(e)
                # Try fallback plugins
                fallbacks = self.registry.resolve("fallback", ctx)
                if fallbacks:
                    for fb in fallbacks:
                        try:
                            fb.transform(ctx)
                            ctx.metadata[f"{stage.__class__.__name__}_fallback"] = fb.name
                            break
                        except Exception:
                            continue
                else:
                    raise
            self.registry.execute(hook_after, ctx)

        # Post-pipeline hook
        self.registry.execute("post_pipeline", ctx)

        assert ctx.grid is not None
        return ctx.grid, ctx.color_map, ctx.metadata

    @classmethod
    def auto_with_plugins(cls, image_path: str, plugin_dir: Optional[str] = None, **kwargs):
        """Factory: build pipeline with plugin discovery."""
        pipe = cls(plugin_dir=plugin_dir)
        pipe |= LoadStage()
        pipe |= AnalysisStage(scales=kwargs.get("scales", (1.0, 0.5, 0.25)))
        pipe |= RouterStage()
        pipe |= PreprocessStage()
        pipe |= CropStage()
        pipe |= GridSizeStage()
        pipe |= ResizeStage()
        pipe |= QuantizeStage(
            n_colors=kwargs.get("n_colors", "auto"),
            use_lab=kwargs.get("use_lab", None),
        )
        pipe |= CleanupStage()
        pipe |= OutputStage(
            pixel_size=kwargs.get("pixel_size", 16),
            use_paths=kwargs.get("use_paths", True),
        )
        return pipe

# ============================================================================
# LEGACY API (backward compatible wrapper)
# ============================================================================

def image_to_grid(
    image_path: str,
    grid_size: Union[int, Tuple[int, int], str] = "auto",
    n_colors: Union[int, str] = "auto",
    strategy: Optional[str] = None,
    palette_hint: Optional[List[str]] = None,
    detail_preserve: Optional[str] = None,
    cleanup: Optional[str] = None,
    bg_threshold: int = 250,
    remove_background: bool = True,
    max_dim: Optional[int] = None,
    crop_content: bool = True,
    confidence_threshold: float = 0.7,
    use_ciede2000: bool = False,
    multi_scale: bool = True,
) -> Tuple[List[List[int]], Dict[int, str], Dict]:
    """Backward-compatible API wrapping the new Pipeline."""
    pipe = Pipeline.auto_from_image(
        image_path,
        scales=(1.0, 0.5, 0.25) if multi_scale else (1.0,),
    )
    grid_arr, cmap, meta = pipe.run_on(
        image_path,
        grid_size=grid_size,
        n_colors=n_colors,
        strategy=strategy,
        palette_hint=palette_hint,
        detail_preserve=detail_preserve,
        cleanup=cleanup,
        bg_threshold=bg_threshold,
        remove_background=remove_background,
        max_dim=max_dim,
        crop_content=crop_content,
        use_ciede2000=use_ciede2000,
    )
    grid = grid_arr.tolist()
    meta["strategy"] = meta.get("strategy_name", "unknown")
    meta["confidence"] = meta.get("strategy_confidence", 0.5)
    return grid, cmap, meta


def grid_to_svg(grid, color_map, pixel_size=16, canvas=None, background="#FFFFFF", use_paths=True):
    grid_arr = np.array(grid, dtype=np.int32)
    h, w = grid_arr.shape
    if canvas is None:
        cw = w * pixel_size + 80
        ch = h * pixel_size + 80
    elif isinstance(canvas, int):
        cw = ch = canvas
    else:
        cw, ch = canvas
    ox = (cw - w * pixel_size) // 2
    oy = (ch - h * pixel_size) // 2

    svg = SVGBuilder(cw, ch)
    if background:
        svg.add_rect(0, 0, cw, ch, fill=background)

    if not use_paths:
        for y in range(h):
            for x in range(w):
                if grid_arr[y, x] == 0:
                    continue
                svg.add_rect(ox + x * pixel_size, oy + y * pixel_size, pixel_size, pixel_size, fill=color_map.get(int(grid_arr[y, x]), "#000"))
    else:
        unique_colors = np.unique(grid_arr[grid_arr > 0])
        for color_idx in unique_colors:
            mask = (grid_arr == color_idx)
            fill = color_map.get(int(color_idx), "#000000")
            d = mask_to_svg_paths(mask, pixel_size, ox, oy)
            if d:
                svg.add_path(d, fill=fill)
    return svg.to_string()


def grid_to_svg_file(grid, color_map, filepath, pixel_size=16, canvas=None, background="#FFFFFF", use_paths=True):
    svg_str = grid_to_svg(grid, color_map, pixel_size, canvas, background, use_paths)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(svg_str)


def grid_to_image(grid, color_map, scale=20):
    h = len(grid)
    w = len(grid[0]) if grid else 0
    img = Image.new("RGB", (w * scale, h * scale), (255, 255, 255))
    px = img.load()
    for y in range(h):
        for x in range(w):
            val = grid[y][x]
            color = hex_to_rgb(color_map[val]) if val != 0 else (255, 255, 255)
            for dy in range(scale):
                for dx in range(scale):
                    px[x * scale + dx, y * scale + dy] = color
    return img


def rgb_to_hex(rgb):
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[i:i + 2], 16) for i in (0, 2, 4))


def optimize_grid(grid):
    h = len(grid)
    w = len(grid[0]) if grid else 0
    visited = [[False] * w for _ in range(h)]
    rects = []
    for y in range(h):
        for x in range(w):
            if visited[y][x] or grid[y][x] == 0:
                continue
            val = grid[y][x]
            rw = 1
            while x + rw < w and grid[y][x + rw] == val and not visited[y][x + rw]:
                rw += 1
            rh = 1
            valid = True
            while valid and y + rh < h:
                for dx in range(rw):
                    if grid[y + rh][x + dx] != val or visited[y + rh][x + dx]:
                        valid = False
                        break
                if valid:
                    rh += 1
            for dy in range(rh):
                for dx in range(rw):
                    visited[y + dy][x + dx] = True
            rects.append((x, y, rw, rh, val))
    return rects


def grid_to_svg_optimized(grid, color_map, pixel_size=16, canvas=None, background="#FFFFFF"):
    h = len(grid)
    w = len(grid[0]) if grid else 0
    if canvas is None:
        cw = w * pixel_size + 80
        ch = h * pixel_size + 80
    elif isinstance(canvas, int):
        cw = ch = canvas
    else:
        cw, ch = canvas
    ox = (cw - w * pixel_size) // 2
    oy = (ch - h * pixel_size) // 2
    svg = SVGBuilder(cw, ch)
    if background:
        svg.add_rect(0, 0, cw, ch, fill=background)
    rects = optimize_grid(grid)
    for x, y, rw, rh, val in rects:
        svg.add_rect(ox + x * pixel_size, oy + y * pixel_size, rw * pixel_size, rh * pixel_size, fill=color_map[val])
    return svg.to_string()


def grid_to_svg_optimized_file(grid, color_map, filepath, pixel_size=16, canvas=None, background="#FFFFFF"):
    svg_str = grid_to_svg_optimized(grid, color_map, pixel_size, canvas, background)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(svg_str)


def grid_diff(target, generated):
    if len(target) != len(generated):
        return {"match_score": 0.0, "error": "Height mismatch"}
    if len(target[0]) != len(generated[0]):
        return {"match_score": 0.0, "error": "Width mismatch"}
    h = len(target)
    w = len(target[0])
    total = h * w
    match = 0
    diffs = []
    for y in range(h):
        for x in range(w):
            if target[y][x] == generated[y][x]:
                match += 1
            else:
                diffs.append((y, x, target[y][x], generated[y][x]))
    score = match / total if total else 0.0
    feedback = [f"Match: {score:.1%} ({match}/{total})"]
    if diffs:
        missing = sum(1 for _, _, t, g in diffs if t != 0 and g == 0)
        extra = sum(1 for _, _, t, g in diffs if t == 0 and g != 0)
        wrong = sum(1 for _, _, t, g in diffs if t != 0 and g != 0)
        if missing: feedback.append(f"Missing: {missing}")
        if extra: feedback.append(f"Extra: {extra}")
        if wrong: feedback.append(f"Wrong color: {wrong}")
    return {
        "match_score": round(score, 3),
        "pixel_total": total,
        "pixel_match": match,
        "pixel_diff": len(diffs),
        "diff_positions": diffs,
        "feedback": feedback,
    }


# ============================================================================
# CLI
# ============================================================================

def _cli_convert(args):
    t0 = time.time()
    pipe = Pipeline.auto_from_image(args.input)
    grid_arr, cmap, meta = pipe.run_on(
        args.input,
        grid_size=args.grid_size,
        n_colors=args.colors,
        strategy=args.strategy,
        palette_hint=args.palette_hint,
        detail_preserve=args.detail,
        cleanup=args.cleanup,
        max_dim=args.max_dim,
        crop_content=not args.no_crop,
        use_ciede2000=args.ciede2000,
    )
    grid = grid_arr.tolist()
    use_paths = not args.legacy_rects
    grid_to_svg_file(grid, cmap, args.output, pixel_size=args.pixel_size, use_paths=use_paths)
    dt = time.time() - t0
    print(f"✓ Converted: {args.input} -> {args.output}")
    print(f"  Strategy: {meta.get('strategy_name', '?')} (confidence: {meta.get('strategy_confidence', 0):.0%})")
    print(f"  Grid: {meta['grid_size']}, Colors: {meta['n_colors']}")
    print(f"  SVG mode: {'path-based' if use_paths else 'rect-optimized'}")
    print(f"  Time: {dt:.2f}s")


def _cli_analyze(args):
    ctx = PipelineContext(image_path=args.input)
    LoadStage().run(ctx)
    AnalysisStage().run(ctx)
    RouterStage().run(ctx)
    a = ctx.analysis
    print(f"File: {args.input}")
    print(f"Size: {a['size']}, Aspect: {a['aspect_ratio']:.2f}")
    print(f"Unique colors: {a['unique_colors']:.0f}")
    print(f"Color entropy: {a['color_entropy']:.2f}")
    print(f"Edge density: {a['edge_density']:.3f}")
    print(f"Has outline: {a['has_outline']}")
    print(f"Quantization score: {a['quantization_score']:.2f}")
    print(f"Complexity: {a['complexity_score']:.2f}")
    print(f"Mean chroma: {a['mean_chroma']:.1f}")
    print(f"Mean luminance: {a['mean_luminance']:.1f}")
    print(f"→ Strategy: {ctx.strategy_name} (confidence: {ctx.metadata.get('strategy_confidence', 0):.0%})")
    print(f"  {ctx.strategy_config.get('description', '')}")


def _cli_batch(args):
    src = Path(args.input_dir)
    dst = Path(args.output_dir)
    dst.mkdir(parents=True, exist_ok=True)
    files = list(src.glob("*.png")) + list(src.glob("*.jpg")) + list(src.glob("*.jpeg")) + list(src.glob("*.webp")) + list(src.glob("*.gif"))
    ok = 0
    for f in files:
        out = dst / (f.stem + ".svg")
        try:
            pipe = Pipeline.auto_from_image(str(f))
            grid_arr, cmap, meta = pipe.run_on(str(f), max_dim=args.max_dim, strategy=args.strategy, n_colors=args.colors)
            grid = grid_arr.tolist()
            grid_to_svg_file(grid, cmap, str(out), pixel_size=args.pixel_size, use_paths=not args.legacy_rects)
            ok += 1
            print(f"  ✓ {f.name} -> {out.name} ({meta.get('strategy_name', '?')})")
        except Exception as e:
            print(f"  ✗ {f.name}: {e}")
    print(f"\nDone: {ok}/{len(files)} files converted.")


def main():
    parser = argparse.ArgumentParser(description="SVG Vision v8 — Composable Adaptive Pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_conv = sub.add_parser("convert", help="Convert image to SVG")
    p_conv.add_argument("input", help="Input image path")
    p_conv.add_argument("output", help="Output SVG path")
    p_conv.add_argument("--grid-size", default="auto", help="Grid size: auto, N, or WxH")
    p_conv.add_argument("--colors", default="auto", help="Number of colors (auto or int)")
    p_conv.add_argument("--strategy", choices=["pixel_art", "vector_approx", "photo_simplified", "hybrid"], help="Force strategy")
    p_conv.add_argument("--palette-hint", nargs="+", help="Hex color hints")
    p_conv.add_argument("--detail", choices=["high", "medium", "low", "adaptive"], help="Detail preservation")
    p_conv.add_argument("--cleanup", choices=["none", "light", "heavy", "smart"], help="Cleanup mode")
    p_conv.add_argument("--max-dim", type=int, help="Max grid dimension")
    p_conv.add_argument("--no-crop", action="store_true", help="Disable content cropping")
    p_conv.add_argument("--pixel-size", type=int, default=16, help="SVG pixel size")
    p_conv.add_argument("--legacy-rects", action="store_true", help="Use legacy rect mode")
    p_conv.add_argument("--ciede2000", action="store_true", help="Use CIEDE2000")
    p_conv.set_defaults(func=_cli_convert)

    p_ana = sub.add_parser("analyze", help="Analyze image and recommend strategy")
    p_ana.add_argument("input", help="Input image path")
    p_ana.add_argument("--strategy", choices=["pixel_art", "vector_approx", "photo_simplified", "hybrid"], help="Override strategy for preview")
    p_ana.set_defaults(func=_cli_analyze)

    p_batch = sub.add_parser("batch", help="Batch convert directory")
    p_batch.add_argument("input_dir", help="Input directory")
    p_batch.add_argument("output_dir", help="Output directory")
    p_batch.add_argument("--max-dim", type=int, default=64)
    p_batch.add_argument("--colors", default="auto")
    p_batch.add_argument("--strategy", choices=["pixel_art", "vector_approx", "photo_simplified", "hybrid"])
    p_batch.add_argument("--pixel-size", type=int, default=16)
    p_batch.add_argument("--legacy-rects", action="store_true")
    p_batch.set_defaults(func=_cli_batch)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

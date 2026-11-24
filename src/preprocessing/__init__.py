"""Image processing: colour handling, denoising, CLAHE, segmentation, normalisation."""

from src.preprocessing.pipeline import (
    PRESETS,
    PreprocessConfig,
    apply_mask,
    bgr_to_rgb,
    bilateral_filter,
    clahe_enhance,
    crop_to_mask,
    denoise,
    display_stages,
    estimate_background_colour,
    gaussian_filter,
    get_preset,
    image_quality_metrics,
    median_filter,
    normalize,
    preprocess_image,
    preprocess_with_stages,
    resize,
    rgb_to_bgr,
    segment_vegetable,
    to_hsv,
    to_lab,
)

__all__ = [
    "PRESETS", "PreprocessConfig", "apply_mask", "bgr_to_rgb", "bilateral_filter",
    "clahe_enhance", "crop_to_mask", "denoise", "display_stages",
    "estimate_background_colour", "gaussian_filter", "get_preset",
    "image_quality_metrics", "median_filter", "normalize", "preprocess_image",
    "preprocess_with_stages", "resize", "rgb_to_bgr", "segment_vegetable",
    "to_hsv", "to_lab",
]

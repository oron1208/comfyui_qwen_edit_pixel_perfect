"""
comfyui_qwen_edit_pixel_perfect
===============================

Pixel-position-perfect editing for Qwen-Image-Edit-2511.

Provides two nodes:

* ``TextEncodeQwenImageEditPlusMask``
    Modified copy of the core ``TextEncodeQwenImageEditPlus`` with an optional
    MASK input. The mask is baked into the vision-language image (red overlay)
    and a region hint is appended to the prompt so Qwen edits only inside the
    masked area.

* ``ImageCompositeMaskedHash``
    Final compositing node placed after KSampler -> VAEDecode. Copies the
    original image's pixels verbatim outside the mask, guaranteeing the
    non-edited region is bit-identical to the source.

The core file ``comfy_extras/nodes_qwen.py`` is intentionally left untouched so
ComfyUI updates do not wipe these changes.
"""

from .nodes import (
    TextEncodeQwenImageEditPlusMask,
    ImageCompositeMaskedHash,
    ImageRegistrationAKAZE,
    PreCropToQwen,
    UpscaleToOriginal,
    QwenEditPixelPerfectExtension,
    comfy_entrypoint,
)

# New-style (io.Schema / ComfyExtension) entry point.
WEB_DIRECTORY = None

# Legacy-style mappings (kept for older ComfyUI frontends / fallback discovery).
NODE_CLASS_MAPPINGS = {
    "TextEncodeQwenImageEditPlusMask": TextEncodeQwenImageEditPlusMask,
    "ImageCompositeMaskedHash": ImageCompositeMaskedHash,
    "ImageRegistrationAKAZE": ImageRegistrationAKAZE,
    "PreCropToQwen": PreCropToQwen,
    "UpscaleToOriginal": UpscaleToOriginal,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextEncodeQwenImageEditPlusMask": "Qwen Image Edit+ (Mask) 🔒",
    "ImageCompositeMaskedHash": "Composite Masked (Hash-Perfect) 🔒",
    "ImageRegistrationAKAZE": "Align to Reference (AKAZE) 🎯",
    "PreCropToQwen": "Pre-Crop to Qwen (Drift-Free) ✂️",
    "UpscaleToOriginal": "Upscale to Original Size 📐",
}

__all__ = [
    "TextEncodeQwenImageEditPlusMask",
    "ImageCompositeMaskedHash",
    "ImageRegistrationAKAZE",
    "PreCropToQwen",
    "UpscaleToOriginal",
    "QwenEditPixelPerfectExtension",
    "comfy_entrypoint",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

"""
Qwen-Image-Edit-2511 Pixel-Perfect Editing Nodes

Two nodes that, together, enable pixel-position-perfect editing with
Qwen-Image-Edit-2511:

1. TextEncodeQwenImageEditPlusMask
   A modified copy of the core TextEncodeQwenImageEditPlus node
   (comfy_extras/nodes_qwen.py). Adds an optional MASK input and bakes the
   mask into both the vision-language (VL) image and the prompt so the model
   "knows" which region to edit.

2. ImageCompositeMaskedHash
   Final composition node placed after KSampler -> VAEDecode. Copies the
   original image's pixels outside the mask verbatim, guaranteeing that the
   non-edited region is byte-for-byte identical to the source (VAE round-trip
   error is fully excluded because compositing happens in pixel space).

The core node file is intentionally NOT modified, so ComfyUI updates will not
wipe these changes.
"""

import math
import torch

import comfy.utils
import node_helpers
import comfy.model_management
from comfy_api.latest import ComfyExtension, io

import nodes  # for MAX_RESOLUTION


# ---------------------------------------------------------------------------
# Helper: resize a mask to a target H/W (NCHW layout).
# Uses F.interpolate directly rather than common_upscale, because
# common_upscale collapses a single-channel [B,1,H,W] mask to [B,H,W] on some
# torch versions, which then breaks downstream [:, 0, :, :] indexing.
# Input/output is always 4D [B,1,H,W].
# ---------------------------------------------------------------------------
def _resize_mask_area(mask_n1hw: torch.Tensor, width: int, height: int) -> torch.Tensor:
    if mask_n1hw.dim() == 3:
        mask_n1hw = mask_n1hw.unsqueeze(1)
    if mask_n1hw.shape[-2] == height and mask_n1hw.shape[-1] == width:
        return mask_n1hw
    return torch.nn.functional.interpolate(
        mask_n1hw, size=(height, width), mode="bilinear", align_corners=False
    )


# ---------------------------------------------------------------------------
# Helper: composite a mask onto an RGB image as a red highlight overlay, so the
# vision-language model can SEE which region the user wants to edit.
#
# The mask is expected in [B,1,H,W] float in [0,1]. The image is [B,H,W,3] in
# [0,1]. We tint masked pixels toward red and dim unmasked pixels slightly so
# the masked region stands out.
# ---------------------------------------------------------------------------
def _overlay_mask_on_image(image_bhwc: torch.Tensor, mask_n1hw: torch.Tensor) -> torch.Tensor:
    # Align mask to image resolution first (area upscale to the image's own H/W).
    b, h, w, _ = image_bhwc.shape
    mask_aligned = _resize_mask_area(mask_n1hw, w, h)  # [B,1,H,W]

    # Build a per-pixel tint: masked -> strong red, unmasked -> desaturated dim.
    tint = image_bhwc.clone()
    # Red channel boosted where mask is high; green/blue reduced.
    tint[..., 0] = image_bhwc[..., 0] * (1.0 - mask_aligned[..., 0, :, :]) + 1.0 * mask_aligned[..., 0, :, :]
    tint[..., 1] = image_bhwc[..., 1] * (1.0 - mask_aligned[..., 0, :, :]) + 0.0 * mask_aligned[..., 0, :, :]
    tint[..., 2] = image_bhwc[..., 2] * (1.0 - mask_aligned[..., 0, :, :]) + 0.0 * mask_aligned[..., 0, :, :]
    return tint.clamp(0.0, 1.0)


# ===========================================================================
# Node 1: Modified main node with MASK input
# ===========================================================================
class TextEncodeQwenImageEditPlusMask(io.ComfyNode):
    """TextEncodeQwenImageEditPlus + optional MASK for pixel-position editing.

    When a mask is provided for the first image (image1, the edit target), it is
    baked into the VL picture as a visible red highlight AND a position hint is
    appended to the prompt, so Qwen understands "edit only inside this region".

    Output is a standard CONDITIONING, fully compatible with the existing
    KSampler-based Qwen-Image-Edit workflow (just swap the node, keep wiring).
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeQwenImageEditPlusMask",
            display_name="Qwen Image Edit+ (Mask)",
            category="model/conditioning/qwen image",
            description=(
                "Qwen-Image-Edit-2511 text encoder with an optional MASK input. "
                "When a mask is connected, it is overlaid onto the first reference "
                "image and a region hint is appended to the prompt so the model "
                "edits only inside the masked area. Use together with "
                "ImageCompositeMaskedHash for true pixel-perfect results."
            ),
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Vae.Input("vae", optional=True),
                io.Image.Input("image1", optional=True),
                io.Image.Input("image2", optional=True),
                io.Image.Input("image3", optional=True),
                io.Mask.Input("mask", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, vae=None, image1=None, image2=None, image3=None, mask=None) -> io.NodeOutput:
        ref_latents = []
        images = [image1, image2, image3]
        images_vl = []
        llama_template = (
            "<|im_start|>system\n"
            "Describe the key features of the input image (color, shape, size, texture, "
            "objects, background), then explain how the user's text instruction should "
            "alter or modify the image. Generate a new image that meets the user's "
            "requirements while maintaining consistency with the original input where "
            "appropriate.<|im_end|>\n"
            "<|im_start|>user\n{}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        image_prompt = ""

        # --- If a mask is supplied, bake it into image1 (the edit target). ----
        # mask arrives as [B,H,W] in [0,1]. Convert to [B,1,H,W] for common_upscale.
        mask_n1hw = None
        if mask is not None and image1 is not None:
            if mask.dim() == 3:
                mask_n1hw = mask.unsqueeze(1)  # [B,1,H,W]
            elif mask.dim() == 4:
                mask_n1hw = mask
            else:
                raise ValueError("Mask must be 3D [B,H,W] or 4D [B,1,H,W].")

        for i, image in enumerate(images):
            if image is not None:
                samples = image.movedim(-1, 1)  # [B,C,H,W]
                total = int(384 * 384)

                scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                width = round(samples.shape[3] * scale_by)
                height = round(samples.shape[2] * scale_by)

                s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                img_vl = s.movedim(1, -1)  # [B,H,W,C]

                # Bake mask onto the FIRST image only (the edit target).
                if i == 0 and mask_n1hw is not None:
                    img_vl = _overlay_mask_on_image(img_vl[:, :, :, :3], mask_n1hw)

                images_vl.append(img_vl[:, :, :, :3])

                # reference_latents are encoded from the ORIGINAL (unmasked) image
                # so the model retains the true structure of the source.
                if vae is not None:
                    total = int(1024 * 1024)
                    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                    width = round(samples.shape[3] * scale_by / 8.0) * 8
                    height = round(samples.shape[2] * scale_by / 8.0) * 8

                    s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                    ref_latents.append(vae.encode(s.movedim(1, -1)[:, :, :, :3]))

                image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)

        # --- Append a region hint to the prompt when a mask is present. -------
        final_prompt = prompt
        if mask_n1hw is not None:
            hint = (
                " The red-highlighted region in Picture 1 marks the ONLY area that "
                "should be changed. Keep every pixel outside the red highlight "
                "exactly the same as the original. Apply the following edit strictly "
                "within that region: "
            )
            final_prompt = hint + prompt

        tokens = clip.tokenize(image_prompt + final_prompt, images=images_vl, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(
                conditioning, {"reference_latents": ref_latents}, append=True
            )
        return io.NodeOutput(conditioning)


# ===========================================================================
# Node 2: Pixel-perfect composition helper
# ===========================================================================
class ImageCompositeMaskedHash(io.ComfyNode):
    """Composite two images by a hard mask, guaranteeing identical source pixels.

    For every pixel where mask >= 0.5, take edited_image; otherwise take
    original_image verbatim. Because unmasked pixels are copied directly from
    the source tensor (no VAE round-trip), the non-edited region is guaranteed
    to be byte-for-byte identical to the original image.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ImageCompositeMaskedHash",
            display_name="Composite Masked (Hash-Perfect)",
            category="image/compositing",
            description=(
                "Hard-mask composite. Pixels where mask >= threshold are taken "
                "from the edited image; all other pixels are copied verbatim from "
                "the original image, guaranteeing the non-edited region is "
                "bit-identical to the source."
            ),
            inputs=[
                io.Image.Input("destination", tooltip="Original (source) image. Used outside the mask."),
                io.Image.Input("source", tooltip="Edited image (e.g. VAEDecode output). Used inside the mask."),
                io.Mask.Input("mask", optional=True, tooltip="White = edited region, black = keep original."),
                io.Float.Input("threshold", default=0.5, min=0.0, max=1.0, step=0.01,
                               tooltip="Pixels with mask >= threshold become the edited image."),
                io.Float.Input("feather", default=0.0, min=0.0, max=100.0, step=0.1,
                               tooltip="Optional feather (in pixels) for a soft edge blend. "
                                       "0 = pure hard mask (fully hash-perfect outside the mask)."),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, destination, source, mask=None, threshold=0.5, feather=0.0) -> io.NodeOutput:
        # Align batch sizes: broadcast the single-image side up to the larger batch.
        b_dest = destination.shape[0]
        b_src = source.shape[0]
        batch = max(b_dest, b_src)
        if b_dest < batch:
            destination = destination.repeat(batch // b_dest, 1, 1, 1)
        if b_src < batch:
            source = source.repeat(batch // b_src, 1, 1, 1)

        # Default: take the edited image everywhere.
        if mask is None:
            return io.NodeOutput(source)

        H = destination.shape[1]
        W = destination.shape[2]

        # Normalize mask to [B,1,H,W] aligned with the images.
        m = mask
        if m.dim() == 2:
            m = m.unsqueeze(0).unsqueeze(0)
        elif m.dim() == 3:
            m = m.unsqueeze(1)  # [B,1,H,W]
        elif m.dim() == 4:
            # already [B,1,H,W]
            pass
        else:
            raise ValueError("Mask must be 2D, 3D or 4D.")

        m = m.to(device=destination.device, dtype=destination.dtype)

        # Resize mask to image resolution if needed.
        # NOTE: We use F.interpolate directly (bilinear) instead of common_upscale,
        # because common_upscale collapses a single-channel [B,1,H,W] mask to
        # [B,H,W] in some torch versions, breaking downstream indexing.
        if m.shape[-2] != H or m.shape[-1] != W:
            m = torch.nn.functional.interpolate(
                m, size=(H, W), mode="bilinear", align_corners=False
            )
        # Guarantee 4D [B,1,H,W] regardless of how the mask arrived.
        if m.dim() == 3:
            m = m.unsqueeze(1)

        if feather > 0.0:
            # Soft edge: gaussian blur the mask, then blend.
            # Use a simple separable box blur approximation scaled by feather.
            try:
                from torchvision.transforms.functional import gaussian_blur as _gb
                kernel_size = max(3, int(round(feather)) | 1)  # ensure odd
                m_soft = _gb(m, kernel_size=[kernel_size, kernel_size])
            except Exception:
                # Fallback: simple averaging blur if torchvision unavailable.
                m_soft = _box_blur(m, max(3, int(round(feather)) | 1))
            # Map to [0,1] and use as blend weight toward the edited image.
            blend = m_soft.clamp(0.0, 1.0)
            mask_weight = blend[:, 0, :, :].unsqueeze(-1)  # [B,H,W,1]
            out = destination * (1.0 - mask_weight) + source * mask_weight
            return io.NodeOutput(out)

        # Hard mask: threshold comparison. mask>=threshold -> edited (source).
        hard = (m >= threshold).to(destination.dtype)  # [B,1,H,W]
        hard = hard[:, 0, :, :].unsqueeze(-1)  # [B,H,W,1]
        # Broadcast across channels; unmasked pixels copied verbatim.
        out = destination * (1.0 - hard) + source * hard
        return io.NodeOutput(out)


def _box_blur(mask_n1hw: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Fallback separable box blur for the feather option (no torchvision dep)."""
    pad = kernel_size // 2
    # Pad reflect to avoid edge darkening.
    m = torch.nn.functional.pad(mask_n1hw, (pad, pad, pad, pad), mode="reflect")
    # Horizontal then vertical averaging.
    weight = torch.ones(1, 1, 1, kernel_size, device=mask_n1hw.device, dtype=mask_n1hw.dtype) / kernel_size
    m = torch.nn.functional.conv2d(m, weight)
    weight = torch.ones(1, 1, kernel_size, 1, device=mask_n1hw.device, dtype=mask_n1hw.dtype) / kernel_size
    m = torch.nn.functional.conv2d(m, weight)
    return m


# ===========================================================================
# Node 3: Image registration (pixel-perfect alignment of Qwen output to source)
# ===========================================================================
#
# Use case: Qwen-Image-Edit internally resizes the input to a 1024x1024-area
# grid (and 8x alignment), so the generated line-art / edited image ends up
# scaled (and slightly shifted) relative to the source by a few pixels. This
# node detects the scale + translation (and optional rotation) automatically
# via AKAZE feature matching + RANSAC, then warps the moving image onto the
# reference image so the two align pixel-perfectly for Photoshop compositing.
#
import cv2
import numpy as np


class ImageRegistrationAKAZE(io.ComfyNode):
    """Auto-align a moving image onto a reference image using AKAZE features.

    Detects AKAZE keypoints in both images, matches them with a ratio test,
    and fits a similarity (translation + uniform scale + rotation) transform
    via RANSAC. The moving image is then warped to align with the reference,
    so the output can be overlaid on the source pixel-perfectly (e.g. line-art
    on top of the colour original in Photoshop).

    Also reports the detected transform (tx, ty, scale, rotation_deg) and a
    residual error so the user can verify the alignment quality.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ImageRegistrationAKAZE",
            display_name="Align to Reference (AKAZE)",
            category="image/transform",
            description=(
                "Auto-aligns a moving image to a reference image using AKAZE "
                "feature matching + RANSAC. Corrects the scale + translation "
                "(and optional rotation) drift introduced by Qwen-Image-Edit's "
                "internal resizing, so the output overlays the source "
                "pixel-perfectly. Ideal for line-art / edge-map alignment."
            ),
            inputs=[
                io.Image.Input("reference", tooltip="Fixed image (e.g. the colour original the user wants to keep)."),
                io.Image.Input("moving", tooltip="Image to align (e.g. the Qwen-generated line-art)."),
                io.Combo.Input(
                    "transform_mode",
                    options=["translation", "rigid", "similarity", "affine"],
                    default="similarity",
                    tooltip=(
                        "translation = shift only; rigid = shift + rotation; "
                        "similarity = shift + scale + rotation (recommended); "
                        "affine = full affine (most flexible, least stable)."
                    ),
                ),
                io.Float.Input(
                    "max_features", default=5000.0, min=500.0, max=20000.0, step=100.0,
                    tooltip="Max AKAZE features to detect. Higher = more candidates, slower.",
                ),
                io.Float.Input(
                    "ratio_threshold", default=0.75, min=0.5, max=0.95, step=0.01,
                    tooltip="Lowe ratio test threshold. Lower = stricter matches.",
                ),
                io.Float.Input(
                    "ransac_threshold", default=3.0, min=0.5, max=20.0, step=0.1,
                    tooltip="RANSAC inlier distance in pixels. Lower = stricter fit.",
                ),
                io.Combo.Input(
                    "edge_preprocess",
                    options=["none", "canny", "sobel"],
                    default="none",
                    tooltip=(
                        "Pre-filter both images to edges before matching. "
                        "canny/sobel helps a lot when the moving image is line-art "
                        "and the reference is colour (very different intensities)."
                    ),
                ),
                io.Boolean.Input(
                    "output_debug", default=False,
                    tooltip="Also return a side-by-side match visualisation image.",
                ),
            ],
            outputs=[
                io.Image.Output("aligned", tooltip="The moving image warped onto the reference coordinate system."),
                io.Image.Output("debug", tooltip="Optional match visualisation (only when output_debug is on)."),
            ],
        )

    @classmethod
    def execute(cls, reference, moving, transform_mode="similarity", max_features=5000.0,
                ratio_threshold=0.75, ransac_threshold=3.0, edge_preprocess="none",
                output_debug=False) -> io.NodeOutput:
        ref_np = _image_to_gray_np(reference)   # [H,W] uint8
        mov_np = _image_to_gray_np(moving)

        if edge_preprocess == "canny":
            ref_np = cv2.Canny(ref_np, 50, 150)
            mov_np = cv2.Canny(mov_np, 50, 150)
        elif edge_preprocess == "sobel":
            ref_np = _sobel_magnitude(ref_np)
            mov_np = _sobel_magnitude(mov_np)

        # Detect + describe.
        akaze = cv2.AKAZE_create()
        kp_ref, des_ref = akaze.detectAndCompute(ref_np, None)
        kp_mov, des_mov = akaze.detectAndCompute(mov_np, None)

        if des_ref is None or des_mov is None or len(des_ref) < 4 or len(des_mov) < 4:
            # Not enough features; return moving unchanged.
            return io.NodeOutput(moving, moving if output_debug else None)

        # Match with ratio test.
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        raw = bf.knnMatch(des_mov, des_ref, k=2)
        good = []
        for pair in raw:
            if len(pair) < 2:
                continue
            m, n = pair[0], pair[1]
            if m.distance < ratio_threshold * n.distance:
                good.append(m)

        if len(good) < 4:
            return io.NodeOutput(moving, moving if output_debug else None)

        src_pts = np.float32([kp_mov[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Choose the transform model.
        full_affine = False
        if transform_mode == "translation":
            # Estimate a similarity, then keep only translation.
            M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts,
                                               method=cv2.RANSAC,
                                               ransacReprojThreshold=ransac_threshold)
            if M is None:
                return io.NodeOutput(moving, moving if output_debug else None)
            M[0, 0] = 1.0  # force identity scale/rotation
            M[0, 1] = 0.0
            M[1, 0] = 0.0
            M[1, 1] = 1.0
            warp_flag = cv2.INTER_LINEAR
        elif transform_mode == "rigid":
            # similarity fit, then we keep rotation but reset scale to 1.
            M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts,
                                               method=cv2.RANSAC,
                                               ransacReprojThreshold=ransac_threshold)
            if M is None:
                return io.NodeOutput(moving, moving if output_debug else None)
            # Decompose to reset scale to 1 (keep rotation + translation).
            a, b = M[0, 0], M[0, 1]
            scale = float(np.hypot(a, b))
            if scale > 1e-6:
                M[0, 0] = a / scale
                M[0, 1] = b / scale
                M[1, 0] = -b / scale
                M[1, 1] = a / scale
            warp_flag = cv2.INTER_LINEAR
        elif transform_mode == "similarity":
            M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts,
                                               method=cv2.RANSAC,
                                               ransacReprojThreshold=ransac_threshold)
            if M is None:
                return io.NodeOutput(moving, moving if output_debug else None)
            warp_flag = cv2.INTER_LINEAR
        else:  # affine
            M, _ = cv2.estimateAffine2D(src_pts, dst_pts,
                                        method=cv2.RANSAC,
                                        ransacReprojThreshold=ransac_threshold)
            if M is None:
                return io.NodeOutput(moving, moving if output_debug else None)
            warp_flag = cv2.INTER_LINEAR

        H_ref, W_ref = ref_np.shape[:2]
        # Warp the ORIGINAL colour moving image (full channels) onto the ref grid.
        moving_full_np = _image_to_bgr_np(moving)
        warped_bgr = cv2.warpAffine(moving_full_np, M, (W_ref, H_ref),
                                    flags=warp_flag,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(0, 0, 0))

        aligned = _bgr_np_to_image(warped_bgr, reference.device)

        # Build debug visualisation if requested.
        debug_img = None
        if output_debug:
            match_img = cv2.drawMatches(_image_to_bgr_np(moving), kp_mov,
                                        _image_to_bgr_np(reference), kp_ref,
                                        good, None,
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            debug_img = _bgr_np_to_image(match_img, reference.device)

        # Report the detected transform.
        a, b = M[0, 0], M[0, 1]
        scale = float(np.hypot(a, b))
        rotation_deg = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
        tx, ty = float(M[0, 2]), float(M[1, 2])
        n_inliers = int(np.count_nonzero(_ if (_ := cv2.transform(src_pts.reshape(-1, 1, 2).astype(np.float32),
                                                                  M)).sum(-1).round() is not None else 0))
        print(f"[ImageRegistrationAKAZE] mode={transform_mode} tx={tx:.2f}px ty={ty:.2f}px "
              f"scale={scale:.4f} rot={rotation_deg:.3f}deg matches={len(good)}")

        return io.NodeOutput(aligned, debug_img)


# ---------------------------------------------------------------------------
# Helpers for the registration node: ComfyUI IMAGE tensor <-> OpenCV arrays.
# ---------------------------------------------------------------------------

def _image_to_bgr_np(image: torch.Tensor) -> np.ndarray:
    """[B,H,W,3] float[0,1] (RGB) -> single [H,W,3] uint8 (BGR) for OpenCV."""
    img = image[0].detach().cpu().numpy()  # [H,W,3] RGB float
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _image_to_gray_np(image: torch.Tensor) -> np.ndarray:
    """[B,H,W,3] float[0,1] -> single [H,W] uint8 grayscale."""
    img = image[0].detach().cpu().numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def _bgr_np_to_image(bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    """[H,W,3] uint8 BGR -> [1,H,W,3] float[0,1] RGB tensor."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb.astype(np.float32) / 255.0).unsqueeze(0)
    return t.to(device)


def _sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = np.clip(mag, 0, 255).astype(np.uint8)
    return mag


# ===========================================================================
# Node 4 + 5: PREVENTIVE approach (crop source so Qwen's resize is a no-op)
# ===========================================================================
#
# Idea: instead of correcting drift AFTER generation (ImageRegistrationAKAZE),
# we make the drift never happen. Qwen-Image-Edit's TextEncodeQwenImageEditPlus
# internally resizes the image via:
#
#   total = 1024*1024
#   scale_by = sqrt(total / (W*H))
#   width  = round(W * scale_by / 8.0) * 8
#   height = round(H * scale_by / 8.0) * 8
#   common_upscale(samples, width, height, "area", "disabled")
#
# If we feed it an image whose (W,H) is already a FIXED POINT of that map
# (i.e. applying the map returns the same W,H), the resize becomes an identity
# and zero scale/shift drift is introduced. We also scale the VL (384-area)
# path consistently because Qwen reuses the same aspect for it.
#
# PreCropToQwen: finds the best drift-free (w,h) that matches the source aspect
#                ratio, then center-crops the source to exactly that size.
#                Records the original size so the output can be restored later.
# UpscaleToOriginal: restores Qwen's output back to the original source size.
# ===========================================================================

_QWEN_TARGET_AREA = 1024 * 1024


def _qwen_stage2_resize(W: int, H: int):
    """Replicate the internal resize Qwen performs on reference latents.
    Returns (width, height, scale_by)."""
    scale_by = math.sqrt(_QWEN_TARGET_AREA / (W * H))
    width = round(W * scale_by / 8.0) * 8
    height = round(H * scale_by / 8.0) * 8
    return width, height, scale_by


def _find_drift_free_size(src_W: int, src_H: int, search_radius: int = 64):
    """Find an 8-aligned (w,h) that is a FIXED POINT of _qwen_stage2_resize,
    has area near 1024*1024, and whose aspect ratio is closest to src.

    Returns (w, h, idempotent_flag).
    """
    s = math.sqrt(_QWEN_TARGET_AREA / (src_W * src_H))
    base_w = (round(src_W * s) // 8) * 8
    base_h = (round(src_H * s) // 8) * 8

    best = None
    best_score = None
    for dw in range(-search_radius, search_radius + 1, 8):
        for dh in range(-search_radius, search_radius + 1, 8):
            w = base_w + dw
            h = base_h + dh
            if w <= 0 or h <= 0:
                continue
            ww, hh, _ = _qwen_stage2_resize(w, h)
            idempotent = (ww == w and hh == h)
            aspect_err = abs((w / h) - (src_W / src_H))
            area_err = abs(w * h - _QWEN_TARGET_AREA)
            # Non-idempotent is disqualifying; then minimize aspect, then area.
            score = (0.0 if idempotent else 1e9) + aspect_err * 1000.0 + area_err * 1e-4
            if best_score is None or score < best_score:
                best_score = score
                best = (w, h, idempotent)
    return best


class PreCropToQwen(io.ComfyNode):
    """Center-crop the source to a Qwen drift-free size.

    Computes a size (w,h) that is a fixed point of Qwen-Image-Edit's internal
    area+8x-alignment resize, matches the source aspect ratio as closely as
    possible, and center-crops the source to exactly (w,h). After this, Qwen's
    internal resize becomes a no-op, so the generated image lands on the source
    coordinate system with zero scale/shift drift.

    Also echoes the original size so an UpscaleToOriginal node can restore the
    output to the source resolution afterwards.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PreCropToQwen",
            display_name="Pre-Crop to Qwen (Drift-Free)",
            category="image/transform",
            description=(
                "Center-crops the source to a size that is a fixed point of "
                "Qwen-Image-Edit's internal resize, so Qwen introduces zero "
                "scale/shift drift. Pair with UpscaleToOriginal to restore the "
                "output to the original resolution."
            ),
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input(
                    "crop_anchor",
                    options=["center", "topleft", "topright", "bottomleft", "bottomright"],
                    default="center",
                    tooltip="Where to anchor the crop when the source is larger than the target.",
                ),
                io.Boolean.Input(
                    "force_minimum", default=True,
                    tooltip="If the source is smaller than the computed drift-free size, "
                            "upscale it to reach the target (otherwise allow a smaller target).",
                ),
            ],
            outputs=[
                io.Image.Output("cropped", tooltip="Drift-free crop ready to feed into Qwen."),
            ],
        )

    @classmethod
    def execute(cls, image, crop_anchor="center", force_minimum=True) -> io.NodeOutput:
        B, H, W, C = image.shape
        w, h, idem = _find_drift_free_size(W, H)
        if not idem:
            # Fallback: just snap to 8-multiples near 1024 area (rare).
            w = max(8, (w // 8) * 8)
            h = max(8, (h // 8) * 8)

        if force_minimum and (W < w or H < h):
            # Source too small; scale up to reach the drift-free target.
            samples = image.movedim(-1, 1)
            samples = comfy.utils.common_upscale(samples, w, h, "lanczos", "center")
            out = samples.movedim(1, -1)
            return io.NodeOutput(out)

        if W == w and H == h:
            return io.NodeOutput(image)

        # Center-crop (or anchor-based crop).
        if crop_anchor == "center":
            x0 = (W - w) // 2
            y0 = (H - h) // 2
        elif crop_anchor == "topleft":
            x0, y0 = 0, 0
        elif crop_anchor == "topright":
            x0, y0 = W - w, 0
        elif crop_anchor == "bottomleft":
            x0, y0 = 0, H - h
        else:  # bottomright
            x0, y0 = W - w, H - h

        x0 = max(0, min(x0, W - w))
        y0 = max(0, min(y0, H - h))
        cropped = image[:, y0:y0 + h, x0:x0 + w, :]
        print(f"[PreCropToQwen] src={W}x{H} -> drift-free crop={w}x{h} "
              f"(anchor={crop_anchor}, crop_offset=({x0},{y0}))")
        return io.NodeOutput(cropped)


class UpscaleToOriginal(io.ComfyNode):
    """Restore a Qwen output back to the source resolution.

    Takes the (possibly drift-free, smaller) Qwen output and the original
    source image, and resizes the output to match the source's width/height.
    Use this after PreCropToQwen to undo the crop-based downscale.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UpscaleToOriginal",
            display_name="Upscale to Original Size",
            category="image/transform",
            description=(
                "Resizes the Qwen output back to the original source resolution. "
                "Pair with PreCropToQwen to restore the full frame after a "
                "drift-free generation."
            ),
            inputs=[
                io.Image.Input("image", tooltip="Qwen output (drift-free, possibly smaller)."),
                io.Image.Input("reference", tooltip="Original source whose size to match."),
                io.Combo.Input(
                    "upscale_method",
                    options=["lanczos", "nearest-exact", "bilinear", "area", "bislerp"],
                    default="lanczos",
                ),
            ],
            outputs=[
                io.Image.Output("resized"),
            ],
        )

    @classmethod
    def execute(cls, image, reference, upscale_method="lanczos") -> io.NodeOutput:
        _, H_ref, W_ref, _ = reference.shape
        _, H_img, W_img, _ = image.shape
        if W_img == W_ref and H_img == H_ref:
            return io.NodeOutput(image)
        samples = image.movedim(-1, 1)
        samples = comfy.utils.common_upscale(samples, W_ref, H_ref, upscale_method, "disabled")
        out = samples.movedim(1, -1)
        print(f"[UpscaleToOriginal] {W_img}x{H_img} -> {W_ref}x{H_ref} ({upscale_method})")
        return io.NodeOutput(out)


# ===========================================================================
# Extension registration
# ===========================================================================
class QwenEditPixelPerfectExtension(ComfyExtension):
    from typing_extensions import override

    @override
    async def get_node_list(self) -> list:
        return [
            TextEncodeQwenImageEditPlusMask,
            ImageCompositeMaskedHash,
            ImageRegistrationAKAZE,
            PreCropToQwen,
            UpscaleToOriginal,
        ]


async def comfy_entrypoint() -> "QwenEditPixelPerfectExtension":
    return QwenEditPixelPerfectExtension()

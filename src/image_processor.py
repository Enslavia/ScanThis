"""
Image processing pipeline for ScanThis2.
Replicates lookscanned.io-style document scanning.
"""

import cv2
import numpy as np
import fitz  # PyMuPDF
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import random
import io
from PIL import Image


class ColorMode(Enum):
    COLOR = "color"
    GRAYSCALE = "grayscale"
    BLACK_WHITE = "bw"


@dataclass
class ProcessingParams:
    """Parameters for the image processing pipeline."""
    color_mode: ColorMode = ColorMode.COLOR
    rotation: float = 0.5          # base rotation in degrees
    rotation_variance: float = 0.0  # random variance around rotation
    brightness: float = 0.0       # -100 to +100
    contrast: float = 0.0        # -100 to +100
    blur: float = 0.0            # kernel radius (0 = none)
    noise: float = 0.0           # 0 to 50 (sigma)
    yellowing: float = 0.0       # 0 to 1 (sepia/yellow tint)
    resolution: int = 300        # DPI

    def get_rotation_with_variance(self) -> float:
        """Return rotation with applied random variance."""
        if self.rotation_variance > 0:
            return self.rotation + random.uniform(-self.rotation_variance, self.rotation_variance)
        return self.rotation


def default_scanned_params() -> ProcessingParams:
    """
    Return parameters that produce a subtle, readable scanned-document look.
    """
    return ProcessingParams(
        color_mode=ColorMode.COLOR,
        rotation=0.5,
        rotation_variance=0.0,
        brightness=-2.0,
        contrast=3.0,
        blur=0.0,
        noise=0.3,
        yellowing=0.02,
        resolution=300,
    )


def _rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image by angle degrees around its center."""
    if angle == 0:
        return img
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(m[0, 0])
    sin = np.abs(m[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    m[0, 2] += (new_w / 2) - center[0]
    m[1, 2] += (new_h / 2) - center[1]
    # White border: (255,255,255) for 3-channel, 255 for grayscale
    if len(img.shape) == 3:
        border_val = (255, 255, 255)
    else:
        border_val = 255
    return cv2.warpAffine(img, m, (new_w, new_h),
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border_val)


def _apply_brightness(img: np.ndarray, brightness: float) -> np.ndarray:
    """Apply brightness adjustment. Range: -100 to +100."""
    if brightness == 0:
        return img
    delta = brightness * 2.55
    result = img.astype(np.float32) + delta
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def _apply_contrast(img: np.ndarray, contrast: float) -> np.ndarray:
    """Apply contrast adjustment. Range: -100 to +100."""
    if contrast == 0:
        return img
    alpha = 1.0 + (contrast / 100.0) * 2.0
    alpha = max(0.01, min(3.0, alpha))
    return cv2.convertScaleAbs(img, alpha=alpha, beta=0)


def _apply_blur(img: np.ndarray, blur_radius: float) -> np.ndarray:
    """Apply Gaussian blur. radius=0 means no blur."""
    if blur_radius <= 0:
        return img
    k = int(blur_radius * 2 + 1)
    if k % 2 == 0:
        k += 1
    k = max(1, k)
    return cv2.GaussianBlur(img, (k, k), 0)


def _apply_noise(img: np.ndarray, noise_sigma: float) -> np.ndarray:
    """Add Gaussian noise to simulate paper texture."""
    if noise_sigma <= 0:
        return img
    sigma = noise_sigma / 10.0
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise * 255.0
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def _apply_yellowing(img: np.ndarray, yellowing_amount: float) -> np.ndarray:
    """Apply warm/yellow aging effect to simulate old paper."""
    if yellowing_amount <= 0:
        return img
    kernel = np.array([
        [0.393 + 0.607 * (1 - yellowing_amount),
         0.769 - 0.269 * yellowing_amount,
         0.189 - 0.189 * yellowing_amount],
        [0.349 - 0.349 * yellowing_amount,
         0.686 + 0.314 * (1 - yellowing_amount),
         0.168 - 0.168 * yellowing_amount],
        [0.272 - 0.272 * yellowing_amount,
         0.534 - 0.134 * yellowing_amount,
         0.131 + 0.869 * yellowing_amount]
    ])
    sepia = cv2.transform(img, kernel)
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)
    return cv2.addWeighted(sepia, yellowing_amount, img, 1.0 - yellowing_amount, 0)


def _to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert image to grayscale."""
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _to_black_white(img: np.ndarray) -> np.ndarray:
    """Convert image to black & white using adaptive threshold."""
    gray = _to_grayscale(img)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)


def process_image(img: np.ndarray, params: ProcessingParams,
                  preview_scale: float = 1.0,
                  apply_random_variance: bool = False) -> np.ndarray:
    """
    Main processing pipeline.

    For preview mode (preview_scale < 1.0): downscale first, then apply effects
    so that noise/grain and blur appear proportional to the displayed size.

    Args:
        img: Input image as numpy array (BGR or grayscale).
        params: Processing parameters.
        preview_scale: Scale factor for preview (1.0 = full resolution export).
        apply_random_variance: If True, apply random rotation variance.

    Returns:
        Processed image as numpy array (BGR).
    """
    # Ensure 3-channel BGR for processing
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    result = img.copy()

    # --- Preview: downscale first so effects scale proportionally ---
    if preview_scale != 1.0 and preview_scale > 0 and preview_scale < 1.0:
        h, w = result.shape[:2]
        new_w = max(1, int(w * preview_scale))
        new_h = max(1, int(h * preview_scale))
        result = cv2.resize(result, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # --- Color Mode ---
    if params.color_mode == ColorMode.GRAYSCALE:
        gray = _to_grayscale(result)
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif params.color_mode == ColorMode.BLACK_WHITE:
        bw = _to_black_white(result)
        result = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

    # --- Rotation ---
    rotation_angle = params.rotation
    if apply_random_variance and params.rotation_variance > 0:
        rotation_angle = params.get_rotation_with_variance()
    result = _rotate_image(result, rotation_angle)

    # --- Brightness ---
    result = _apply_brightness(result, params.brightness)

    # --- Contrast ---
    result = _apply_contrast(result, params.contrast)

    # --- Blur ---
    result = _apply_blur(result, params.blur)

    # --- Noise ---
    result = _apply_noise(result, params.noise)

    # --- Yellowing ---
    result = _apply_yellowing(result, params.yellowing)

    return result


def process_pdf_page(pdf_page, params: ProcessingParams,
                      preview_scale: float = 1.0,
                      apply_random_variance: bool = False) -> np.ndarray:
    """
    Rasterize a single PDF page and process it.

    Args:
        pdf_page: PyMuPDF page object.
        params: Processing parameters.
        preview_scale: Scale for preview mode.
        apply_random_variance: Apply random rotation variance.

    Returns:
        Processed image as numpy array (BGR for display).
    """
    zoom = params.resolution / 72.0
    mat = pdf_page.transformation_matrix * fitz.Matrix(zoom, zoom)
    pix = pdf_page.get_pixmap(matrix=mat)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return process_image(img, params, preview_scale, apply_random_variance)


def export_images_to_pdf(images: list, output_path: str, dpi: int = 300) -> str:
    """
    Export a list of processed images to a PDF.

    Args:
        images: List of processed images (numpy arrays, BGR).
        output_path: Path to save the PDF.
        dpi: Output DPI.

    Returns:
        Path to the saved PDF.
    """
    doc = fitz.open()
    for img in images:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        # JPEG encoding: much smaller files than PNG, full resolution preserved
        pil_img = Image.fromarray(rgb)
        buf = io.BytesIO()
        pil_img.save(buf, format='JPEG', quality=92)
        img_bytes = buf.getvalue()
        # A4 page size (595×842 pts) — comfortable in Preview,
        # image fills the page at full resolution.
        page_w, page_h = 595, 842
        page_rect = fitz.Rect(0, 0, page_w, page_h)
        page = doc.new_page(width=page_w, height=page_h)
        page.insert_image(page_rect, stream=img_bytes)
    doc.save(output_path)
    doc.close()
    return output_path


def create_proxy_image(img: np.ndarray, max_dim: int = 600) -> np.ndarray:
    """Create a downscaled proxy image for fast preview."""
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

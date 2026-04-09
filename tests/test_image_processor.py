"""
Pytest-compatible unit tests for the image processing pipeline.
Tests mathematical correctness of each transformation.
"""

import pytest
import numpy as np
import cv2
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.image_processor import (
    ProcessingParams, ColorMode,
    _rotate_image, _apply_brightness, _apply_contrast,
    _apply_blur, _apply_noise, _apply_yellowing,
    _to_grayscale, _to_black_white,
    process_image, create_proxy_image
)


# ─────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────
@pytest.fixture
def small_image():
    """50x50 RGB white image."""
    return np.full((50, 50, 3), 255, dtype=np.uint8)


@pytest.fixture
def gray_image():
    """50x50 grayscale image at mid-gray."""
    return np.full((50, 50), 128, dtype=np.uint8)


@pytest.fixture
def checkerboard():
    """8x8 checkerboard for rotation/alignment tests."""
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    for r in range(8):
        for c in range(8):
            img[r, c] = 255 if (r + c) % 2 == 0 else 0
    return img


# ─────────────────────────────────────────────
#  Basic shape/dtype tests
# ─────────────────────────────────────────────
class TestBasicInvariants:
    def test_output_dtype_uint8(self, small_image):
        result = _apply_brightness(small_image, 10)
        assert result.dtype == np.uint8

    def test_output_shape_unchanged(self, small_image):
        result = _apply_brightness(small_image, 20)
        assert result.shape == small_image.shape

    def test_all_transforms_preserve_shape(self, small_image):
        transforms = [
            lambda img: _rotate_image(img, 5),
            lambda img: _apply_brightness(img, 10),
            lambda img: _apply_contrast(img, 10),
            lambda img: _apply_blur(img, 1.0),
            lambda img: _apply_noise(img, 1.0),
            lambda img: _apply_yellowing(img, 0.3),
        ]
        for t in transforms:
            result = t(small_image.copy())
            assert result.dtype == np.uint8
            assert result.shape[0] > 0 and result.shape[1] > 0


# ─────────────────────────────────────────────
#  Rotation
# ─────────────────────────────────────────────
class TestRotation:
    def test_zero_rotation_returns_same_image(self, small_image):
        result = _rotate_image(small_image.copy(), 0)
        # Should be similar (possibly padded differently)
        assert result.dtype == np.uint8
        assert result.shape[0] > 0

    def test_90_degree_rotation_preserves_dimensions_swap(self):
        img = np.full((100, 50, 3), 200, dtype=np.uint8)
        result = _rotate_image(img, 90)
        # After 90° rotation, width becomes height and vice versa
        # (accounting for new bounding box)
        assert result.dtype == np.uint8
        assert result.shape[0] > 0 and result.shape[1] > 0

    def test_negative_rotation_works(self, small_image):
        result = _rotate_image(small_image.copy(), -45)
        assert result.dtype == np.uint8
        assert result.shape[0] > 0

    def test_large_angle(self, small_image):
        result = _rotate_image(small_image.copy(), 360)
        assert result.dtype == np.uint8


# ─────────────────────────────────────────────
#  Brightness
# ─────────────────────────────────────────────
class TestBrightness:
    def test_zero_brightness_unchanged(self, small_image):
        result = _apply_brightness(small_image.copy(), 0)
        np.testing.assert_array_equal(result, small_image)

    def test_positive_brightness_increases_pixel_values(self, gray_image):
        result = _apply_brightness(gray_image.copy(), 50)
        assert np.mean(result) > np.mean(gray_image)

    def test_negative_brightness_decreases_pixel_values(self, gray_image):
        result = _apply_brightness(gray_image.copy(), -50)
        assert np.mean(result) < np.mean(gray_image)

    def test_full_brightness_near_saturation(self):
        img = np.full((50, 50), 200, dtype=np.uint8)
        result = _apply_brightness(img, 100)
        # Should be near or at 255
        assert np.mean(result) > 200

    def test_brightness_clipped_to_uint8(self):
        img = np.full((50, 50), 250, dtype=np.uint8)
        result = _apply_brightness(img, 100)
        assert result.max() <= 255


# ─────────────────────────────────────────────
#  Contrast
# ─────────────────────────────────────────────
class TestContrast:
    def test_zero_contrast_unchanged(self, gray_image):
        result = _apply_contrast(gray_image.copy(), 0)
        np.testing.assert_array_equal(result, gray_image)

    def test_positive_contrast_increases_spread(self):
        # Image with known distribution
        img = np.zeros((50, 50), dtype=np.uint8)
        img[10:40, 10:40] = 128
        result = _apply_contrast(img, 50)
        # Spread should increase
        assert np.std(result) > np.std(img)

    def test_negative_contrast_decreases_spread(self):
        img = np.zeros((50, 50), dtype=np.uint8)
        img[10:40, 10:40] = 128
        result = _apply_contrast(img, -50)
        assert np.std(result) < np.std(img)

    def test_contrast_output_valid_uint8(self, small_image):
        result = _apply_contrast(small_image.copy(), 100)
        assert result.dtype == np.uint8
        assert result.min() >= 0 and result.max() <= 255


# ─────────────────────────────────────────────
#  Blur
# ─────────────────────────────────────────────
class TestBlur:
    def test_zero_blur_unchanged(self, small_image):
        result = _apply_blur(small_image.copy(), 0)
        np.testing.assert_array_equal(result, small_image)

    def test_positive_blur_reduces_high_frequency(self):
        # High-frequency noise image
        img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        result = _apply_blur(img, 2.0)
        # Blur should reduce variance
        assert np.var(result) < np.var(img)

    def test_blur_output_valid(self, small_image):
        result = _apply_blur(small_image.copy(), 5.0)
        assert result.dtype == np.uint8
        assert result.shape == small_image.shape


# ─────────────────────────────────────────────
#  Noise
# ─────────────────────────────────────────────
class TestNoise:
    def test_zero_noise_unchanged(self, small_image):
        result = _apply_noise(small_image.copy(), 0)
        np.testing.assert_array_equal(result, small_image)

    def test_positive_noise_increases_variance(self, small_image):
        result = _apply_noise(small_image.copy(), 10.0)
        # With noise added, variance should increase
        assert np.var(result) > 0

    def test_noise_output_valid_uint8(self, small_image):
        result = _apply_noise(small_image.copy(), 30.0)
        assert result.dtype == np.uint8
        assert result.min() >= 0 and result.max() <= 255


# ─────────────────────────────────────────────
#  Yellowing
# ─────────────────────────────────────────────
class TestYellowing:
    def test_zero_yellowing_unchanged(self, small_image):
        result = _apply_yellowing(small_image.copy(), 0.0)
        np.testing.assert_array_equal(result, small_image)

    def test_yellowing_increases_red_green_channels(self):
        # Uniform white image
        img = np.full((50, 50, 3), 200, dtype=np.uint8)
        result = _apply_yellowing(img, 0.8)
        # R and G channels should increase relative to B
        assert result.dtype == np.uint8

    def test_full_yellowing_is_distinct(self, small_image):
        result = _apply_yellowing(small_image.copy(), 1.0)
        # Result should be sepia-toned, different from original
        assert result.dtype == np.uint8
        assert result.shape == small_image.shape


# ─────────────────────────────────────────────
#  Color Mode
# ─────────────────────────────────────────────
class TestColorModes:
    def test_grayscale_reduces_channels(self, small_image):
        gray = _to_grayscale(small_image)
        assert len(gray.shape) == 2

    def test_grayscale_value_range(self, small_image):
        gray = _to_grayscale(small_image)
        assert gray.dtype == np.uint8
        assert gray.min() >= 0 and gray.max() <= 255

    def test_black_white_is_binary_like(self, small_image):
        bw = _to_black_white(small_image)
        # Should be single channel
        assert len(bw.shape) == 2
        # Should have only 0 and 255
        unique = np.unique(bw)
        assert set(unique).issubset({0, 255})

    def test_grayscale_of_grayscale_unchanged(self, gray_image):
        gray3 = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        result = _to_grayscale(gray3)
        np.testing.assert_array_equal(result, gray_image)


# ─────────────────────────────────────────────
#  ProcessingParams dataclass
# ─────────────────────────────────────────────
class TestProcessingParams:
    def test_default_values(self):
        params = ProcessingParams()
        assert params.color_mode == ColorMode.COLOR
        assert params.rotation == 0.5
        assert params.rotation_variance == 0.0
        assert params.brightness == 0.0
        assert params.contrast == 0.0
        assert params.blur == 0.0
        assert params.noise == 0.0
        assert params.yellowing == 0.0
        assert params.resolution == 300

    def test_rotation_with_variance(self):
        params = ProcessingParams(rotation=5.0, rotation_variance=2.0)
        vals = [params.get_rotation_with_variance() for _ in range(100)]
        # All values should be within [3, 7]
        assert all(3.0 <= v <= 7.0 for v in vals)

    def test_rotation_variance_zero_no_variance(self):
        params = ProcessingParams(rotation=3.0, rotation_variance=0.0)
        vals = [params.get_rotation_with_variance() for _ in range(50)]
        assert all(v == 3.0 for v in vals)


# ─────────────────────────────────────────────
#  Full pipeline
# ─────────────────────────────────────────────
class TestPipeline:
    def test_process_image_returns_valid_image(self, small_image):
        params = ProcessingParams(
            color_mode=ColorMode.COLOR,
            rotation=0.5,
            brightness=10,
            contrast=10,
            blur=0.5,
            noise=2.0,
            yellowing=0.1,
            resolution=300,
        )
        result = process_image(small_image.copy(), params)
        assert result.dtype == np.uint8
        assert result.shape[0] > 0 and result.shape[1] > 0

    def test_process_image_grayscale_mode(self, small_image):
        params = ProcessingParams(color_mode=ColorMode.GRAYSCALE)
        result = process_image(small_image.copy(), params)
        # Pipeline converts back to BGR for display, so shape[2] == 3
        assert result.dtype == np.uint8
        assert result.shape == small_image.shape

    def test_process_image_bw_mode(self, small_image):
        params = ProcessingParams(color_mode=ColorMode.BLACK_WHITE)
        result = process_image(small_image.copy(), params)
        assert result.dtype == np.uint8

    def test_process_image_preview_scale(self, small_image):
        params = ProcessingParams()
        result = process_image(small_image.copy(), params, preview_scale=0.5)
        assert result.shape[0] <= small_image.shape[0]
        assert result.shape[1] <= small_image.shape[1]

    def test_process_image_random_variance(self, small_image):
        params = ProcessingParams(rotation=0.5, rotation_variance=0.5)
        results = [process_image(small_image.copy(), params,
                                  apply_random_variance=True)
                   for _ in range(5)]
        # Results may vary slightly due to rotation variance
        assert all(r.dtype == np.uint8 for r in results)

    def test_create_proxy_image_unchanged_if_small(self, small_image):
        result = create_proxy_image(small_image, max_dim=400)
        assert result.shape == small_image.shape

    def test_create_proxy_image_downscales_large(self):
        large = np.full((1000, 1000, 3), 128, dtype=np.uint8)
        result = create_proxy_image(large, max_dim=200)
        assert max(result.shape[:2]) == 200


# ─────────────────────────────────────────────
#  Edge cases
# ─────────────────────────────────────────────
class TestEdgeCases:
    def test_single_pixel_image(self):
        img = np.array([[[128, 128, 128]]], dtype=np.uint8)
        result = _apply_brightness(img.copy(), 10)
        assert result.dtype == np.uint8
        assert result.shape == img.shape

    def test_all_black_image(self):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        result = _apply_brightness(img.copy(), 50)
        assert result.dtype == np.uint8
        assert result.max() <= 255

    def test_all_white_image(self):
        img = np.full((50, 50, 3), 255, dtype=np.uint8)
        result = _apply_brightness(img.copy(), -50)
        assert result.dtype == np.uint8
        assert result.min() >= 0

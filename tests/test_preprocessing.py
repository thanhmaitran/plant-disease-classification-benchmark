import numpy as np

from plant_disease.preprocessing import extract_hog, normalise
from plant_disease.robustness import apply, blur, brightness, gaussian_noise


def _imgs(n=2, size=96):
    return np.random.default_rng(0).integers(0, 255, (n, size, size, 3), dtype=np.uint8)


def test_hog_dimension_for_96px():
    assert extract_hog(_imgs()).shape == (2, 4356)


def test_normalise_range_and_dtype():
    x = normalise(_imgs())
    assert x.dtype == np.float32 and 0.0 <= x.min() and x.max() <= 1.0


def test_corruptions_keep_shape_and_dtype():
    x = _imgs()
    for out in (brightness(x, 1.2), gaussian_noise(x, 0.03), blur(x, 1)):
        assert out.shape == x.shape and out.dtype == np.uint8


def test_brightness_identity_and_darkening():
    x = _imgs()
    assert np.array_equal(brightness(x, 1.0), x)
    assert brightness(x, 0.5).mean() < x.mean()


def test_noise_is_seeded():
    x = _imgs()
    assert np.array_equal(gaussian_noise(x, 0.05, 3), gaussian_noise(x, 0.05, 3))


def test_blur_does_not_mix_images():
    x = _imgs(2)
    assert np.array_equal(blur(x[:1], 2)[0], blur(x, 2)[0])


def test_apply_unknown_name():
    import pytest
    with pytest.raises(KeyError):
        apply("rain", _imgs(), 1)

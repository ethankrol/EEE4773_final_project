"""Extract handcrafted LBP, HSV, RGB, HOG, and Gabor image features."""

import numpy as np
from skimage import color
from skimage.feature import hog, local_binary_pattern
from skimage.filters import gabor


def _to_rgb_float(image):
    """Convert one RGB image from uint8 or normalized float to float values in [0, 1]."""
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image /= 255.0
    return np.clip(image, 0.0, 1.0)

def extract_image_features(image, hog_orientations=9, hog_pixels_per_cell=(8, 8),
                           lbp_points=8, lbp_radius=1, lbp_grid=(1, 1),
                           color_bins=16, color_grid=(1, 1), include_gabor=False):
    """Extract HOG, spatial LBP, RGB/HSV histograms, and optional Gabor texture features."""
    rgb = _to_rgb_float(image)
    gray = color.rgb2gray(rgb)
    gray_uint8 = np.round(gray * 255).astype(np.uint8)
    features = {"hog": _extract_hog(gray, hog_orientations, hog_pixels_per_cell)}
    features["lbp"] = _extract_lbp(gray_uint8, lbp_points, lbp_radius, lbp_grid)
    features.update(_extract_color_features(rgb, color_bins, color_grid))
    if include_gabor:
        features["gabor"] = _extract_gabor(gray)
    return features


def _extract_hog(gray, orientations, pixels_per_cell):
    """Calculate the normalized HOG descriptor for one grayscale image."""
    return hog(gray, orientations=orientations, pixels_per_cell=pixels_per_cell,
               cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True).astype(np.float32)


def _extract_lbp(gray_uint8, points, radius, grid):
    """Calculate and concatenate normalized uniform LBP histograms across grid cells."""
    lbp = local_binary_pattern(gray_uint8, points, radius, method="uniform")
    bins = points + 2
    histograms = [_normalized_histogram(cell, bins, (0, bins))
                  for row in np.array_split(lbp, grid[0])
                  for cell in np.array_split(row, grid[1], axis=1)]
    return np.concatenate(histograms)


def _normalized_histogram(values, bins, value_range):
    """Return a float32 histogram normalized by its total count."""
    histogram, _ = np.histogram(values, bins=bins, range=value_range)
    histogram = histogram.astype(np.float32)
    return histogram / max(histogram.sum(), 1.0)


def _extract_color_features(rgb, bins, grid):
    """Calculate concatenated RGB and HSV channel histograms across grid cells."""
    spaces = {"rgb": rgb, "hsv": color.rgb2hsv(rgb)}
    features = {name: [] for name in spaces}
    for name, image in spaces.items():
        features[name] = _color_grid_histograms(image, bins, grid)
    return {name: np.concatenate(values) for name, values in features.items()}


def _color_grid_histograms(image, bins, grid):
    """Calculate normalized histograms for every color channel in each grid cell."""
    cells = [cell for row in np.array_split(image, grid[0])
             for cell in np.array_split(row, grid[1], axis=1)]
    return [_normalized_histogram(cell[..., channel], bins, (0.0, 1.0))
            for cell in cells for channel in range(3)]


def _extract_gabor(gray):
    """Calculate Gabor magnitude statistics and normalized histograms at fixed orientations."""
    responses = []
    for frequency in (0.1, 0.2, 0.3):
        for theta in (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4):
            magnitude = _gabor_magnitude(gray, frequency, theta)
            responses.extend(_gabor_summary(magnitude))
    return np.asarray(responses, dtype=np.float32)


def _gabor_magnitude(gray, frequency, theta):
    """Return the magnitude response for one fixed Gabor frequency and angle."""
    real, imag = gabor(gray, frequency=frequency, theta=theta)
    return np.hypot(real, imag)


def _gabor_summary(magnitude):
    """Summarize one Gabor magnitude response with its statistics and histogram."""
    upper = max(float(magnitude.max()), 1e-6)
    histogram, _ = np.histogram(magnitude, bins=8, range=(0.0, upper))
    histogram = histogram.astype(np.float32)
    normalized = histogram / max(histogram.sum(), 1.0)
    return [float(magnitude.mean()), float(magnitude.std()), *normalized.tolist()]

def extract_feature_dataset(images, feature_groups=("hog", "lbp", "hsv"), **feature_options):
    """Extract requested handcrafted feature groups for a batch of images."""
    rows = [extract_image_features(image, **feature_options) for image in images]
    return {name: np.stack([row[name] for row in rows]) for name in feature_groups}

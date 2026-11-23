"""Hand-made colour + texture features for the random-forest baseline.

Per image (resized to 256x256) we compute:
  - HSV colour histograms: hue 18 bins, saturation 8, value 8      -> 34 numbers
  - LBP texture histogram (uniform, 8 neighbours, radius 1)         -> 10 numbers
  - GLCM texture: 5 properties x 2 distances (mean over 4 angles)   -> 10 numbers
Total: 54 numbers per image.

Features are cached in data/cache/features.csv so they are computed only once.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

ROOT = Path(__file__).resolve().parent.parent
CACHE_FILE = ROOT / "data/cache/features.csv"
IMG_SIZE = 256

HSV_BINS = {"h": 18, "s": 8, "v": 8}
LBP_POINTS, LBP_RADIUS = 8, 1
GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
GLCM_DISTANCES = [1, 3]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]


def colour_histogram(img):
    """Normalised histogram of each HSV channel."""
    hsv = np.array(img.convert("HSV"))
    parts = []
    for channel, bins in enumerate(HSV_BINS.values()):
        hist, _ = np.histogram(hsv[:, :, channel], bins=bins, range=(0, 256))
        parts.append(hist / hist.sum())
    return np.concatenate(parts)


def lbp_histogram(gray):
    """Local Binary Pattern histogram: describes small-scale texture (spots, veins)."""
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
    n_bins = LBP_POINTS + 2
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins))
    return hist / hist.sum()


def glcm_features(gray):
    """Grey-level co-occurrence statistics (contrast, smoothness, ...)."""
    glcm = graycomatrix(gray, GLCM_DISTANCES, GLCM_ANGLES, levels=256,
                        symmetric=True, normed=True)
    # graycoprops gives (distances x angles); average over the angles
    return np.concatenate([graycoprops(glcm, p).mean(axis=1) for p in GLCM_PROPS])


def extract_features(path):
    """All 54 features for one image file."""
    with Image.open(path) as img:
        img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    gray = np.array(img.convert("L"), dtype=np.uint8)
    return np.concatenate([colour_histogram(img), lbp_histogram(gray), glcm_features(gray)])


def feature_names():
    """Column names, in the same order as extract_features()."""
    names = [f"hsv_{ch}_{i}" for ch, bins in HSV_BINS.items() for i in range(bins)]
    names += [f"lbp_{i}" for i in range(LBP_POINTS + 2)]
    names += [f"glcm_{p}_d{d}" for p in GLCM_PROPS for d in GLCM_DISTANCES]
    return names


def features_for(paths):
    """Feature matrix for a list of relative paths, using/updating the cache."""
    cache = pd.read_csv(CACHE_FILE, index_col="path") if CACHE_FILE.exists() else None
    missing = [p for p in paths if cache is None or p not in cache.index]
    if missing:
        print(f"  computing features for {len(missing)} images ...")
        rows = []
        for i, p in enumerate(missing):
            rows.append(extract_features(ROOT / p))
            if (i + 1) % 250 == 0:
                print(f"    {i + 1}/{len(missing)}")
        new = pd.DataFrame(rows, index=pd.Index(missing, name="path"), columns=feature_names())
        cache = new if cache is None else pd.concat([cache, new])
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        cache.to_csv(CACHE_FILE)
    return cache.loc[list(paths)].to_numpy()

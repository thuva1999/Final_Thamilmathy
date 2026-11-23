"""Step 3a: make train / validation / test splits without duplicate leakage.

Uses data/image_index.csv from Step 2 (run src/explore_data.py first).

Rules:
  - Near-duplicate photos (pHash distance <= 4) form one "group".
  - A group whose photos carry DIFFERENT class labels is a label conflict:
    all its photos are removed (we cannot know which label is right).
  - A whole group always goes into the same split, so no photo has a copy
    on the other side (no leakage).
  - SAR-CLD-2024: stratified 70 / 15 / 15 split of the groups, seed 42.
  - EWU (external test only): conflicts removed, one photo kept per group.

Outputs:
  data/splits/train.csv, val.csv, test.csv   (SAR-CLD-2024)
  data/splits/ewu_external.csv               (EWU, all 5 classes, + shared flag)
  data/splits/removed.csv                    (every removed photo + reason)
  data/class_mapping.csv                     (which classes the datasets share)
  results/split_counts.csv                   (images per class and split)

Run from the project folder:  python src/make_splits.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import train_test_split

from explore_data import NEAR_DUP_MAX_DIST, hash_bits

ROOT = Path(__file__).resolve().parent.parent
INDEX_FILE = ROOT / "data/image_index.csv"
SPLITS = ROOT / "data/splits"
RESULTS = ROOT / "results"
SEED = 42


def hamming_matrix(df):
    """Number of different hash bits for every pair of images in df."""
    bits = hash_bits(df)
    return bits @ (1 - bits).T + (1 - bits) @ bits.T


def add_groups(df):
    """Give near-duplicate photos the same group id (connected components)."""
    close = hamming_matrix(df) <= NEAR_DUP_MAX_DIST
    _, labels = connected_components(csr_matrix(close), directed=False)
    df = df.copy()
    df["group"] = labels
    return df


def remove_conflicts(df):
    """Drop every group that contains more than one class. Returns (kept, removed)."""
    classes_per_group = df.groupby("group")["class"].nunique()
    bad_groups = classes_per_group[classes_per_group > 1].index
    is_bad = df["group"].isin(bad_groups)
    removed = df[is_bad].assign(reason="same photo in different classes")
    return df[~is_bad], removed


def split_groups(df):
    """Stratified 70/15/15 split of groups (by class), then map back to photos."""
    groups = df.groupby("group")["class"].first().reset_index()
    train_g, rest_g = train_test_split(groups, test_size=0.30,
                                       stratify=groups["class"], random_state=SEED)
    val_g, test_g = train_test_split(rest_g, test_size=0.50,
                                     stratify=rest_g["class"], random_state=SEED)
    split_of_group = {}
    for name, part in [("train", train_g), ("val", val_g), ("test", test_g)]:
        split_of_group.update({g: name for g in part["group"]})
    df = df.copy()
    df["split"] = df["group"].map(split_of_group)
    return df


def check_no_leakage(df):
    """Make sure no near-duplicate pair crosses two different splits."""
    dist = hamming_matrix(df)
    splits = df["split"].to_numpy()
    close = dist <= NEAR_DUP_MAX_DIST
    crossing = close & (splits[:, None] != splits[None, :])
    n_cross = int(np.triu(crossing, k=1).sum())
    print(f"  near-duplicate pairs across splits: {n_cross} (must be 0)")
    assert n_cross == 0, "leakage between splits!"


def make_sar_splits(index):
    """SAR-CLD-2024: groups -> remove conflicts -> stratified group split."""
    sar = add_groups(index[index["dataset"] == "SAR-CLD-2024"].reset_index(drop=True))
    sar, removed = remove_conflicts(sar)
    sar = split_groups(sar.reset_index(drop=True))
    check_no_leakage(sar)
    for name in ["train", "val", "test"]:
        part = sar[sar["split"] == name][["path", "class", "group"]]
        part.to_csv(SPLITS / f"{name}.csv", index=False)
    return sar, removed


def make_ewu_external(index, sar_classes):
    """EWU: remove conflicts, keep one photo per duplicate group, mark shared classes."""
    ewu = add_groups(index[index["dataset"] == "EWU"].reset_index(drop=True))
    ewu, conflicts = remove_conflicts(ewu)
    keep = ewu.drop_duplicates("group", keep="first")
    extra_copies = ewu[~ewu.index.isin(keep.index)].assign(reason="extra copy of the same photo")
    keep = keep.assign(shared_with_sar=keep["class"].isin(sar_classes))
    keep[["path", "class", "shared_with_sar"]].to_csv(SPLITS / "ewu_external.csv", index=False)
    return keep, pd.concat([conflicts, extra_copies])


def write_class_mapping(sar_classes, ewu_classes):
    """Table of every class and its name in each dataset (blank = not present)."""
    rows = []
    for name in sorted(set(sar_classes) | set(ewu_classes)):
        in_sar, in_ewu = name in sar_classes, name in ewu_classes
        rows.append({"class": name,
                     "sar_cld_2024_folder": name if in_sar else "",
                     "ewu_folder": name if in_ewu else "",
                     "used_in_external_test": in_sar and in_ewu})
    mapping = pd.DataFrame(rows)
    mapping.to_csv(ROOT / "data/class_mapping.csv", index=False)
    return mapping


def main():
    SPLITS.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(INDEX_FILE)
    sar_classes = sorted(index[index["dataset"] == "SAR-CLD-2024"]["class"].unique())
    ewu_classes = sorted(index[index["dataset"] == "EWU"]["class"].unique())

    print("SAR-CLD-2024 ...")
    sar, sar_removed = make_sar_splits(index)
    print("EWU ...")
    ewu, ewu_removed = make_ewu_external(index, sar_classes)

    removed = pd.concat([sar_removed, ewu_removed])
    removed[["dataset", "class", "path", "reason"]].to_csv(SPLITS / "removed.csv", index=False)

    mapping = write_class_mapping(sar_classes, ewu_classes)

    counts = pd.crosstab(sar["class"], sar["split"])[["train", "val", "test"]]
    counts.loc["TOTAL"] = counts.sum()
    counts.to_csv(RESULTS / "split_counts.csv")

    print("\nSAR-CLD-2024 images per split:")
    print(counts.to_string())
    print(f"\nRemoved photos: {len(removed)}")
    print(removed.groupby(["dataset", "reason"]).size().to_string())
    print(f"\nEWU external test: {len(ewu)} photos "
          f"({ewu['shared_with_sar'].sum()} in classes shared with SAR-CLD-2024)")
    print(ewu.groupby("class").size().to_string())
    print("\nClass mapping (data/class_mapping.csv):")
    print(mapping.to_string(index=False))


if __name__ == "__main__":
    main()

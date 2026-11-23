"""Step 2: explore the three datasets for the proposal's Data section.

What it does (all outputs go to results/, plus one index file in data/):
  1. class counts per image dataset          -> class_counts.csv, class_counts.png
  2. one example photo per class             -> example_grid.png
  3. image sizes (width x height)            -> image_sizes.csv
  4. duplicate / near-duplicate check (pHash) -> duplicate_pairs.csv, duplicate_examples.png
  5. first rows of the treatment table       -> treatment_first_rows.csv
  + data/image_index.csv: every image with class, size and hash (used again in Step 3)
  + results/data_summary.txt: the key numbers in plain text

Run from the project folder:  python src/explore_data.py
"""
from pathlib import Path

import imagehash
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATASETS = {
    "SAR-CLD-2024": ROOT / "data/raw/sarcld2024/Original Dataset",
    "EWU": ROOT / "data/raw/ewu_cotton/Cotton_Original_Dataset",
}
TREATMENT_DIR = ROOT / "data/raw/treatment"
RESULTS = ROOT / "results"
INDEX_FILE = ROOT / "data/image_index.csv"

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp"}
NEAR_DUP_MAX_DIST = 4   # pHash has 64 bits; <= 4 different bits = "looks the same"
SEED = 42

# Chart style (one blue for bars, dark grey text, light hairline grid)
BAR_COLOR = "#2a78d6"
TEXT_COLOR = "#0b0b0b"
MUTED_COLOR = "#52514e"
GRID_COLOR = "#e4e3df"


# ---------- 1. Build an index of all images ----------

def list_images():
    """One row per image: dataset, class (= folder name), path relative to the project."""
    rows = []
    for dataset, folder in DATASETS.items():
        for class_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
            for f in sorted(class_dir.iterdir()):
                if f.suffix.lower() in IMAGE_TYPES:
                    rows.append({"dataset": dataset, "class": class_dir.name,
                                 "path": f.relative_to(ROOT).as_posix()})
    return pd.DataFrame(rows)


def add_size_and_hash(df):
    """Open every image once: read its size and compute its perceptual hash."""
    widths, heights, hashes = [], [], []
    for i, rel_path in enumerate(df["path"]):
        with Image.open(ROOT / rel_path) as img:
            widths.append(img.width)
            heights.append(img.height)
            hashes.append(str(imagehash.phash(img)))
        if (i + 1) % 500 == 0:
            print(f"  hashed {i + 1}/{len(df)} images")
    df = df.copy()
    df["width"], df["height"], df["phash"] = widths, heights, hashes
    return df


# ---------- 2. Class counts ----------

def class_counts(df):
    """Table: number of images per class in each dataset."""
    counts = df.groupby(["dataset", "class"]).size().rename("images").reset_index()
    counts.to_csv(RESULTS / "class_counts.csv", index=False)
    return counts


def style_axis(ax):
    """Recessive axes: no box, light vertical grid, grey tick labels."""
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED_COLOR, length=0)


def plot_class_counts(counts):
    """Horizontal bar chart per dataset, largest class on top, count at the bar tip."""
    datasets = list(DATASETS)
    sizes = [len(counts[counts["dataset"] == d]) for d in datasets]
    fig, axes = plt.subplots(len(datasets), 1, figsize=(8, 1.2 + 0.45 * sum(sizes)),
                             gridspec_kw={"height_ratios": sizes}, sharex=True)
    max_count = counts["images"].max()
    for ax, dataset in zip(axes, datasets):
        part = counts[counts["dataset"] == dataset].sort_values("images")
        ax.barh(part["class"], part["images"], height=0.55, color=BAR_COLOR)
        for y, value in enumerate(part["images"]):
            ax.text(value + max_count * 0.01, y, f"{value}", va="center",
                    color=TEXT_COLOR, fontsize=9)
        ax.set_title(f"{dataset} (original images, n = {part['images'].sum():,})",
                     loc="left", color=TEXT_COLOR, fontsize=11)
        ax.tick_params(axis="y", labelcolor=TEXT_COLOR)
        style_axis(ax)
    axes[-1].set_xlabel("Number of images", color=MUTED_COLOR)
    axes[-1].set_xlim(0, max_count * 1.12)
    fig.tight_layout()
    fig.savefig(RESULTS / "class_counts.png", dpi=150)
    plt.close(fig)


# ---------- 3. Example grid ----------

def plot_example_grid(df):
    """One random (but fixed-seed) photo per class; one row per dataset."""
    datasets = list(DATASETS)
    n_cols = df.groupby("dataset")["class"].nunique().max()
    fig, axes = plt.subplots(len(datasets), n_cols, figsize=(2.2 * n_cols, 2.7 * len(datasets)))
    for row, dataset in enumerate(datasets):
        part = df[df["dataset"] == dataset]
        examples = part.groupby("class").sample(1, random_state=SEED)
        for col in range(n_cols):
            ax = axes[row, col]
            ax.axis("off")
            if col >= len(examples):
                continue
            item = examples.iloc[col]
            with Image.open(ROOT / item["path"]) as img:
                ax.imshow(img.convert("RGB"))
            ax.set_title(item["class"].replace(" ", "\n", 1), fontsize=9, color=TEXT_COLOR)
        axes[row, 0].text(-0.08, 0.5, dataset, transform=axes[row, 0].transAxes,
                          rotation=90, ha="right", va="center", fontsize=11, color=TEXT_COLOR)
    fig.tight_layout()
    fig.savefig(RESULTS / "example_grid.png", dpi=150)
    plt.close(fig)


# ---------- 4. Image sizes ----------

def image_sizes(df):
    """Table: how many images have each width x height, per dataset."""
    sizes = (df.groupby(["dataset", "width", "height"]).size().rename("images")
             .reset_index().sort_values(["dataset", "images"], ascending=[True, False]))
    sizes.to_csv(RESULTS / "image_sizes.csv", index=False)
    return sizes


# ---------- 5. Duplicate check ----------

def hash_bits(df):
    """Turn each 64-bit hex hash into a row of 0/1 numbers."""
    return np.array([imagehash.hex_to_hash(h).hash.flatten() for h in df["phash"]],
                    dtype=np.float32)


def find_duplicates(df):
    """All image pairs whose hashes differ in <= NEAR_DUP_MAX_DIST bits."""
    bits = hash_bits(df)
    # Hamming distance for all pairs at once: count of bits that differ
    dist = bits @ (1 - bits).T + (1 - bits) @ bits.T
    i_idx, j_idx = np.where(np.triu(dist <= NEAR_DUP_MAX_DIST, k=1))
    rows = []
    for i, j in zip(i_idx, j_idx):
        a, b = df.iloc[i], df.iloc[j]
        scope = (f"within {a['dataset']}" if a["dataset"] == b["dataset"]
                 else "across datasets")
        rows.append({
            "scope": scope,
            "label": "same class" if a["class"] == b["class"] else "DIFFERENT class",
            "distance": int(dist[i, j]),
            "class_a": a["class"], "path_a": a["path"],
            "class_b": b["class"], "path_b": b["path"],
        })
    pairs = pd.DataFrame(rows, columns=["scope", "label", "distance", "class_a",
                                        "path_a", "class_b", "path_b"])
    pairs = pairs.sort_values(["scope", "distance"])
    pairs.to_csv(RESULTS / "duplicate_pairs.csv", index=False)
    return pairs


def plot_duplicate_examples(pairs, max_pairs=6):
    """Show a few duplicate pairs side by side (different-class pairs first)."""
    if pairs.empty:
        return
    shown = pairs.sort_values(["label", "distance"]).head(max_pairs)
    fig, axes = plt.subplots(len(shown), 2, figsize=(5, 2.4 * len(shown)), squeeze=False)
    for row, (_, pair) in enumerate(shown.iterrows()):
        for col, side in enumerate(["a", "b"]):
            ax = axes[row, col]
            with Image.open(ROOT / pair[f"path_{side}"]) as img:
                ax.imshow(img.convert("RGB"))
            ax.set_title(f"{pair[f'class_{side}']}\n{Path(pair[f'path_{side}']).name}",
                         fontsize=8, color=TEXT_COLOR)
            ax.axis("off")
        axes[row, 0].text(-0.05, 0.5, f"distance {pair['distance']}",
                          transform=axes[row, 0].transAxes, rotation=90,
                          ha="right", va="center", fontsize=8, color=MUTED_COLOR)
    fig.tight_layout()
    fig.savefig(RESULTS / "duplicate_examples.png", dpi=150)
    plt.close(fig)


# ---------- 6. Treatment table ----------

def load_treatment():
    """Read the treatment CSV and drop the empty separator rows."""
    csv_file = next(TREATMENT_DIR.rglob("*.csv"))
    table = pd.read_csv(csv_file, encoding_errors="replace").dropna(how="all")
    table["Disease"] = table["Disease"].str.strip()
    table.head(10).to_csv(RESULTS / "treatment_first_rows.csv", index=False)
    return table


# ---------- Summary ----------

def write_summary(df, counts, sizes, pairs, table):
    """Collect the key numbers in one text file (and print them)."""
    lines = ["CottonCare - data exploration summary", ""]
    for dataset in DATASETS:
        part = counts[counts["dataset"] == dataset]
        biggest, smallest = part.loc[part["images"].idxmax()], part.loc[part["images"].idxmin()]
        lines += [f"{dataset}: {part['images'].sum()} images, {len(part)} classes",
                  f"  largest class:  {biggest['class']} ({biggest['images']})",
                  f"  smallest class: {smallest['class']} ({smallest['images']})",
                  f"  imbalance ratio: {biggest['images'] / smallest['images']:.1f}x"]
        top = sizes[sizes["dataset"] == dataset].head(3)
        size_text = ", ".join(f"{w}x{h} ({n})" for w, h, n in
                              zip(top["width"], top["height"], top["images"]))
        n_sizes = len(sizes[sizes["dataset"] == dataset])
        lines += [f"  image sizes: {n_sizes} different; most common: {size_text}", ""]

    shared = sorted(set(df[df["dataset"] == "SAR-CLD-2024"]["class"])
                    & set(df[df["dataset"] == "EWU"]["class"]))
    lines += [f"Classes in both image datasets: {', '.join(shared)}", ""]

    lines += [f"Near-duplicate pairs (pHash distance <= {NEAR_DUP_MAX_DIST}): {len(pairs)}"]
    if not pairs.empty:
        for (scope, label), group in pairs.groupby(["scope", "label"]):
            lines.append(f"  {scope}, {label}: {len(group)} pairs "
                         f"({(group['distance'] == 0).sum()} with distance 0)")
        involved = set(pairs["path_a"]) | set(pairs["path_b"])
        lines.append(f"  images involved: {len(involved)}")
    lines.append("")

    sar_classes = set(df[df["dataset"] == "SAR-CLD-2024"]["class"])
    diseases = list(dict.fromkeys(table["Disease"].dropna()))
    lines += [f"Treatment table: {len(table)} rows, {len(diseases)} diseases",
              f"  features: {', '.join(table['Feature'].dropna().unique())}",
              f"  SAR-CLD classes found in table: {sorted(sar_classes & set(diseases))}",
              f"  SAR-CLD classes NOT in table:   {sorted(sar_classes - set(diseases))}"]

    text = "\n".join(lines)
    (RESULTS / "data_summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text)


def main():
    RESULTS.mkdir(exist_ok=True)
    print("Listing images ...")
    df = list_images()
    print(f"  found {len(df)} images. Reading sizes and hashing ...")
    df = add_size_and_hash(df)
    df.to_csv(INDEX_FILE, index=False)
    print("First rows of the image index:")
    print(df.head().to_string(index=False))

    counts = class_counts(df)
    plot_class_counts(counts)
    plot_example_grid(df)
    sizes = image_sizes(df)
    print("Checking duplicates ...")
    pairs = find_duplicates(df)
    plot_duplicate_examples(pairs)
    table = load_treatment()
    print("\nFirst rows of the treatment table:")
    print(table.head(4).to_string(index=False, max_colwidth=50))
    write_summary(df, counts, sizes, pairs, table)
    print(f"\nSaved results to {RESULTS} and the image index to {INDEX_FILE}")


if __name__ == "__main__":
    main()

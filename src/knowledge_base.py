"""Step 6a: knowledge base for the advice (RAG retrieval part).

Source: "Cotton disease treatment dataset", Mendeley Data 2025, DOI 10.17632/533j2mzd4s.1
(table with columns Disease, Feature, Description).

What this module does:
  - cleans the raw table (blank rows, typos in feature names) and saves
    data/knowledge_base.csv, which the app uses (the raw file stays in data/raw)
  - groups each feature into one of the advice parts:
        what_it_is   : Symptoms, Favorable_Conditions
        do_now       : physical, biological and chemical measures
        prevent      : Preventive_measures
  - retrieve(class): exact mapping classifier class -> rows (first choice);
    sentence-embedding search over the table as a backup for unknown names

Run from the project folder:  python src/knowledge_base.py   (builds + tests it)
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data/raw/treatment"
KB_FILE = ROOT / "data/knowledge_base.csv"

SOURCE = ("Cotton disease treatment dataset, Mendeley Data (2025), "
          "DOI 10.17632/533j2mzd4s.1")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MIN_SIMILARITY = 0.35   # embedding backup: below this, no match is returned

# Fix the spelling of feature names in the raw table
FEATURE_FIXES = {
    "Hebicides": "Herbicides",
    "Fungisides": "Fungicides",
    "Preventive_measeures": "Preventive_measures",
    "Biological": "Biological_measures",
}

# Which advice part each feature belongs to
FEATURE_PART = {
    "Symptoms": "what_it_is",
    "Favorable_Conditions": "what_it_is",
    "Physical_measures": "do_now",
    "Biological_measures": "do_now",
    "Bactericides": "do_now",
    "Fungicides": "do_now",
    "Insecticides": "do_now",
    "Pesticides": "do_now",
    "Herbicides": "do_now",
    "Preventive_measures": "prevent",
}

# Classifier class -> disease name in the table. None = no table rows (healthy leaf).
CLASS_TO_DISEASE = {
    "Bacterial Blight": "Bacterial Blight",
    "Curl Virus": "Curl Virus",
    "Healthy Leaf": None,
    "Herbicide Growth Damage": "Herbicide Growth Damage",
    "Leaf Hopper Jassids": "Leaf Hopper Jassids",
    "Leaf Redding": "Leaf Redding",
    "Leaf Variegation": "Leaf Variegation",
}


# ---------- Building the cleaned table ----------

def clean_text(text):
    """Tidy one Description: one item per line, no empty lines or extra spaces."""
    lines = [" ".join(line.split()) for line in str(text).splitlines()]
    return "\n".join(line for line in lines if line)


def build_knowledge_base():
    """Read the raw Mendeley table, clean it and save data/knowledge_base.csv."""
    raw_file = next(RAW_DIR.rglob("*.csv"))
    table = pd.read_csv(raw_file, encoding_errors="replace").dropna(how="all")
    table["disease"] = table["Disease"].str.strip()
    table["feature"] = table["Feature"].str.strip().replace(FEATURE_FIXES)
    table["part"] = table["feature"].map(FEATURE_PART)
    table["text"] = table["Description"].map(clean_text)
    unknown = table[table["part"].isna()]["feature"].unique()
    assert len(unknown) == 0, f"feature names without an advice part: {unknown}"
    table["row_id"] = [f"KB{i + 1:03d}" for i in range(len(table))]
    kb = table[["row_id", "disease", "feature", "part", "text"]]
    kb.to_csv(KB_FILE, index=False, encoding="utf-8")
    return kb


@lru_cache(maxsize=1)
def load_knowledge_base():
    """The cleaned table (built from the raw file the first time)."""
    if not KB_FILE.exists():
        return build_knowledge_base()
    return pd.read_csv(KB_FILE, encoding="utf-8")


# ---------- Retrieval ----------

def rows_for_disease(disease):
    """All table rows of one disease, in table order."""
    kb = load_knowledge_base()
    return kb[kb["disease"] == disease]


@lru_cache(maxsize=1)
def _embedder():
    """Small English sentence-embedding model (downloaded once, ~90 MB)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _disease_vectors():
    """One embedding per disease: its name + symptoms."""
    kb = load_knowledge_base()
    diseases = list(dict.fromkeys(kb["disease"]))
    texts = [d + ". " + " ".join(kb[(kb["disease"] == d) & (kb["feature"] == "Symptoms")]["text"])
             for d in diseases]
    vectors = _embedder().encode(texts, normalize_embeddings=True)
    return diseases, vectors


def search_disease(query):
    """Backup: the disease whose name + symptoms are closest to the query text."""
    diseases, vectors = _disease_vectors()
    q = _embedder().encode([query], normalize_embeddings=True)[0]
    scores = vectors @ q
    best = int(np.argmax(scores))
    return diseases[best], float(scores[best])


def retrieve(class_name):
    """Rows for a classifier class.
    Returns (rows, method): method is 'exact', 'embedding', 'healthy' or 'none'."""
    if class_name in CLASS_TO_DISEASE:
        disease = CLASS_TO_DISEASE[class_name]
        if disease is None:
            return rows_for_disease("__none__"), "healthy"
        return rows_for_disease(disease), "exact"
    disease, score = search_disease(class_name)
    if score < MIN_SIMILARITY:
        return rows_for_disease("__none__"), "none"
    return rows_for_disease(disease), "embedding"


def symptoms_of(class_name):
    """Only the Symptoms text of a class (used for 'how to tell A and B apart')."""
    rows, _ = retrieve(class_name)
    return "\n".join(rows[rows["feature"] == "Symptoms"]["text"])


def main():
    kb = build_knowledge_base()
    print(f"Saved {KB_FILE.relative_to(ROOT)}: {len(kb)} rows, "
          f"{kb['disease'].nunique()} diseases")
    print(kb.groupby("part").size().to_string())

    print("\nExact mapping (classifier class -> rows):")
    for class_name in CLASS_TO_DISEASE:
        rows, method = retrieve(class_name)
        print(f"  {class_name:25s} {method:8s} {len(rows)} rows")

    print("\nEmbedding backup (names that are not classifier classes):")
    for query in ["whitefly leaf curling", "white powder on leaves",
                  "Fusarium Wilt", "red leaf edges magnesium"]:
        disease, score = search_disease(query)
        print(f"  {query!r:32s} -> {disease} (similarity {score:.2f})")


if __name__ == "__main__":
    main()
